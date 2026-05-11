"""Local detector backend — runs the privacy-filter model in-process via transformers."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Annotated, Any

from .base import Span, trim_span

DEFAULT_MODEL = "openai/privacy-filter"

# Default sliding-window chunking parameters. The model's positional limit is
# 512 tokens; reserving 2 for [CLS]/[SEP] leaves 510 usable. The overlap
# exists so a PII span that straddles two chunks is fully observed in at
# least one of them — 32 tokens covers any realistic single-entity surface
# form (the longest plausible PII string is a multi-segment address at
# ~25 tokens; 32 leaves headroom).
_DEFAULT_CHUNK_TOKENS = 510
_DEFAULT_OVERLAP_TOKENS = 32


@dataclass
class LocalDetector:
    """Wraps a HuggingFace token-classification pipeline.

    Lazy: model weights are loaded on first ``detect`` call. Thread-safe load.

    Long inputs (more tokens than the model's positional limit, typically
    512 for BERT/XLM-R/roberta encoders) are split into overlapping
    sliding windows; spans from each window are reprojected to the
    original character offsets and deduplicated. Without this, anything
    past the first ~512 tokens would be silently truncated by the
    underlying ``transformers`` pipeline.

    ``device``:
      - ``"auto"`` (default): cuda if available, else cpu
      - ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...

    ``dtype`` is left to the model default unless explicitly passed.
    """

    model_id: Annotated[
        str, "HuggingFace model id to load (token-classification head)."
    ] = DEFAULT_MODEL
    device: Annotated[
        str, "'auto', 'cpu', 'cuda', or a specific cuda index like 'cuda:0'."
    ] = "auto"
    dtype: Annotated[Any, "Optional torch dtype override (e.g. torch.bfloat16)."] = None
    aggregation_strategy: Annotated[
        str, "transformers token-classification aggregation strategy."
    ] = "simple"
    trim_whitespace: Annotated[
        bool, "Strip leading/trailing whitespace from each span (BPE artifact fix)."
    ] = True
    chunk_tokens: Annotated[
        int,
        "Sliding-window size in model tokens. Defaults to 510 (assumes a "
        "512-token model with 2 special tokens reserved). Must be > overlap.",
    ] = _DEFAULT_CHUNK_TOKENS
    chunk_overlap_tokens: Annotated[
        int,
        "Token overlap between consecutive windows. Must exceed the longest "
        "PII span you expect to detect; default 32 covers all realistic cases.",
    ] = _DEFAULT_OVERLAP_TOKENS
    batch_size: Annotated[
        int,
        "Forward-pass batch size used both for within-input chunk batching "
        "and for ``detect_batch`` cross-input batching. Default 8 is "
        "memory-safe on CPU; raise on GPU.",
    ] = 8
    _pipeline: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import torch
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                    pipeline,
                )
            except ImportError as e:  # pragma: no cover - import guard
                raise ImportError(
                    "LocalDetector requires `torch` and `transformers`. "
                    "Install with: uv add 'gheim[local]'"
                ) from e

            resolved_device = (
                "cuda" if (self.device == "auto" and torch.cuda.is_available())
                else "cpu" if self.device == "auto"
                else self.device
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            kwargs: dict[str, Any] = {}
            if self.dtype is not None:
                kwargs["dtype"] = self.dtype
            model = AutoModelForTokenClassification.from_pretrained(self.model_id, **kwargs)
            model.to(resolved_device)
            model.eval()

            pipe_device = -1 if resolved_device == "cpu" else 0
            self._pipeline = pipeline(
                task="token-classification",
                model=model,
                tokenizer=tokenizer,
                device=pipe_device,
                aggregation_strategy=self.aggregation_strategy,
            )
            return self._pipeline

    def detect(
        self,
        text: Annotated[str, "Source text to scan for PII."],
        *,
        model: Annotated[str | None, "Per-call model override; must equal self.model_id if set."] = None,
    ) -> list[Span]:
        if model is not None and model != self.model_id:
            raise ValueError(
                f"LocalDetector was constructed with model={self.model_id!r}; "
                f"per-call model override ({model!r}) not supported."
            )
        if not text:
            return []
        results = self.detect_batch([text])
        return results[0]

    def detect_batch(
        self,
        texts: Annotated[list[str], "List of source texts to scan in one batched pass."],
        *,
        model: Annotated[str | None, "Per-call model override; must equal self.model_id if set."] = None,
    ) -> list[list[Span]]:
        """Detect PII in many inputs in a single batched forward pass.

        Texts that exceed the model's token limit are split into overlapping
        sliding windows; all chunks across all inputs are batched together
        for the GPU/CPU pass, and the resulting spans are reprojected and
        deduplicated per input. Empty texts produce empty output lists at
        the matching position.
        """
        if model is not None and model != self.model_id:
            raise ValueError(
                f"LocalDetector was constructed with model={self.model_id!r}; "
                f"per-call model override ({model!r}) not supported."
            )
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be "
                f"smaller than chunk_tokens ({self.chunk_tokens})"
            )
        if not texts:
            return []
        pipe = self._load()
        # Build the flat list of (input_index, char_offset, chunk_text)
        # tuples across all inputs, then run a single batched call.
        plan: list[tuple[int, int, str]] = []
        for input_idx, text in enumerate(texts):
            if not text:
                continue
            offsets = _token_char_offsets(pipe.tokenizer, text)
            if len(offsets) <= self.chunk_tokens:
                plan.append((input_idx, 0, text))
                continue
            stride = self.chunk_tokens - self.chunk_overlap_tokens
            n_tokens = len(offsets)
            for win_start in range(0, n_tokens, stride):
                win_end = min(win_start + self.chunk_tokens, n_tokens)
                char_start = offsets[win_start][0]
                char_end = offsets[win_end - 1][1]
                plan.append((input_idx, char_start, text[char_start:char_end]))
                if win_end >= n_tokens:
                    break

        out: list[list[Span]] = [[] for _ in texts]
        if not plan:
            return out

        # Batched forward pass. The HF pipeline accepts a list and returns
        # a list-of-lists in the same order; ``batch_size`` limits how many
        # are tokenized + run in one tensor.
        chunk_texts = [t for _, _, t in plan]
        raw_batches = pipe(chunk_texts, batch_size=self.batch_size)

        # Per-input dedup sets keyed on (start, end, label).
        seen: list[set[tuple[int, int, str]]] = [set() for _ in texts]
        for (input_idx, char_offset, _), raw in zip(plan, raw_batches, strict=True):
            full_text = texts[input_idx]
            for r in raw:
                start = int(r["start"]) + char_offset
                end = int(r["end"]) + char_offset
                if self.trim_whitespace:
                    trimmed = trim_span(full_text, start, end)
                    if trimmed is None:
                        continue
                    start, end = trimmed
                label = str(r.get("entity_group") or r.get("entity"))
                key = (start, end, label)
                if key in seen[input_idx]:
                    continue
                seen[input_idx].add(key)
                out[input_idx].append(Span(
                    label=label,
                    start=start,
                    end=end,
                    text=full_text[start:end],
                    score=float(r.get("score", 1.0)),
                ))

        for spans in out:
            spans.sort(key=lambda s: (s.start, s.end))
        return out


def _token_char_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    """Return a list of (char_start, char_end) for each non-special token in ``text``.

    Uses the tokenizer's offset mapping with truncation disabled so the
    full input is represented; special tokens are excluded so the returned
    list maps 1:1 to model positions inside a window.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    raw = enc["offset_mapping"]
    # Filter out (0, 0) entries — fast tokenizers sometimes emit them for
    # control characters that produce no surface; they would corrupt the
    # window slicing if kept.
    out: list[tuple[int, int]] = []
    for a, b in raw:
        a_i, b_i = int(a), int(b)
        if a_i == 0 and b_i == 0:
            continue
        out.append((a_i, b_i))
    return out
