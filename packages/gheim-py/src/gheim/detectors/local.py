"""Local detector backend — runs the privacy-filter model in-process via transformers."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Annotated, Any

from .base import Span, trim_span

DEFAULT_MODEL = "openai/privacy-filter"


@dataclass
class LocalDetector:
    """Wraps a HuggingFace token-classification pipeline.

    Lazy: model weights are loaded on first ``detect`` call. Thread-safe load.

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
        pipe = self._load()
        raw = pipe(text)
        out: list[Span] = []
        for r in raw:
            span = _to_span(text, r, trim=self.trim_whitespace)
            if span is not None:
                out.append(span)
        return out


def _to_span(source: str, raw: dict[str, Any], *, trim: bool) -> Span | None:
    start = int(raw["start"])
    end = int(raw["end"])
    if trim:
        trimmed = trim_span(source, start, end)
        if trimmed is None:
            return None
        start, end = trimmed
    return Span(
        label=str(raw.get("entity_group") or raw.get("entity")),
        start=start,
        end=end,
        text=source[start:end],
        score=float(raw.get("score", 1.0)),
    )
