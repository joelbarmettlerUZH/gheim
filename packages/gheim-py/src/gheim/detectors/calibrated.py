"""Calibrated detector — subtracts a constant `o_bias` from the O-class
logit before argmax. Originally introduced to recover the v1 model's
"single-category mode" pathology (Müller + IBAN co-occurrence flipped the
person prediction to O); the v3 / gheim-ch-560m training pipeline
(Geonames gazetteer + overlap resolver + multi-LLM consensus labelling)
largely resolved that pathology, but the calibration still buys a small
precision/recall improvement and remains the default.

Sweep on gheim-ch-560m (see eval/calibration_sweep.json,
eval/calibration_sweep_q8.json): ``o_bias=0.5`` is Pareto-clean on both
fp32 PyTorch and q8 ONNX backends.

  fp32 PyTorch (212 probe + 300-chunk test slice)
    bias 0.0  probe 91.5%  test char-F1 0.8947
    bias 0.5  probe 91.5%  test char-F1 0.8974   (+0.27pp, no probe cost)
    bias 1.0+ trades test F1 for marginal probe gains

  q8 ONNX (same suite)
    bias 0.0  probe 73.1%  test char-F1 0.8565
    bias 0.5  probe 74.5%  test char-F1 0.8608   (+1.4pp / +0.43pp)
    bias 1.0+ flattens

This is what ``default_detector()`` returns, so users get the better
behavior without having to opt in.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Annotated, Any

from .base import Span, trim_span
from .local import _DEFAULT_CHUNK_TOKENS, _DEFAULT_OVERLAP_TOKENS, DEFAULT_MODEL


@dataclass
class CalibratedDetector:
    """Local model with O-logit calibration.

    Same surface as ``LocalDetector`` (lazy weight load, sliding-window
    chunking for long inputs, threadsafe), but bypasses the HF
    ``pipeline("token-classification")`` so it can intercept raw logits
    and apply ``o_bias`` before argmax. The aggregation that follows is
    a from-scratch reimplementation of ``aggregation_strategy="simple"``:
    contiguous same-label tokens collapse into one span; offsets are
    char-level via the tokenizer's ``offset_mapping``.

    Default ``o_bias=0.5`` is the Pareto-clean point from the bias sweep
    on gheim-ch-560m: small precision boost on fp32 (+0.27pp char-F1),
    larger boost on the q8 ONNX deployment backend (+0.43pp char-F1,
    +1.4pp probe perfect-rate), no observed cost on either.
    """

    model_id: Annotated[
        str, "HuggingFace model id (token-classification head)."
    ] = DEFAULT_MODEL
    o_bias: Annotated[
        float,
        "Constant subtracted from the O-class logit before argmax. 0 = "
        "uncalibrated (matches LocalDetector). Higher values recover more "
        "suppressed entities at the cost of precision. Sweet spot per "
        "eval/calibration_sweep.json: 0.5 (Pareto-clean), 1.0 (more "
        "aggressive), 1.5+ starts trading too much precision.",
    ] = 0.5
    device: Annotated[
        str, "'auto', 'cpu', 'cuda', or 'cuda:N'."
    ] = "auto"
    dtype: Annotated[Any, "Optional torch dtype override (e.g. torch.bfloat16)."] = None
    trim_whitespace: Annotated[
        bool, "Strip leading/trailing whitespace from each span."
    ] = True
    chunk_tokens: Annotated[
        int, "Sliding-window size in model tokens. Default 510 covers a "
        "512-token model with [CLS]/[SEP] reserved.",
    ] = _DEFAULT_CHUNK_TOKENS
    chunk_overlap_tokens: Annotated[
        int, "Token overlap between consecutive windows. Default 32.",
    ] = _DEFAULT_OVERLAP_TOKENS
    batch_size: Annotated[
        int, "Forward-pass batch size for the chunked + multi-input path.",
    ] = 8

    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _id2label: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _o_id: int = field(default=0, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)
    _load_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForTokenClassification, AutoTokenizer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "CalibratedDetector requires `torch` and `transformers`. "
                    "Install with: uv add 'gheim[local]'"
                ) from e

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            mdl = AutoModelForTokenClassification.from_pretrained(self.model_id)
            if self.dtype is not None:
                mdl = mdl.to(dtype=self.dtype)
            resolved_device = (
                "cuda" if (self.device == "auto" and torch.cuda.is_available())
                else "cpu" if self.device == "auto"
                else self.device
            )
            mdl = mdl.to(resolved_device).eval()
            self._device = resolved_device
            self._model = mdl
            self._id2label = mdl.config.id2label
            # Resolve "O" class id (model config or fall back to label2id).
            label2id = mdl.config.label2id
            self._o_id = int(label2id.get("O", 0))

    def detect(
        self,
        text: Annotated[str, "Input text to scan."],
        model: Annotated[
            str | None, "Per-call model override; must equal self.model_id if set."
        ] = None,
    ) -> list[Span]:
        if model is not None and model != self.model_id:
            raise ValueError(
                f"CalibratedDetector was constructed with model_id={self.model_id!r}; "
                f"per-call override {model!r} not supported."
            )
        if not text:
            return []
        self._ensure_loaded()
        return self._detect_chunked(text)

    def _detect_chunked(self, text: str) -> list[Span]:
        """Chunk-aware inference. For inputs that fit in one window the
        chunking is a no-op; for longer ones we slide the window with
        overlap and reproject offsets back to the original text.
        """
        import torch  # noqa: PLC0415  — lazy import for the no-`torch` install path

        # Tokenize the whole input once with offsets so we know per-token
        # positions in the ORIGINAL text. We never feed this single tokenization
        # to the model — instead we re-encode each window separately because
        # the model expects [CLS]/[SEP] and a length ≤ chunk_tokens + 2.
        full_enc = self._tokenizer(
            text, return_offsets_mapping=True,
            add_special_tokens=False, truncation=False,
        )
        offsets_full = full_enc["offset_mapping"]
        n_tokens = len(offsets_full)

        if n_tokens <= self.chunk_tokens:
            # Single-window fast path.
            return self._detect_window(text, char_offset=0)

        # Sliding window: stride = chunk - overlap.
        stride = max(1, self.chunk_tokens - self.chunk_overlap_tokens)
        seen: set[tuple[int, int, str]] = set()
        out: list[Span] = []
        for win_start in range(0, n_tokens, stride):
            win_end = min(win_start + self.chunk_tokens, n_tokens)
            char_start = offsets_full[win_start][0]
            char_end = offsets_full[win_end - 1][1]
            sub = text[char_start:char_end]
            for sp in self._detect_window(sub, char_offset=char_start):
                key = (sp.start, sp.end, sp.label)
                if key in seen:
                    continue
                seen.add(key)
                out.append(sp)
            if win_end >= n_tokens:
                break
        out.sort(key=lambda s: (s.start, s.end))
        return out

    def _detect_window(self, text: str, char_offset: int) -> list[Span]:
        """Single forward pass on `text`, applying o_bias to the O logit
        before argmax. char_offset shifts emitted spans into the original
        text coordinate frame (used by chunked inference)."""
        import torch  # noqa: PLC0415

        enc = self._tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt",
            truncation=True, max_length=self.chunk_tokens + 2,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits[0].cpu()
        # The calibration knob: subtract o_bias from O class logit.
        if self.o_bias != 0.0:
            logits[:, self._o_id] -= float(self.o_bias)
        pred_ids = logits.argmax(dim=-1).tolist()

        # BIO aggregation: collapse contiguous same-category tokens into
        # one span. This mirrors transformers' "simple" aggregation but
        # without the post-softmax pruning that the pipeline does.
        spans: list[Span] = []
        cur_start, cur_end, cur_cat = None, None, None
        for (s, e), pid in zip(offsets, pred_ids):
            if s == e:  # special token (CLS, SEP, pad)
                continue
            lab = self._id2label[int(pid)]
            cat = None if lab == "O" else lab.split("-", 1)[1]
            if cat == cur_cat and cat is not None:
                cur_end = e
            else:
                if cur_cat is not None:
                    spans.append(self._make_span(text, cur_start, cur_end, cur_cat,
                                                  char_offset))
                cur_start, cur_end, cur_cat = s, e, cat
        if cur_cat is not None:
            spans.append(self._make_span(text, cur_start, cur_end, cur_cat,
                                          char_offset))
        return spans

    def _make_span(
        self, text: str, start: int, end: int, label: str, char_offset: int,
    ) -> Span:
        if self.trim_whitespace:
            start, end = trim_span(text, start, end)
        return Span(
            start=start + char_offset,
            end=end + char_offset,
            label=label,
            text=text[start:end],
            score=1.0,  # we collapsed across the softmax; no per-span score
        )
