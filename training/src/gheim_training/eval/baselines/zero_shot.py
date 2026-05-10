"""Detector wrapper that runs any HF token-classification model.

Used both as the zero-shot baseline (``openai/privacy-filter``) and as the
"finetuned" detector once we've trained ``joelbarmettler/gheim-1`` — they share
identical inference code, only the model id differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from ...data.schema import Span


@dataclass
class HFDetector:
    model_id_or_path: Annotated[
        str,
        "Either a HuggingFace Hub model id (e.g. 'openai/privacy-filter') or "
        "a local checkpoint path. Loaded lazily on first detect() call.",
    ]
    aggregation_strategy: Annotated[
        str,
        "Token-classification aggregation: 'simple' merges adjacent tokens "
        "with the same predicted label; 'first' / 'average' / 'max' control "
        "subword tie-breaking. 'simple' matches gheim's production wrapper.",
    ] = "simple"
    # transformers.Pipeline has no clean public type; _load() guarantees
    # non-None before .detect() runs.
    _pipe: Any = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        tok = AutoTokenizer.from_pretrained(self.model_id_or_path, trust_remote_code=True)
        mdl = AutoModelForTokenClassification.from_pretrained(self.model_id_or_path, trust_remote_code=True)
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            device=device,
            aggregation_strategy=self.aggregation_strategy,
        )

    def detect(self, text: str) -> list[Span]:
        if not text:
            return []
        self._load()
        assert self._pipe is not None
        raw: list[Span] = []
        for r in self._pipe(text):
            label = str(r.get("entity_group") or r.get("entity"))
            start, end = int(r["start"]), int(r["end"])
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end <= start:
                continue
            raw.append(Span(start=start, end=end, label=label))
        # Mirror gheim's Session.apply_spans merging: adjacent same-label spans
        # separated by ≤1 whitespace char get merged. This is the in-production
        # behaviour any deployed gheim user would see.
        return _merge_adjacent(text, raw)


_MERGE_GAP = 1


def _merge_adjacent(text: str, spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for sp in spans:
        if not out:
            out.append(sp)
            continue
        prev = out[-1]
        if sp.start < prev.end:
            continue  # overlap — first-wins
        gap = text[prev.end:sp.start]
        if sp.label == prev.label and len(gap) <= _MERGE_GAP and gap.strip() == "":
            out[-1] = Span(start=prev.start, end=sp.end, label=prev.label)
        else:
            out.append(sp)
    return out
