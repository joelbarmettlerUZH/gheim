"""V3 synthetic chunk emission format.

Every synthetic generator emits chunks in the Layer-1 JSONL shape:

    {
      "id": "<unique id>",
      "text": "<chunk text>",
      "spans": [{"start": int, "end": int, "label": str}, ...],
      "language": "de_ch | fr_ch | it_ch | rm | en",
      "source": "<synthetic_source_name>",
      "template_id": "<which template family produced this>",
      "meta": {...},  # arbitrary
    }

This is the same shape the the pipeline balancer's ``_load_synthetic_metadata``
expects (treats the synthetic layer like Layer 1 / Layer 9), so the
balancer needs zero changes to consume these outputs.

The schema is intentionally permissive (no labelers / no signals — that's
the labelling pipeline's per-span provenance; for synthetic, the chunk is its own ground
truth with confidence implicitly 1.0).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Span:
    """Character-offset PII span. ``label`` is a label_space.CATEGORIES value."""
    start: int
    end: int
    label: str


@dataclass(slots=True)
class Chunk:
    """One synthetic chunk."""
    id: str
    text: str
    spans: list[Span]
    language: str  # de_ch | fr_ch | it_ch | rm | en
    source: str   # synthetic_<name>
    template_id: str
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Cheap sanity: every span offset is within text, labels are
        valid, spans don't overlap. Catches generator bugs early."""
        from ...label_space import CATEGORIES
        ordered = sorted(self.spans, key=lambda s: (s.start, s.end))
        prev_end = -1
        for sp in ordered:
            if sp.label not in CATEGORIES:
                raise ValueError(f"{self.id}: unknown label {sp.label!r}")
            if sp.start < 0 or sp.end > len(self.text) or sp.end <= sp.start:
                raise ValueError(
                    f"{self.id}: bad offsets [{sp.start},{sp.end}) "
                    f"text_len={len(self.text)}"
                )
            if sp.start < prev_end:
                raise ValueError(
                    f"{self.id}: overlapping spans (this {sp}, "
                    f"prev_end={prev_end})"
                )
            prev_end = sp.end

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "spans": [{"start": s.start, "end": s.end, "label": s.label}
                      for s in sorted(self.spans, key=lambda x: x.start)],
            "language": self.language,
            "source": self.source,
            "template_id": self.template_id,
            "meta": self.meta,
        }


def write_jsonl(path: Path, chunks: Iterable[Chunk]) -> int:
    """Validate every chunk then write JSONL. Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            c.validate()
            f.write(json.dumps(c.to_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def locate(text: str, value: str, start_from: int = 0) -> tuple[int, int]:
    """Find ``value`` in ``text`` (starting from offset ``start_from``)
    and return (start, end). Raises if not found — keeps generators
    honest about template rendering bugs."""
    i = text.find(value, start_from)
    if i < 0:
        raise ValueError(f"value {value!r} not in text starting at {start_from}: "
                         f"{text!r}")
    return i, i + len(value)


def spans_from_values(text: str,
                      pairs: list[tuple[str, str]]) -> list[Span]:
    """Locate each (value, label) pair sequentially in text. Useful when
    a template hand-writes its body and just declares which substrings
    are spans."""
    out: list[Span] = []
    cursor = 0
    for value, label in pairs:
        s, e = locate(text, value, cursor)
        out.append(Span(start=s, end=e, label=label))
        cursor = e
    return out
