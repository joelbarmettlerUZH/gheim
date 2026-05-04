"""Canonical training-example schema.

All data layers (synthetic, AI4Privacy, Apertus, English anchor, real)
normalize to ``Example`` before being combined in build.py.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from .label_space import CATEGORIES

Language = Literal["de_ch", "fr_ch", "it_ch", "rm", "gsw", "en"]
Source = Literal["synthetic", "ai4privacy", "apertus", "real", "english_anchor"]


@dataclass(frozen=True, slots=True)
class Span:
    """A character-offset PII span. ``label`` is one of CATEGORIES."""

    start: Annotated[int, "Inclusive char offset into the example text."]
    end: Annotated[int, "Exclusive char offset into the example text."]
    label: Annotated[str, "PII category from label_space.CATEGORIES."]

    def __post_init__(self) -> None:
        if self.label not in CATEGORIES:
            raise ValueError(f"unknown category {self.label!r}; expected one of {CATEGORIES}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")


@dataclass(slots=True)
class Example:
    text: str
    spans: list[Span]
    language: Language
    source: Source
    template_id: str | None = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> Example:
        d = json.loads(s)
        spans = [Span(**sp) for sp in d.pop("spans")]
        return cls(spans=spans, **d)

    def validate_offsets(self) -> None:
        """Cheap sanity: spans don't overlap and surface forms are non-empty."""
        ordered = sorted(self.spans, key=lambda s: s.start)
        prev_end = -1
        for sp in ordered:
            if sp.start < prev_end:
                raise ValueError(f"overlapping spans in example: {sp} after end={prev_end}")
            if sp.end > len(self.text):
                raise ValueError(f"span {sp} exceeds text length {len(self.text)}")
            if not self.text[sp.start:sp.end].strip():
                raise ValueError(f"empty/whitespace span {sp} in {self.text!r}")
            prev_end = sp.end


def write_jsonl(path: Path, examples: Iterable[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json())
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Example]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Example.from_json(line)
