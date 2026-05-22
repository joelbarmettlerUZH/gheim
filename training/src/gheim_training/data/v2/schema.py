"""V2 dataset schema with per-span provenance metadata.

The v1 dataset stores spans as just ``(start, end, label)`` plus a
single chunk-level ``synthetic`` boolean. V2 adds per-span provenance
so downstream consumers can:

- filter to high-confidence spans only (``len(signals) >= 2``);
- inspect which deterministic regex rule fired (``regex_subtype``);
- replay the v1 build by filtering to the v1 signal set.

Schema (one record per labelled chunk):

    {
      "id": "<chunk id>",
      "text": "<chunk text>",
      "language": "de_ch | fr_ch | it_ch | rm | en",
      "subset": "entscheidsuche | curia_vista | fineweb | romansh | synthetic",
      "doc_id": "...",
      "spans": [
        {
          "start": 12, "end": 25,
          "label": "private_email",
          "value": "alice@example.ch",
          "signals": ["gemma", "qwen", "regex"],
          "confidence": 1.0,
          "regex_subtype": null
        },
        ...
      ],
      "build_pipeline": "gheim-pii-v2.0",
      "labelers": ["gemma-4-26B-A4B-it-AWQ-4bit",
                   "Qwen3.6-35B-A3B-AWQ-4bit",
                   "regex+checksum:gheim.detectors.composite._find_regex_spans"],
    }

The schema is enforced by :class:`V2Span` and :class:`V2Example`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from ..label_space import CATEGORIES

V2_PIPELINE_VERSION = "gheim-pii-v2.0"

# Canonical signal names. Any v2 span MUST list its signals as a subset
# of these, in deterministic alphabetical order so identical inputs
# produce identical JSON. "audit" is the four-LLM OpenRouter consensus
# (Kimi/DeepSeek-V4-Pro/MiniMax/GLM) used for label-noise scoring only
# — it never contributes to the dataset's published labels (see merge.py
# where audit is excluded from the confidence denominator).
V2_SIGNALS: tuple[str, ...] = ("audit", "gemma", "nemotron", "qwen", "regex")

V2Signal = Literal["audit", "gemma", "nemotron", "qwen", "regex"]


@dataclass(frozen=True, slots=True)
class V2Span:
    """A character-offset PII span with per-span provenance.

    ``signals`` is which labellers flagged this (value, label) pair on
    this chunk. ``confidence`` is ``len(signals) / n_candidate_signals``
    where ``n_candidate_signals`` is the number of labellers that
    actually ran on this chunk (3 in the standard v2 build:
    gemma + qwen + regex). ``regex_subtype`` is set iff regex flagged
    the span and identifies which sub-pattern fired
    (``"iban_ch"``, ``"ahv"``, ``"vat_che"``, ``"credit_card"``,
    ``"swiss_phone"``, ``"openai_key"``, ``"github_pat"``,
    ``"aws_access_key"``).
    """

    start: Annotated[int, "Inclusive char offset into the chunk text."]
    end: Annotated[int, "Exclusive char offset into the chunk text."]
    label: Annotated[str, "PII category from label_space.CATEGORIES."]
    value: Annotated[str, "Verbatim surface form (chunk_text[start:end])."]
    signals: Annotated[
        tuple[str, ...],
        "Sorted tuple of labellers that flagged this span. "
        "Subset of V2_SIGNALS.",
    ]
    confidence: Annotated[
        float,
        "len(signals) / n_candidate_signals on this chunk. "
        "1.0 = unanimous, 1/3 = single-signal.",
    ]
    regex_subtype: Annotated[
        str | None,
        "Subtype tag from the regex catalogue when 'regex' is in signals.",
    ] = None

    def __post_init__(self) -> None:
        if self.label not in CATEGORIES:
            raise ValueError(
                f"unknown category {self.label!r}; expected one of {CATEGORIES}"
            )
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")
        if not self.signals:
            raise ValueError(f"v2 span must have at least one signal: {self}")
        for s in self.signals:
            if s not in V2_SIGNALS:
                raise ValueError(
                    f"unknown signal {s!r}; expected one of {V2_SIGNALS}"
                )
        if list(self.signals) != sorted(self.signals):
            raise ValueError(
                f"signals must be sorted alphabetically: {self.signals}"
            )
        if not (0.0 < self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in (0, 1]: got {self.confidence}"
            )


@dataclass(slots=True)
class V2Example:
    """One labelled chunk in the v2 dataset."""

    id: str
    text: str
    language: str
    subset: str
    doc_id: str
    spans: list[V2Span]
    build_pipeline: str = V2_PIPELINE_VERSION
    labelers: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        # Ensure deterministic ordering: spans by start, signals already sorted.
        d["spans"] = sorted(d["spans"], key=lambda sp: (sp["start"], sp["end"]))
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> V2Example:
        d = json.loads(s)
        spans_raw = d.pop("spans", [])
        spans = [
            V2Span(
                start=sp["start"],
                end=sp["end"],
                label=sp["label"],
                value=sp["value"],
                signals=tuple(sp["signals"]),
                confidence=sp["confidence"],
                regex_subtype=sp.get("regex_subtype"),
            )
            for sp in spans_raw
        ]
        return cls(spans=spans, **d)


def write_jsonl(path: Path, examples: Iterable[V2Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json())
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[V2Example]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield V2Example.from_json(line)
