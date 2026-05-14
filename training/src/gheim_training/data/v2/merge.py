"""Merge per-source PII span lists into v2 spans with provenance.

The v2 build has up to four signals per chunk:

- ``gemma``  — the original v1 LLM labeller (cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit)
- ``qwen``   — the v2 second LLM labeller (cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit)
- ``regex``  — the deterministic regex catalogue (with checksums + the new
               column-context filter introduced in v2-1)
- ``audit``  — optional, only present on chunks the four-LLM OpenRouter
               audit covered (the n=580 sample)

For each chunk this module collapses the per-source span lists into a
single deduplicated v2 span list. Two spans are considered "the same
entity" when they share the same casefolded value AND the same label.
The resulting v2 span carries:

- the longest observed (start, end) — preserves boundary detail when
  one source caught more context than another (e.g. ``"Müller"`` vs
  ``"Dr. Müller"``);
- the union of source signals;
- a confidence proportional to ``len(signals) / n_candidate_signals``;
- the ``regex_subtype`` if regex contributed, else ``None``.

Why this design (rather than IoU-based span matching): LLM labellers
return ``value`` strings, not character offsets. The dataset's existing
``_verify_and_locate`` helper resolves them by ``str.find``, which
yields the first occurrence. Using value+label as the merge key avoids
double-counting the same surface form when two labellers found
different occurrences of the same string within the chunk.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .schema import V2_SIGNALS, V2Signal, V2Span


@dataclass(frozen=True, slots=True)
class _RawSpan:
    """An incoming span from one source. Lightweight intermediate that
    avoids tying merge to either gheim.detectors.base.Span or
    gheim_training.data.schema.Span."""

    start: int
    end: int
    label: str
    value: str
    regex_subtype: str | None = None


def from_value_pair(text: str, value: str, label: str,
                    regex_subtype: str | None = None) -> _RawSpan | None:
    """Locate ``value`` in ``text`` (first occurrence) and build a
    _RawSpan. Returns None if the value isn't a verbatim substring —
    same M1 gate the original Gemma labeller uses."""
    v = value.strip()
    if not v:
        return None
    idx = text.find(v)
    if idx < 0:
        return None
    return _RawSpan(start=idx, end=idx + len(v), label=label,
                    value=v, regex_subtype=regex_subtype)


def merge_signals(
    text: str,
    *,
    gemma: Iterable[_RawSpan] = (),
    qwen: Iterable[_RawSpan] = (),
    regex: Iterable[_RawSpan] = (),
    audit: Iterable[_RawSpan] = (),
    n_candidate_signals: int = 3,
) -> list[V2Span]:
    """Merge spans from up to four sources into the v2 span list.

    ``n_candidate_signals`` is how many signals were actually run on
    this chunk (typically 3 = gemma + qwen + regex; the audit signal is
    sample-only and not counted toward confidence). It is used as the
    denominator of the per-span confidence so that 1.0 means "every
    candidate signal agreed".
    """
    inputs: dict[V2Signal, list[_RawSpan]] = {
        "gemma": list(gemma),
        "qwen": list(qwen),
        "regex": list(regex),
        "audit": list(audit),
    }

    # Group by (value.casefold(), label). Keep the longest observed
    # (start, end), the union of signals, and any regex subtype.
    grouped: dict[tuple[str, str], dict] = {}
    for source, spans in inputs.items():
        for sp in spans:
            key = (sp.value.casefold().strip(), sp.label)
            entry = grouped.get(key)
            if entry is None:
                grouped[key] = {
                    "start": sp.start,
                    "end": sp.end,
                    "label": sp.label,
                    "value": sp.value,
                    "signals": {source},
                    "regex_subtype": sp.regex_subtype,
                }
                continue
            entry["signals"].add(source)
            # Prefer the longest observed (start, end) so boundary
            # detail is preserved when one source caught more context.
            if (sp.end - sp.start) > (entry["end"] - entry["start"]):
                entry["start"] = sp.start
                entry["end"] = sp.end
                entry["value"] = sp.value
            # Regex subtype only set by the regex source.
            if sp.regex_subtype is not None and entry["regex_subtype"] is None:
                entry["regex_subtype"] = sp.regex_subtype

    out: list[V2Span] = []
    for entry in grouped.values():
        signals = tuple(sorted(entry["signals"]))
        # Audit signal does NOT count toward confidence — it's sample-only.
        non_audit_signals = [s for s in signals if s != "audit"]
        confidence = len(non_audit_signals) / n_candidate_signals
        if confidence == 0.0:
            # Audit-only span (no labeller signal). Skip — these are
            # rare and represent disagreement, not consensus.
            continue
        out.append(V2Span(
            start=entry["start"],
            end=entry["end"],
            label=entry["label"],
            value=entry["value"],
            signals=signals,
            confidence=confidence,
            regex_subtype=entry["regex_subtype"],
        ))
    out.sort(key=lambda s: (s.start, s.end))
    return out
