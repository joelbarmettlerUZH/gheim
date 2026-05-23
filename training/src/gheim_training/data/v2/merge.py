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

v3 additions
------------

The v2 merge groups by (value, label), so two spans with the SAME
value but DIFFERENT labels (e.g. "Maienfeld" tagged as address by
gemma+qwen and as person by nemotron) survive as two separate output
spans. The bake-off audit surfaced 449 chunks (0.23% of v2_balanced)
where this produced exact (start, end) collisions with conflicting
labels — most commonly Swiss place names mislabelled as person by
Nemotron, dates labelled as person by one of the LLMs, and overlapping
URL boundaries.

v3 adds two post-merge fixes:

1. `_apply_swiss_place_gazetteer`: when a span is labelled private_person
   but its value is a known Swiss city/town, demote it to private_address.
   Pulls the gazetteer from Geonames-CH (the same source as the synthetic
   address generator). Closes the "Maienfeld → person" misclass class.

2. `_resolve_label_conflicts`: after the gazetteer step, two spans may
   share exact (start, end). If their labels MATCH, merge their signals
   into one span with the union. If labels DIFFER, pick the span with
   more non-audit signals (tie-break: alphabetically earlier label).

These run AFTER the standard merge_signals dedup, so the existing v2
test suite is unaffected unless callers opt-in via apply_v3_fixes=True.
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
    nemotron: Iterable[_RawSpan] = (),
    regex: Iterable[_RawSpan] = (),
    audit: Iterable[_RawSpan] = (),
    n_candidate_signals: int = 4,
    apply_v3_fixes: bool = False,
) -> list[V2Span]:
    """Merge spans from up to five sources into the v2 span list.

    ``n_candidate_signals`` is how many signals were actually run on
    this chunk. Default 4 = gemma + qwen + nemotron + regex (the v2
    build). The audit signal is sample-only (only present on the
    580-chunk P7 audit subset) and is excluded from the confidence
    denominator. Confidence 1.0 = every candidate labeller agreed.

    When ``apply_v3_fixes=True``, additionally runs:
    - Swiss place-name gazetteer demotion (person → address)
    - label-conflict resolution by majority signal count

    See module docstring for rationale.
    """
    inputs: dict[V2Signal, list[_RawSpan]] = {
        "gemma": list(gemma),
        "qwen": list(qwen),
        "nemotron": list(nemotron),
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
    if apply_v3_fixes:
        out = _apply_swiss_place_gazetteer(out)
        out = _resolve_label_conflicts(out, n_candidate_signals)
    return out


# ---------------------------------------------------------------------------
# v3 post-merge fixes
# ---------------------------------------------------------------------------

import functools


@functools.lru_cache(maxsize=1)
def _swiss_place_names() -> frozenset[str]:
    """Casefolded set of every Swiss place name from Geonames-CH. ~5000
    entries. Used to demote private_person mislabels on city names."""
    from ..synthetic.swiss_geo import all_places
    return frozenset(p.city.casefold().strip() for p in all_places())


def _apply_swiss_place_gazetteer(spans: list[V2Span]) -> list[V2Span]:
    """Demote private_person spans whose value is a known Swiss city to
    private_address. Preserves signals, confidence, and offsets.

    Common case this fixes: Nemotron labels "Maienfeld", "Coira",
    "Samedan" as person; gemma+qwen label them as address. With v3
    fixes on, the person label is suppressed (replaced with address)
    and downstream label-conflict resolution merges the signals into
    a single, correctly-typed address span.
    """
    place_set = _swiss_place_names()
    out: list[V2Span] = []
    for sp in spans:
        if (sp.label == "private_person"
                and sp.value.casefold().strip() in place_set):
            out.append(V2Span(
                start=sp.start, end=sp.end,
                label="private_address",
                value=sp.value, signals=sp.signals,
                confidence=sp.confidence, regex_subtype=sp.regex_subtype,
            ))
        else:
            out.append(sp)
    return out


def _resolve_label_conflicts(spans: list[V2Span],
                             n_candidate_signals: int = 4) -> list[V2Span]:
    """Group spans by exact (start, end). If a group has multiple spans:

    - All same label → merge their signals (union), RECOMPUTE confidence
      from the union (since e.g. nemotron-via-gazetteer is now counted
      toward address). Preserve regex_subtype if any.
    - Different labels → pick winning label by non-audit signal count;
      tie-break by alphabetically earlier label. Discard the loser.
    """
    by_pos: dict[tuple[int, int], list[V2Span]] = {}
    for sp in spans:
        by_pos.setdefault((sp.start, sp.end), []).append(sp)
    out: list[V2Span] = []
    for (s, e), group in by_pos.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        # Group by label within the position group
        by_label: dict[str, list[V2Span]] = {}
        for sp in group:
            by_label.setdefault(sp.label, []).append(sp)
        # If only one label remains (gazetteer-merge case), union all signals
        if len(by_label) == 1:
            (label, label_group), = by_label.items()
            merged_signals = tuple(sorted({s for sp in label_group for s in sp.signals}))
            non_audit = [s for s in merged_signals if s != "audit"]
            new_conf = len(non_audit) / n_candidate_signals if n_candidate_signals else 1.0
            regex_subtype = next(
                (sp.regex_subtype for sp in label_group if sp.regex_subtype),
                None,
            )
            value = label_group[0].value
            out.append(V2Span(
                start=s, end=e, label=label, value=value,
                signals=merged_signals, confidence=new_conf,
                regex_subtype=regex_subtype,
            ))
            continue
        # Multiple labels at same position — pick winner by signal count.
        def _key(sp: V2Span) -> tuple[int, str]:
            non_audit = [x for x in sp.signals if x != "audit"]
            return (-len(non_audit), sp.label)  # most signals first, then alpha
        winner = sorted(group, key=_key)[0]
        out.append(winner)
    out.sort(key=lambda x: (x.start, x.end))
    return out
