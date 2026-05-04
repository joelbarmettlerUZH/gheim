"""Post-generation verification: locate each slot value in the model output.

If every value appears at least once verbatim, we extract char spans and emit
an Example. If any value is missing or duplicated in confusing ways, we drop
the example. Lossy by design — better to discard than mislabel.
"""
from __future__ import annotations

from ..schema import Example, Span


class SlotVerificationError(ValueError):
    pass


def locate_spans(
    text: str,
    slot_bag: list[tuple[str, str]],
) -> list[Span]:
    """Find each (category, value) in ``text`` and return char-spans.

    For each value we take the **first** occurrence. If a value occurs zero
    times, raise SlotVerificationError so the caller can drop the example.
    Multiple occurrences of the same value are tolerated (only the first is
    labeled — the model needs to learn that PII can appear without being
    re-tagged everywhere, which mirrors real data).
    """
    spans: list[Span] = []
    for cat, value in slot_bag:
        idx = text.find(value)
        if idx < 0:
            raise SlotVerificationError(f"missing value in output: [{cat}] {value!r}")
        spans.append(Span(start=idx, end=idx + len(value), label=cat))
    return spans


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Drop any span that overlaps a previously-kept span.

    This handles the rare case where Faker happens to embed one PII value
    inside another (e.g. a phone number appearing inside a longer URL).
    """
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[Span] = []
    last_end = -1
    for sp in spans:
        if sp.start < last_end:
            continue
        kept.append(sp)
        last_end = sp.end
    return kept


def build_example(
    text: str,
    slot_bag: list[tuple[str, str]],
    *,
    language: str,
    template_id: str,
) -> Example:
    spans = locate_spans(text, slot_bag)
    spans = resolve_overlaps(spans)
    if not spans:
        raise SlotVerificationError("no spans survived overlap resolution")
    ex = Example(
        text=text,
        spans=spans,
        language=language,  # type: ignore[arg-type]
        source="apertus",
        template_id=template_id,
    )
    ex.validate_offsets()
    return ex
