"""Detector protocol — anything that maps text to PII spans."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Annotated, Protocol

# Characters stripped when `trim_whitespace=True` is applied to a raw span.
# The upstream `openai/privacy-filter` model tends to include leading
# whitespace in spans (BPE tokenization artifact), which makes restored text
# look wrong (" Alice" instead of "Alice"). Trimming corrects this.
_TRIM_CHARS = " \t\n"


def trim_span(
    text: str, start: int, end: int
) -> tuple[int, int] | None:
    """Return new (start, end) with leading/trailing whitespace stripped.
    Returns None if the trimmed span is empty."""
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    if start >= end:
        return None
    return start, end


@dataclass(frozen=True, slots=True)
class Span:
    """A detected PII span with character offsets into the source text."""

    label: Annotated[str, "Detector label, e.g. 'private_person', 'private_email'."]
    start: Annotated[int, "Inclusive character offset into the original string."]
    end: Annotated[int, "Exclusive character offset into the original string."]
    text: Annotated[str, "Original surface form (text[start:end])."]
    score: Annotated[float, "Detector confidence in [0, 1]."] = 1.0


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def snap_to_words(text: str, spans: list[Span]) -> list[Span]:
    """Widen spans that start or end inside a word to the whole word, then merge
    same-label spans that overlap or touch.

    A token classifier sometimes tags only part of a word (``gano`` of ``Lugano``,
    ``Zbind`` + ``en`` of ``Zbinden``). Masking that leaves the rest of the word
    readable and splits one entity into two placeholders. A span is widened only
    when the characters on both sides of its edge are word characters, so edges
    at spaces or punctuation (``Gay`` in ``Gay-Crosier``) are left alone. The
    merged span's score is the length-weighted mean of its parts.
    """
    snapped: list[Span] = []
    for sp in spans:
        start, end = sp.start, sp.end
        while start > 0 and _is_word_char(text[start - 1]) and _is_word_char(text[start]):
            start -= 1
        while end < len(text) and _is_word_char(text[end - 1]) and _is_word_char(text[end]):
            end += 1
        snapped.append(sp if (start, end) == (sp.start, sp.end)
                       else replace(sp, start=start, end=end, text=text[start:end]))

    merged: list[Span] = []
    last_by_label: dict[str, int] = {}
    for sp in sorted(snapped, key=lambda s: (s.start, s.end)):
        i = last_by_label.get(sp.label)
        prev = merged[i] if i is not None else None
        if prev is not None and sp.start <= prev.end:
            end = max(prev.end, sp.end)
            weight_prev, weight_sp = prev.end - prev.start, sp.end - sp.start
            score = (prev.score * weight_prev + sp.score * weight_sp) / (weight_prev + weight_sp)
            merged[i] = replace(prev, end=end, text=text[prev.start:end], score=score)
        else:
            last_by_label[sp.label] = len(merged)
            merged.append(sp)
    merged.sort(key=lambda s: (s.start, s.end))
    return merged


class Detector(Protocol):
    """Anything that takes a string and returns a list of PII spans."""

    def detect(
        self,
        text: Annotated[str, "Source text to scan for PII."],
        *,
        model: Annotated[str | None, "Optional per-call model id override."] = None,
    ) -> list[Span]:  # pragma: no cover - protocol
        ...
