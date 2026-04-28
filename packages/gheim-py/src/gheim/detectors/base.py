"""Detector protocol — anything that maps text to PII spans."""
from __future__ import annotations

from dataclasses import dataclass
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


class Detector(Protocol):
    """Anything that takes a string and returns a list of PII spans."""

    def detect(
        self,
        text: Annotated[str, "Source text to scan for PII."],
        *,
        model: Annotated[str | None, "Optional per-call model id override."] = None,
    ) -> list[Span]:  # pragma: no cover - protocol
        ...
