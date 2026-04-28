"""Sentinel format, allocation, and regex matching.

A sentinel is a stable, uniquely-numbered placeholder like ``<PERSON_1>`` or
``<EMAIL_2>``. Each PII label maps to a tag (``private_person`` → ``PERSON``),
and each session allocates indices starting at 1 per tag.

The format is ASCII-only angle brackets so that downstream LLM tokenizers are
most likely to preserve it as-is. Brackets + uppercase letters + digits + one
underscore is a narrow enough shape that most producers won't mangle it.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated

# Canonical label → short tag used in the sentinel.
LABEL_TO_TAG: dict[str, str] = {
    "private_person": "PERSON",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_address": "ADDRESS",
    "private_url": "URL",
    "private_date": "DATE",
    "account_number": "ACCOUNT",
    "secret": "SECRET",
}

# Reverse map for completeness (not strictly needed for round-trip).
TAG_TO_LABEL: dict[str, str] = {v: k for k, v in LABEL_TO_TAG.items()}

_TAG_PATTERN = r"[A-Z][A-Z0-9_]*"
_INDEX_PATTERN = r"\d+"

# A fully-formed sentinel anywhere in a string.
SENTINEL_RE = re.compile(rf"<(?P<tag>{_TAG_PATTERN})_(?P<index>{_INDEX_PATTERN})>")

# A *possibly-partial* sentinel anchored at the start of a buffer. Used by the
# streaming deanonymizer to decide "should I hold back these bytes or emit them?"
#
# Matches any strict prefix of a sentinel:
#   "<", "<P", "<PERSON", "<PERSON_", "<PERSON_1", "<PERSON_12>"
# Does NOT match if the buffer contains disqualifying characters (e.g. "<x", "<PERSON ").
_PARTIAL_SENTINEL_RE = re.compile(
    rf"^<({_TAG_PATTERN})?(_{_INDEX_PATTERN}?)?>?"
)


def label_tag(
    label: Annotated[str, "Detector label, e.g. 'private_person' or 'account_number'."],
) -> str:
    """Map a detector label to its sentinel tag. Unknown labels fall back to the upper-cased label."""
    return LABEL_TO_TAG.get(label, label.upper())


@dataclass(frozen=True, slots=True)
class Sentinel:
    """A specific placeholder issued inside a session."""

    tag: Annotated[str, "Short uppercase tag (e.g. 'PERSON')."]
    index: Annotated[int, "1-based per-tag counter."]

    def render(self) -> str:
        return f"<{self.tag}_{self.index}>"

    @classmethod
    def parse(
        cls,
        s: Annotated[str, "Candidate sentinel string to parse, e.g. '<PERSON_7>'."],
    ) -> Sentinel | None:
        m = SENTINEL_RE.fullmatch(s)
        if not m:
            return None
        return cls(tag=m["tag"], index=int(m["index"]))


def find_sentinels(
    text: Annotated[str, "Text to scan for fully-formed sentinels."],
) -> Iterator[tuple[int, int, Sentinel]]:
    """Yield (start, end, sentinel) for each sentinel occurrence in ``text``."""
    for m in SENTINEL_RE.finditer(text):
        yield m.start(), m.end(), Sentinel(tag=m["tag"], index=int(m["index"]))


def is_possible_sentinel_prefix(
    buffer: Annotated[str, "Buffer accumulated since the last '<' was seen by the streaming buffer."],
) -> bool:
    """Could ``buffer`` still become a sentinel if more characters arrive?

    Returns True if the buffer is a non-empty strict prefix of some valid sentinel
    (i.e. consumable) AND does not already contain disqualifying characters.
    A fully-formed sentinel (ending in ``>``) is NOT a prefix — caller should have
    already matched and consumed it.
    """
    if not buffer or buffer[0] != "<":
        return False
    m = _PARTIAL_SENTINEL_RE.match(buffer)
    if m is None:
        return False
    # Must consume the whole buffer — otherwise there's trailing garbage.
    if m.end() != len(buffer):
        return False
    # If it ends with ">" it's already complete — not a prefix.
    if buffer.endswith(">"):
        return False
    return True
