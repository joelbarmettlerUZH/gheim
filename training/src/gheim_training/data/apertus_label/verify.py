"""Verify Apertus's claimed (value, label) pairs and emit clean Examples.

For every claim, locate ``value`` in the chunk via ``str.find``. Drop:
  - Values that don't appear (hallucination)
  - Values shorter than 2 characters (noise)
  - Person values that consist entirely of stopwords / digits
  - Date values that fail strict date-shape validation (no month-name OR
    no DD.MM.YYYY/YYYY-MM-DD pattern → drop). M3 mitigation: Apertus
    over-tags journal-issue/year refs ("11/1993") as dates; this filter
    catches them.

Combine with regex-derived structured PII (from ``prefilter.find_structured``)
and resolve overlaps: regex wins (it's deterministic), Apertus claims that
overlap a regex hit get dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from ..schema import Example, Language, Span
from .prefilter import StructuredHit, find_structured


@dataclass(frozen=True, slots=True)
class ApertusClaim:
    value: str
    label: str  # one of private_person / private_address / private_date


# Tokens that should never be tagged as a person on their own. Stored
# case-folded; comparison is case-insensitive (German sentence-initial "Der"
# vs mid-sentence "der" should both be blocked).
_PERSON_BLOCKLIST = {
    s.casefold() for s in (
        "der", "die", "das", "le", "la", "il", "lo", "i", "gli", "les", "des",
        "herr", "frau", "hr.", "fr.", "m.", "mme", "dr.", "sig.", "sig.ra",
        "med.", "avv.",
    )
}


def _person_ok(value: str) -> bool:
    s = value.strip()
    if len(s) < 2:
        return False
    if all(c.isdigit() or not c.isalnum() for c in s):
        return False
    return s.casefold() not in _PERSON_BLOCKLIST


# Month names in DE/FR/IT/RM/EN that signal a real calendar date (M3).
# Case-folded for matching.
_MONTH_NAMES = {
    m.casefold() for m in (
        # German
        "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
        "August", "September", "Oktober", "November", "Dezember",
        "Jan.", "Feb.", "Mär.", "Apr.", "Jun.", "Jul.", "Aug.",
        "Sep.", "Sept.", "Okt.", "Nov.", "Dez.",
        # French
        "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre",
        # Italian
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
        # Romansh (Rumantsch Grischun)
        "schaner", "favrer", "marz", "avrigl", "matg", "zercladur",
        "fanadur", "avust", "settember", "october", "november", "december",
        # English (occasional in citations)
        "january", "february", "march", "may", "june", "july",
        "october", "december",
    )
}

# Strict numeric date patterns. Any of these qualifies a value as a date even
# without a month name (DD.MM.YYYY, YYYY-MM-DD, etc.).
_NUMERIC_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}[./]\d{1,2}[./]\d{2,4}"          # DD.MM.YYYY or DD.MM.YY
    r"|\d{4}[./\-]\d{1,2}[./\-]\d{1,2}"       # YYYY-MM-DD
    r")\b"
)


def _date_ok(value: str) -> bool:
    """True iff ``value`` looks like a real calendar date.

    A value qualifies if it (a) contains a recognized month name, OR
    (b) matches a strict numeric date pattern. Bare year refs ("1985"),
    journal-issue refs ("11/1993"), procedural refs ("Art. 12"), times
    ("08:32 Uhr"), and partial dates without year/day all fail this check.
    """
    s = value.strip()
    if not s:
        return False
    if _NUMERIC_DATE_RE.search(s):
        return True
    # Month-name path: tokenize on spaces and dots to catch "12. März 1985"
    tokens = [t.strip(".,;: ") for t in re.split(r"[\s./]+", s) if t]
    # Require at least one numeric companion (day or year) for sanity.
    has_number = any(t2.isdigit() for t2 in tokens)
    return has_number and any(t.casefold() in _MONTH_NAMES for t in tokens)


# M5: trim citation-prefixed date claims down to the date portion.
# Common shape: "Urteil X vom 12. März 1985 E. 3a", "arrêt 5C.42/2002 du 29 septembre 2002".
# We extract the first sub-string that itself passes _date_ok.
_DATE_EXTRACT_PATTERNS = (
    # DD. <Monat> YYYY (DE) or DD <month> YYYY (FR/IT)
    re.compile(
        r"\b\d{1,2}\.?\s+(?:" + r"|".join(re.escape(m) for m in (
            "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
            "August", "September", "Oktober", "November", "Dezember",
            "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
            "août", "septembre", "octobre", "novembre", "décembre",
            "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
        )) + r")\s+\d{4}\b",
        re.IGNORECASE,
    ),
    # Numeric DD.MM.YYYY etc.
    re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"),
    re.compile(r"\b\d{4}[./\-]\d{1,2}[./\-]\d{1,2}\b"),
    # Month + year only (least specific, last resort)
    re.compile(
        r"\b(?:" + r"|".join(re.escape(m) for m in (
            "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
            "August", "September", "Oktober", "November", "Dezember",
            "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
            "août", "septembre", "octobre", "novembre", "décembre",
            "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
            "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
        )) + r")\s+\d{4}\b",
        re.IGNORECASE,
    ),
)


def _trim_date_value(value: str) -> str | None:
    """If ``value`` is over-spanned (e.g. 'Urteil X vom 12. März 1985 E. 3a'),
    extract the inner date. Returns None if nothing date-shaped is found."""
    for pat in _DATE_EXTRACT_PATTERNS:
        m = pat.search(value)
        if m:
            return m.group(0)
    return None


def verify_and_combine(
    chunk_text: str,
    apertus_claims: list[ApertusClaim],
    *,
    language: str = "de_ch",
    doc_id: str = "",
    subset: str = "",
) -> Example | None:
    """Locate apertus claims + regex hits, dedupe overlaps, build Example.

    Returns ``None`` when the resulting span set is empty.
    """
    spans: list[Span] = []

    # 1. Regex-derived structured PII (treated as ground truth)
    regex_hits: list[StructuredHit] = find_structured(chunk_text)
    for h in regex_hits:
        spans.append(Span(start=h.start, end=h.end, label=h.label))

    # 2. Apertus claims — locate each, sanity-check, append if non-overlapping
    regex_intervals = [(h.start, h.end) for h in regex_hits]
    apertus_spans: list[Span] = []
    for claim in apertus_claims:
        if claim.label not in {"private_person", "private_address", "private_date"}:
            continue
        if claim.label == "private_person" and not _person_ok(claim.value):
            continue
        # M5: trim over-spanned date claims down to the date portion
        if claim.label == "private_date":
            v0 = claim.value
            if not _date_ok(v0) or len(v0) > 35:
                trimmed = _trim_date_value(v0)
                if trimmed is None:
                    continue
                claim = ApertusClaim(value=trimmed, label="private_date")
            elif not _date_ok(v0):
                continue
        v = claim.value.strip()
        if len(v) < 2:
            continue
        idx = chunk_text.find(v)
        if idx < 0:
            continue  # hallucinated
        end = idx + len(v)
        # Trim leading/trailing whitespace fragments (rare but happens)
        while idx < end and chunk_text[idx].isspace():
            idx += 1
        while end > idx and chunk_text[end - 1].isspace():
            end -= 1
        if end <= idx:
            continue
        # Drop if overlapping any regex interval
        if any(idx < re_end and end > re_start for re_start, re_end in regex_intervals):
            continue
        apertus_spans.append(Span(start=idx, end=end, label=claim.label))

    # Dedupe Apertus spans against each other (first-wins)
    apertus_spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    last_end = -1
    for sp in apertus_spans:
        if sp.start < last_end:
            continue
        spans.append(sp)
        last_end = sp.end

    if not spans:
        return None

    spans.sort(key=lambda s: s.start)
    ex = Example(
        text=chunk_text,
        spans=spans,
        language=cast(Language, language),
        source="apertus",  # Layer 5 lives under the same enum slot as Layer 3
        template_id=f"layer5_{subset}",
        meta={"doc_id": doc_id},
    )
    try:
        ex.validate_offsets()
    except ValueError:
        return None
    return ex
