"""Cheap regex prefilter — keep only chunks containing structured PII.

Filtering rationale: Apertus is expensive to call. We don't want to feed it
chunks that are mostly procedural text with no PII at all (a lot of court
decisions and parliamentary minutes have huge boilerplate spans). A chunk
that contains at least one regex-matchable structured PII (IBAN-CH, AHV,
VAT, email, Swiss phone, credit card, date) is statistically much more
likely to also contain person/address/date worth labeling.

The same regexes also serve as **ground truth** for the structured-PII
categories at training time (see ``label.py``) — Apertus is only asked
about person/address/date.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .stream import Chunk

# Each tuple: (gheim_category, compiled regex). Order matters for overlap
# resolution downstream.
STRUCTURED_PII_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_email", re.compile(r"\b[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b")),
    # URL match is greedy then post-stripped of trailing sentence punctuation.
    # We can't easily do "stop before trailing punct" inside the regex without
    # mis-handling dots in domains/paths.
    ("private_url", re.compile(r"https?://[^\s)>\]]+")),
    ("private_phone", re.compile(
        r"(?:\+41|0041|0)\s?(?:\(0\)\s?)?\d{2}[\s\-/.]?\d{3}[\s\-/.]?\d{2}[\s\-/.]?\d{2}"
    )),
    ("account_number", re.compile(r"\bCH\d{2}(?:\s?\d){17}\b")),  # IBAN-CH
    ("account_number", re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b")),
    ("account_number", re.compile(r"\b756\d{10}\b")),
    ("account_number", re.compile(
        r"\bCHE-\d{3}\.\d{3}\.\d{3}(?:\s+(?:MWST|TVA|IVA))?\b"
    )),
    ("account_number", re.compile(r"\b(?:\d[\s\-]?){15,16}\d\b")),  # CC-shaped
    # Dates: require a 4-digit year and at least one delimited day/month
    # segment. Rejects section references ("2.2.11") and parliamentary IDs
    # ("10.3195"). MM/YYYY is intentionally NOT matched — it false-fires
    # on numeric substrings of fuller dates ("8.2012" inside "22.8.2012").
    ("private_date", re.compile(
        r"\b(?:"
        r"\d{1,2}[./]\d{1,2}[./]\d{4}"        # DD.MM.YYYY or DD/MM/YYYY
        r"|\d{4}[./\-]\d{1,2}[./\-]\d{1,2}"   # YYYY-MM-DD
        r")\b"
    )),
    ("secret", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("secret", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    # ALL-CAPS surnames in academic/legal citation style.
    # Pattern: one or more tokens of 3+ uppercase letters (with optional
    # accented chars, hyphens, periods for initials), separated by spaces.
    # Examples: "ROLAND PFÄFFLI", "PIERRE A. KARRER", "VAN DEN BERG",
    # "STAEHELIN", "AEBI-MÜLLER", "PETER F. SCHLOSSER".
    # Slash-separated co-authors (STAEHELIN/GROLIMUND) match each piece
    # separately because the regex doesn't cross slashes.
    ("private_person", re.compile(
        r"(?:\b[A-Z][A-ZÀ-ÖØ-Ý]{2,}(?:[\-A-ZÀ-ÖØ-Ý]+)*\b\.?\s*)"
        r"(?:\b[A-Z]\.\s+)?"
        r"(?:\b[A-Z][A-ZÀ-ÖØ-Ý]{2,}(?:[\-A-ZÀ-ÖØ-Ý]+)*\b\.?\s*)*"
        r"\b[A-Z][A-ZÀ-ÖØ-Ý]{2,}(?:[\-A-ZÀ-ÖØ-Ý]+)*\b"
        r"|\b[A-Z][A-ZÀ-ÖØ-Ý]{3,}(?:[\-A-ZÀ-ÖØ-Ý]+)*\b"  # standalone surname like "STAEHELIN"
    )),
)

# Tokens that look ALL-CAPS but are NOT person names — case-folded.
_ALLCAPS_BLOCKLIST = {
    s.casefold() for s in (
        # Swiss/EU institutions and codes
        "BGE", "BBL", "BFE", "OR", "ZGB", "ZPO", "StPO", "MWST", "TVA", "IVA",
        "AHV", "AVS", "AVS/AI", "IBAN", "UID", "IDE", "CHE", "EU", "UE",
        "BV", "FF", "BUND", "USA", "GMBH", "AG", "SA", "SARL", "EUGH",
        "EFTA", "OECD", "WHO", "UNO", "UN", "NATO", "EWR", "BJ",
        "IPRG", "CPC", "CPP", "CO", "CC", "LDA", "LDes", "LCD", "OBI",
        "EuGVO", "EuGVVO", "ABl", "EuGH", "EFTA", "EWG", "EWR",
        # French Swiss legal codes (frequent FP source in fr-CH court text)
        "LPGA", "LACI", "LP", "LPP", "LAMal", "LBA", "LAA", "LAVS",
        "LDes", "LIPI", "LIA", "LIFD", "LCart", "LPN", "LPE", "LCR",
        "LDIP", "LCC", "LFus", "LSF", "LSFin", "LB", "LBA", "LBVM",
        "LSR", "LRH", "LTr", "LSC", "LDP", "LADB", "LAGV", "LFPr",
        "LASV", "LAFam", "LAA", "LAI", "LPart", "LFLP", "LCdF", "LDI",
        "LICdN", "LIPR", "LDETI",
        # Italian Swiss legal codes
        "LIVA", "LP", "LRD", "LBVM", "LFG", "LCart",
        # Common law-publication abbreviations that look ALL-CAPS
        "TDPS", "JdT", "ATF", "RJN", "RVJ", "REJB", "FZR", "RDS",
        "ZGB", "OR", "ZGB/OR", "URP", "GVP", "AJP/PJA",
        # Other Swiss-specific institutions
        "OFJ", "OFAS", "OFSP", "DDPS", "DFAE", "DFI", "DFJP", "DFF",
        "DEFR", "DETEC", "OSAS", "BFG", "EJPD",
        # Disease/epidemic acronyms that surface as ALL-CAPS plus hyphen
        "COVID", "COVID-19", "SARS", "SARS-CoV-2", "HIV", "AIDS", "MERS",
        "BSE", "ESBL", "MRSA", "TBC", "EBOLA",
        # Tech / standards / programs commonly all-caps
        "DNA", "RNA", "CO2", "CHF", "EUR", "USD", "GBP",
        "SUVA", "SECO", "BFS", "OFS", "UST",
        "SBB", "CFF", "FFS", "SRG", "SSR",
        # Common abbreviations
        "PDF", "PNG", "JPG", "MP3", "MP4", "WIFI", "API", "URL",
        # Swiss legal journals (frequent FP source in court citations)
        "RNRF", "ZBGR", "ZBJV", "ZSR", "ZSW", "AJP", "BN", "RDS", "JdT",
        "SJZ", "ASA", "RJN", "RVJ", "ICCA", "RIW", "IPRax", "ZZP",
        "ZZPInt", "AGVE", "ZWR", "PJA", "RDAF", "JT", "SJ",
        # Court/admin abbreviations
        "BAG", "BGH", "OLG", "EGMR", "ECHR", "EuGH", "GG", "BGB", "HGB",
        "ZPO/AG", "ZPO/BS", "ZPO/ZH",
        # Citation reference qualifiers
        "ED", "EDS", "VOL", "NO", "NR", "ART", "ZIFF", "CHAP", "PARA",
        "CF", "VGL", "USW", "SIEHE", "VOIR", "ETC", "ETC.", "I.E", "E.G",
        "OP", "CIT", "IBID", "LOC", "AUFL", "EDN", "ED.", "BAND", "BD",
        "PAR", "PARS", "RZ", "FN", "ANM", "REGESTE",
        # Country/canton codes
        "CH", "DE", "FR", "IT", "AT",
        "ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "SO",
        "BS", "BL", "SH", "AR", "AI", "SG", "GR", "AG", "TG", "TI",
        "VD", "VS", "GE", "JU", "NE", "FR",
    )
}


@dataclass(frozen=True, slots=True)
class StructuredHit:
    label: str
    start: int
    end: int
    text: str


_URL_TRAILING_STRIP = ".,;:!?\")>]"


def _trim_url(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip trailing sentence punctuation that the greedy URL regex grabbed."""
    while end > start and text[end - 1] in _URL_TRAILING_STRIP:
        end -= 1
    return start, end


def _is_blocklisted_caps(text: str) -> bool:
    """True iff every word in the (already trimmed) ALL-CAPS span is in the
    block-list — used to drop institution acronyms that shape-match person."""
    tokens = [t for t in re.split(r"[\s\-./]+", text) if t]
    if not tokens:
        return True
    return all(t.casefold() in _ALLCAPS_BLOCKLIST for t in tokens)


def find_structured(text: str) -> list[StructuredHit]:
    """Return all structured-PII matches in ``text``.

    Overlaps are resolved greedily: earlier-start wins, longer span breaks ties.
    URL hits are post-stripped of trailing sentence punctuation.
    ALL-CAPS person hits are filtered against an institution acronym blocklist.
    """
    raw: list[StructuredHit] = []
    for label, pat in STRUCTURED_PII_REGEXES:
        for m in pat.finditer(text):
            start, end = m.start(), m.end()
            value = text[start:end].rstrip(".,;: ")
            end = start + len(value)
            if end <= start:
                continue
            if label == "private_url":
                start, end = _trim_url(text, start, end)
                if end <= start:
                    continue
            if label == "private_person":
                if _is_blocklisted_caps(text[start:end]):
                    continue
                # Require at least 3 alpha chars total — drops single-letter
                # initial sequences like "P. A."
                if sum(1 for c in text[start:end] if c.isalpha()) < 3:
                    continue
            raw.append(StructuredHit(label=label, start=start, end=end,
                                     text=text[start:end]))
    raw.sort(key=lambda h: (h.start, -(h.end - h.start)))
    out: list[StructuredHit] = []
    last_end = -1
    for h in raw:
        if h.start < last_end:
            continue
        out.append(h)
        last_end = h.end
    return out


def passes(text: str, *, min_hits: int = 1) -> bool:
    """True iff ``text`` has at least ``min_hits`` structured-PII matches."""
    n = 0
    for _, pat in STRUCTURED_PII_REGEXES:
        if pat.search(text):
            n += 1
            if n >= min_hits:
                return True
    return False


def filter_chunks(
    chunks: Iterator[Chunk],
    *,
    min_structured_hits: int = 1,
) -> Iterator[Chunk]:
    """Pass-through generator: yields only chunks whose text passes ``passes()``."""
    for c in chunks:
        if passes(c.text, min_hits=min_structured_hits):
            yield c
