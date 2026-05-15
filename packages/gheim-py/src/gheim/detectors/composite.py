"""Composite detector: regex front-end + model on the remainder.

Production-ready Swiss PII detector that combines two complementary
strategies:

1. **Deterministic regex front-end** for structured PII categories where
   shape is unambiguous and checksums verify identity:
   - IBAN-CH (mod-97-10), AHV (EAN-13 mod-10), VAT-CHE (mod-11-10),
     Visa/Mastercard (Luhn)
   - Email, URL, Swiss phone (+41 / 0041 / 0...), explicit dates
     (DD.MM.YYYY, YYYY-MM-DD, "31. Juli 2013" style)
   - API tokens (sk-, ghp_, AKIA…)

   Spans found here are "trusted" — checksums catch most surface-form
   imitations; the model is never asked.

2. **Fine-tuned model** on whatever's left after regex masking. Persons,
   addresses, and edge-case dates fall here. The model's noise-floor
   issue (false positives on PII-free text) is partially mitigated
   because the regex already absorbs the most-confused categories.

Per CLAUDE.md Phase 5a, the regex front-end specifically targets the
categories where (a) shape is verifiable, (b) the model has a known
weakness (private_url and account_number FPs in the bake-off), and (c)
a checksum-failure should fall through to the model rather than being
silently wrong.

The composite is wrapped in the same ``Detector`` interface
(``detect(text) -> list[Span]``) as ``LocalDetector`` so it drops into
the same eval harness with no plumbing changes.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Protocol

from .base import Span


class _ModelDetector(Protocol):
    """Anything with a ``detect(text) -> list[Span]`` method.

    Composite delegates to whatever detector the caller passes in. Tests
    pass a fake; production passes a wrapper around a fine-tuned model.
    """

    def detect(self, text: str) -> list[Span]: ...

# --- Regex catalogue ---
#
# Format: (label, compiled regex, optional checksum validator). Order is
# meaningful for overlap resolution — earlier entries win on tie.
# Validators receive the matched text and return True iff the checksum
# checks out; None means no checksum check.
ChecksumFn = Any  # (str) -> bool


def _luhn_ok(s: str) -> bool:
    """Luhn check for credit card numbers (digits only, length 13-19)."""
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ch_ok(s: str) -> bool:
    """ISO 13616 mod-97-10 check on Swiss IBAN (CH + 19 digits including check)."""
    s = s.replace(" ", "").replace("-", "")
    if not (len(s) == 21 and s.startswith("CH") and s[2:].isdigit()):
        return False
    rearranged = s[4:] + s[0:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    return int(numeric) % 97 == 1


def _ahv_ok(s: str) -> bool:
    """EAN-13 mod-10 check on Swiss AHV (756.XXXX.XXXX.XX or 756XXXXXXXXXX)."""
    digits = [int(c) for c in s if c.isdigit()]
    if not (len(digits) == 13 and digits[0:3] == [7, 5, 6]):
        return False
    body = digits[:12]
    weights = [1 if i % 2 == 0 else 3 for i in range(12)]
    total = sum(d * w for d, w in zip(body, weights, strict=True))
    check = (10 - (total % 10)) % 10
    return digits[12] == check


def _vat_che_ok(s: str) -> bool:
    """ISO 7064 mod-11-10 check on Swiss VAT (CHE-XXX.XXX.XXX with optional MWST suffix)."""
    digits = [int(c) for c in s if c.isdigit()]
    if len(digits) < 9:
        return False
    body = digits[:8]
    check = digits[8]
    weights = (5, 4, 3, 2, 7, 6, 5, 4)
    total = sum(d * w for d, w in zip(body, weights, strict=True))
    expected = 11 - (total % 11)
    if expected == 11:
        expected = 0
    elif expected == 10:
        return False  # invalid by design
    return check == expected


ContextFilterFn = Callable[[str, int, int], bool]
"""``(text, start, end) -> True if the match should be REJECTED``.

Context filters run after the format regex matches and after the
checksum (if any) has passed. They get the surrounding text so they
can tell e.g.\\ a credit card from a numeric-table row whose digits
happen to Luhn-pass by chance (~10% of random 16-digit strings do).
``None`` means no context check, i.e.\\ the regex's checksum is
trusted on its own.
"""


def _looks_like_numeric_list(value: str) -> bool:
    """True if ``value`` looks like a list of separate small integers
    (a salary scale, page numbers, a year range) that the credit-card
    regex matched because the whole row is 15-17 digits.

    Concrete v1.5 casualty caught by the P7 audit:
    ``"18 19 20 21 22 23 24 25"`` — a French salary-scale row in a
    court-decision table. Eight 2-digit tokens, all separated by single
    spaces, monotonic — visibly NOT a credit card.
    """
    tokens = re.findall(r"\d+", value)
    if len(tokens) < 6:
        return False
    # Most tokens are 1-3 digits — indicates list, not a CC chunk.
    short_tokens = sum(1 for t in tokens if len(t) <= 3)
    return short_tokens >= len(tokens) - 1


def _in_numeric_table(text: str, start: int, end: int) -> bool:
    """True if the credit-card-shaped match is actually noise: a
    numeric-table row, a salary scale, or an IMEI.

    Why this exists: the credit-card regex `\\b(?:\\d[\\s\\-]?){15,16}\\d\\b`
    is intentionally lax (Visa / Mastercard / Amex / Diner's all need
    to fit). Combined with Luhn's 10\\% accidental pass rate, that
    means random tabulated stats from court documents and
    parliamentary records get matched as credit cards. Concrete v1
    casualties caught by the P7 four-LLM audit:
    `'2011 2011 2011 1-3 1-3 1'`, `'1892 101 14123 1519'`,
    `'1894 159 4407 52468'`. v1.5 audit added two more:
    `'18 19 20 21 22 23 24 25'` (salary scale) and
    `'3533301011754332'` (IMEI of a phone).

    Three escape hatches keep real cards through:
    1. Vocabulary words near the match (``Karte``, ``card``, ``Visa``,
       ``CVV``, etc.) — signals an actual credit card even in tables.
    2. The match itself is NOT a list of small integers.
    3. The surrounding window is NOT mostly numeric tokens.
    4. The match is NOT in IMEI / DOI / identifier context.
    """
    value = text[start:end]
    pre_window = text[max(0, start - 80):start].lower()
    post_window = text[end:end + 30].lower()

    cc_context = ("karte", "card", "kreditkarte", "visa", "mastercard",
                  "amex", "cvv", "kontonummer", "pan ", "credit ")
    if any(w in pre_window for w in cc_context):
        return False  # real card vocabulary nearby — keep

    # IMEI / device-id context: explicit "imei", "serial", "seriennummer"
    # words within ~80 chars before the match — almost certainly a
    # device identifier, not a payment card.
    id_context = ("imei", "serial", "seriennummer", "doi:", "doi ",
                  "issn", "isbn", "patent")
    if any(w in pre_window for w in id_context):
        return True

    # The match itself looks like a list of small integers (salary
    # scales, page numbers, year ranges).
    if _looks_like_numeric_list(value):
        return True

    # The surrounding window is mostly numeric tokens — likely a
    # statistics row.
    around = text[max(0, start - 30):start] + " " + post_window
    nums = re.findall(r"(?<!\w)\d+(?!\w)", around)
    return len(nums) >= 3


_DOI_PREFIX_RE = re.compile(r"\b10\.\d{4,5}/")
"""Real DOIs start with the 10.NNNN/ registrant prefix, never with a
4-digit year + dot. This regex distinguishes ``10.1249/01.MSS...``
(real DOI) from ``10.02.2022`` (Swiss date)."""


def _is_implausible_swiss_phone(text: str, start: int, end: int) -> bool:
    """True if the Swiss-phone match doesn't look like a real phone.

    The phone regex's ``\\b0`` alternative also matches any 10-digit
    run starting with 0 — fine for Swiss landlines that start with 0,
    but it also fires inside DOI / accession identifiers. Concrete v1.5
    casualty: ``'0000121945'`` extracted from
    ``'DOI: 10.1249/01.MSS.0000121945.36635.61'``.

    Reject the match if:
    - it has 4 or more leading zeros (no real Swiss phone does), OR
    - the surrounding text contains explicit identifier vocabulary
      (DOI, ISSN, ISBN, patent) or a real DOI prefix (``10.NNNN/``).

    Important: do NOT reject on bare ``10.`` substring; that would
    catch any Swiss date written ``DD.MM.YYYY`` (e.g.\\ ``10.02.2022``)
    which often appears next to a real phone in contact blocks.
    """
    value = text[start:end]
    digits = re.sub(r"\D", "", value)
    if digits.startswith("0000"):
        return True
    pre_window = text[max(0, start - 80):start].lower()
    id_context = ("doi:", "doi ", "issn", "isbn", "patent")
    if any(w in pre_window for w in id_context):
        return True
    if _DOI_PREFIX_RE.search(pre_window):
        return True
    return False


# Each entry: (label, regex, validator, context_filter, subtype). The
# subtype is a string tag that identifies which sub-pattern matched
# (e.g. "iban_ch" vs "ahv" vs "credit_card") — useful for downstream
# provenance / dataset-build code that wants to know which deterministic
# rule fired, even though all four collapse to ``account_number`` in
# the published 8-class label space. ``None`` subtype means the label
# itself is the only identifier worth keeping.
_REGEXES: list[tuple[
    str,
    re.Pattern[str],
    ChecksumFn | None,
    ContextFilterFn | None,
    str | None,
]] = [
    ("private_email", re.compile(r"\b[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b"), None, None, None),
    ("private_url", re.compile(r"https?://[^\s)>\]]+"), None, None, None),
    ("private_phone", re.compile(
        r"(?:\+41|0041|\b0)\s?(?:\(0\)\s?)?\d{2}[\s\-/.]?\d{3}[\s\-/.]?\d{2}[\s\-/.]?\d{2}\b"
    ), None, _is_implausible_swiss_phone, "swiss_phone"),
    ("account_number", re.compile(r"\bCH\d{2}(?:[\s\-]?\d){17}\b"), _iban_ch_ok, None, "iban_ch"),
    ("account_number", re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b"), _ahv_ok, None, "ahv"),
    ("account_number", re.compile(r"\b756\d{10}\b"), _ahv_ok, None, "ahv"),
    ("account_number", re.compile(
        r"\bCHE-\d{3}\.\d{3}\.\d{3}(?:\s+(?:MWST|TVA|IVA))?\b"
    ), _vat_che_ok, None, "vat_che"),
    # Credit cards: Luhn + numeric-table-context filter.
    ("account_number", re.compile(r"\b(?:\d[\s\-]?){15,16}\d\b"),
     _luhn_ok, _in_numeric_table, "credit_card"),
    # Dates: require day + month + 4-digit year. M/YYYY rejected.
    ("private_date", re.compile(
        r"\b(?:\d{1,2}[./]\d{1,2}[./]\d{4}"
        r"|\d{4}[./\-]\d{1,2}[./\-]\d{1,2})\b"
    ), None, None, None),
    ("secret", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"), None, None, "openai_key"),
    ("secret", re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), None, None, "github_pat"),
    ("secret", re.compile(r"\bAKIA[A-Z0-9]{16}\b"), None, None, "aws_access_key"),
]


def _trim_url(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip trailing sentence punctuation from a URL match."""
    while end > start and text[end - 1] in ".,;:!?\")>]":
        end -= 1
    return start, end


def _find_regex_spans_with_subtype(text: str) -> list[tuple[Span, str | None]]:
    """Like :func:`_find_regex_spans` but also tags each span with the
    subtype of the rule that fired (e.g. ``"iban_ch"``, ``"ahv"``,
    ``"credit_card"``). Used by the v2 dataset-build pipeline that
    records per-span provenance; the runtime detector uses the
    subtype-less wrapper :func:`_find_regex_spans`."""
    candidates: list[tuple[Span, str | None]] = []
    for label, pat, validator, context_filter, subtype in _REGEXES:
        for m in pat.finditer(text):
            start, end = m.start(), m.end()
            if label == "private_url":
                start, end = _trim_url(text, start, end)
            value = text[start:end]
            if not value.strip():
                continue
            if validator is not None and not validator(value):
                continue  # checksum failed — let the model decide
            if context_filter is not None and context_filter(text, start, end):
                continue  # surrounding context says this is not really PII
            candidates.append((
                Span(start=start, end=end, label=label, text=value, score=1.0),
                subtype,
            ))
    candidates.sort(key=lambda x: (x[0].start, -(x[0].end - x[0].start)))
    out: list[tuple[Span, str | None]] = []
    last_end = -1
    for sp, sub in candidates:
        if sp.start < last_end:
            continue
        out.append((sp, sub))
        last_end = sp.end
    return out


def _find_regex_spans(text: str) -> list[Span]:
    """Apply all regexes (with validators + context filters) and return
    non-overlapping Span list, greedy by start offset, longer-on-tie.

    Subtype information (which specific rule fired) is dropped here;
    callers that need it should use :func:`_find_regex_spans_with_subtype`.
    """
    return [sp for sp, _ in _find_regex_spans_with_subtype(text)]


def _mask_spans(text: str, spans: list[Span]) -> str:
    """Replace each span's characters with the same number of spaces.

    Length-preserving so the model's predicted offsets match the original
    text 1:1 — no offset translation needed when merging predictions.
    """
    if not spans:
        return text
    chars = list(text)
    for sp in spans:
        for i in range(sp.start, sp.end):
            if i < len(chars):
                chars[i] = " "
    return "".join(chars)


@dataclass
class CompositeDetector:
    """Regex front-end (with checksum validation) + fine-tuned model on
    the remainder.

    The model can be ANY object that exposes ``detect(text) -> list[Span]``.
    In production, pass a wrapper around the bake-off leader (HFDetector
    from the training package, or a HuggingFace pipeline wrapped in any
    detector class). The composite does not import a specific model class
    so it doesn't accidentally couple to one detector implementation's
    post-processing (e.g. LocalDetector returns fragmented subword spans;
    HFDetector merges them — using LocalDetector here would 13× the
    composite's FP rate on PII-free controls).
    """

    # Typed as Any rather than _ModelDetector because callers may pass
    # detectors that return Span dataclasses from a different package
    # (e.g. gheim_training's Span); structurally compatible but ty can't
    # prove the protocol match across packages.
    model: Annotated[
        Any,
        "Pre-constructed model detector (must expose detect(text) -> list[Span]).",
    ]

    def detect(self, text: str) -> list[Span]:
        """Two-pass detection: regex first (with checksum), then model on
        the masked remainder. Returns a single overlap-resolved span list.
        """
        if not text:
            return []
        regex_spans = _find_regex_spans(text)
        masked = _mask_spans(text, regex_spans)
        model_spans = self.model.detect(masked)
        # Overlap resolution: regex wins on any overlap (regex spans are
        # high-confidence / checksum-validated). Then model spans fill
        # the gaps.
        all_spans = list(regex_spans)
        regex_ranges = [(s.start, s.end) for s in regex_spans]
        for sp in model_spans:
            if any(sp.start < re_end and re_start < sp.end
                   for re_start, re_end in regex_ranges):
                continue  # overlaps a regex span — drop the model claim
            all_spans.append(sp)
        all_spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        return all_spans
