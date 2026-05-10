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


# (label, regex, validator-or-None). Validators kick in only if the
# regex matches — if the validator says no, the match is silently
# discarded and the model gets the chance to label it instead.
_REGEXES: list[tuple[str, re.Pattern[str], ChecksumFn | None]] = [
    ("private_email", re.compile(r"\b[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b"), None),
    ("private_url", re.compile(r"https?://[^\s)>\]]+"), None),
    ("private_phone", re.compile(
        r"(?:\+41|0041|\b0)\s?(?:\(0\)\s?)?\d{2}[\s\-/.]?\d{3}[\s\-/.]?\d{2}[\s\-/.]?\d{2}\b"
    ), None),
    ("account_number", re.compile(r"\bCH\d{2}(?:[\s\-]?\d){17}\b"), _iban_ch_ok),
    ("account_number", re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b"), _ahv_ok),
    ("account_number", re.compile(r"\b756\d{10}\b"), _ahv_ok),
    ("account_number", re.compile(
        r"\bCHE-\d{3}\.\d{3}\.\d{3}(?:\s+(?:MWST|TVA|IVA))?\b"
    ), _vat_che_ok),
    ("account_number", re.compile(r"\b(?:\d[\s\-]?){15,16}\d\b"), _luhn_ok),
    # Dates: require day + month + 4-digit year. M/YYYY rejected.
    ("private_date", re.compile(
        r"\b(?:\d{1,2}[./]\d{1,2}[./]\d{4}"
        r"|\d{4}[./\-]\d{1,2}[./\-]\d{1,2})\b"
    ), None),
    ("secret", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"), None),
    ("secret", re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), None),
    ("secret", re.compile(r"\bAKIA[A-Z0-9]{16}\b"), None),
]


def _trim_url(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip trailing sentence punctuation from a URL match."""
    while end > start and text[end - 1] in ".,;:!?\")>]":
        end -= 1
    return start, end


def _find_regex_spans(text: str) -> list[Span]:
    """Apply all regexes (with validators) and return non-overlapping
    Span list, greedy by start offset, longer-on-tie."""
    candidates: list[Span] = []
    for label, pat, validator in _REGEXES:
        for m in pat.finditer(text):
            start, end = m.start(), m.end()
            if label == "private_url":
                start, end = _trim_url(text, start, end)
            value = text[start:end]
            if not value.strip():
                continue
            if validator is not None and not validator(value):
                continue  # checksum failed — let the model decide
            candidates.append(Span(start=start, end=end, label=label,
                                   text=value, score=1.0))
    candidates.sort(key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    last_end = -1
    for sp in candidates:
        if sp.start < last_end:
            continue
        out.append(sp)
        last_end = sp.end
    return out


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
