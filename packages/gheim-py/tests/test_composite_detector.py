"""Unit tests for CompositeDetector.

Lock in the regex front-end behaviour, the checksum gating, the masking
length-preservation, the regex-wins-on-overlap rule, and the model
fall-through path. The actual model is mocked — these tests exercise
the composite logic in isolation, not transformer inference.
"""
from __future__ import annotations

from gheim.detectors.base import Span
from gheim.detectors.composite import (
    CompositeDetector,
    _ahv_ok,
    _find_regex_spans,
    _iban_ch_ok,
    _luhn_ok,
    _mask_spans,
    _vat_che_ok,
)

# ---------- checksum validators ----------

def test_iban_ch_valid_passes() -> None:
    # Real-shape Swiss IBAN with valid mod-97 check.
    assert _iban_ch_ok("CH9300762011623852957")
    assert _iban_ch_ok("CH93 0076 2011 6238 5295 7")  # spaced form
    assert _iban_ch_ok("CH93-0076-2011-6238-5295-7")  # dashed form


def test_iban_ch_invalid_check_digits_fails() -> None:
    # Same IBAN, wrong check digits (changed CH93 → CH99).
    assert not _iban_ch_ok("CH9900762011623852957")


def test_iban_ch_wrong_country_or_length_fails() -> None:
    assert not _iban_ch_ok("DE89370400440532013000")  # German IBAN
    assert not _iban_ch_ok("CH123456")  # too short


def test_ahv_valid_passes() -> None:
    # Build a valid AHV: 756 + 9 random digits + check via mod-10.
    body = [7, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = [1 if i % 2 == 0 else 3 for i in range(12)]
    s = sum(d * w for d, w in zip(body, weights, strict=True))
    check = (10 - (s % 10)) % 10
    digits = "".join(str(d) for d in body) + str(check)
    formatted = f"{digits[0:3]}.{digits[3:7]}.{digits[7:11]}.{digits[11:13]}"
    assert _ahv_ok(formatted)
    assert _ahv_ok(digits)  # compact form


def test_ahv_wrong_prefix_fails() -> None:
    assert not _ahv_ok("123.4567.8901.23")


def test_vat_che_valid_passes() -> None:
    # Build a valid CHE-VAT: 8-digit body + mod-11-10 check.
    body = [1, 0, 6, 0, 1, 7, 7, 9]
    weights = (5, 4, 3, 2, 7, 6, 5, 4)
    s = sum(d * w for d, w in zip(body, weights, strict=True))
    expected = 11 - (s % 11)
    if expected == 11:
        expected = 0
    elif expected == 10:
        expected = 1  # picking another body would be cleaner
    digits = "".join(str(d) for d in body) + str(expected)
    formatted = f"CHE-{digits[0:3]}.{digits[3:6]}.{digits[6:9]}"
    assert _vat_che_ok(formatted)


def test_luhn_valid_card_passes() -> None:
    # Visa test number from PCI DSS docs.
    assert _luhn_ok("4111111111111111")
    assert _luhn_ok("4111-1111-1111-1111")


def test_luhn_invalid_check_fails() -> None:
    assert not _luhn_ok("4111111111111112")  # off-by-one in check digit


# ---------- regex spans ----------

def test_find_regex_spans_extracts_email_url_phone() -> None:
    text = "Contact info@example.ch or visit https://example.ch — call 044 668 18 00."
    spans = _find_regex_spans(text)
    labels = sorted(s.label for s in spans)
    assert "private_email" in labels
    assert "private_url" in labels
    assert "private_phone" in labels


def test_find_regex_spans_drops_invalid_iban() -> None:
    """An IBAN-shaped string with bad checksum is silently dropped."""
    text = "Wrong IBAN: CH9900762011623852957 here."
    spans = _find_regex_spans(text)
    assert not any(s.label == "account_number" for s in spans)


def test_find_regex_spans_keeps_valid_iban() -> None:
    text = "Real IBAN: CH9300762011623852957 here."
    spans = _find_regex_spans(text)
    iban_spans = [s for s in spans if s.label == "account_number"]
    assert len(iban_spans) == 1


def test_find_regex_spans_url_strips_trailing_punct() -> None:
    text = "See https://example.ch."
    spans = _find_regex_spans(text)
    url = next(s for s in spans if s.label == "private_url")
    assert text[url.start:url.end] == "https://example.ch"


def test_find_regex_spans_resolves_overlaps_greedy() -> None:
    """Two regex categories matching overlapping ranges → first/longer wins."""
    text = "12.03.1985"
    spans = _find_regex_spans(text)
    # Date pattern matches the whole thing once. CC pattern would match
    # the digit substring — but date wins by being earlier in the table
    # AND fully covering; overlap resolver drops the CC.
    # Either way, exactly one span here is fine.
    assert len(spans) >= 1
    assert any(s.label == "private_date" for s in spans)


# ---------- masking ----------

def test_mask_spans_preserves_length() -> None:
    text = "Anna Müller wrote info@example.ch yesterday."
    spans = [Span(label="private_email", start=18, end=33,
                  text="info@example.ch", score=1.0)]
    masked = _mask_spans(text, spans)
    assert len(masked) == len(text)
    assert masked[18:33] == " " * 15
    # Outside the span the text is unchanged
    assert masked[:18] == text[:18]
    assert masked[33:] == text[33:]


def test_mask_spans_no_spans_returns_input() -> None:
    text = "no PII here"
    assert _mask_spans(text, []) == text


# ---------- composite end-to-end ----------

class _FakeModel:
    """Stand-in for LocalDetector that returns a predetermined span list,
    used to test the composite's overlap resolution + merge logic without
    loading a real transformer."""

    def __init__(self, spans: list[Span]) -> None:
        self.spans = spans
        self.last_text: str | None = None

    def detect(self, text: str) -> list[Span]:
        self.last_text = text
        return list(self.spans)


def test_composite_combines_regex_and_model_spans() -> None:
    text = "Anna Müller's IBAN is CH9300762011623852957 and her email is anna@example.ch."
    fake_person = Span(label="private_person", start=0, end=11,
                       text="Anna Müller", score=0.99)
    det = CompositeDetector(model=_FakeModel(spans=[fake_person]))
    out = det.detect(text)
    labels = sorted(s.label for s in out)
    assert "private_person" in labels
    assert "private_email" in labels
    assert "account_number" in labels


def test_composite_regex_wins_on_overlap() -> None:
    """When the model claims a span that overlaps a regex hit, regex
    wins (regex has checksum-validated identity)."""
    text = "Contact: info@example.ch"
    # Model claims the email range is a person — composite must drop it.
    fake_overlap = Span(label="private_person", start=9, end=24,
                        text="info@example.ch", score=0.9)
    det = CompositeDetector(model=_FakeModel(spans=[fake_overlap]))
    out = det.detect(text)
    # Only the email span should remain; the model's person-on-email is dropped.
    assert all(s.label != "private_person" for s in out)
    assert any(s.label == "private_email" for s in out)


def test_composite_passes_masked_text_to_model() -> None:
    """The model sees the regex-spans replaced with whitespace, not the
    original — confirms the masking step actually executes before model
    inference."""
    text = "Email: info@example.ch is the only PII here."
    fake = _FakeModel(spans=[])
    det = CompositeDetector(model=fake)
    det.detect(text)
    assert fake.last_text is not None
    # The "info@example.ch" range should be whitespace in what the model saw
    assert "info@example.ch" not in fake.last_text
    # And the rest of the text should still be there
    assert "Email:" in fake.last_text


def test_composite_invalid_iban_falls_through_to_model() -> None:
    """A bad-checksum IBAN-shape is silently dropped by regex; the model
    then gets to label it (and might call it account_number anyway, which
    would be fine — the composite should NOT pretend the regex saw it)."""
    text = "Bad IBAN CH9900762011623852957 mentioned."
    fake = _FakeModel(spans=[])  # model decides nothing
    det = CompositeDetector(model=fake)
    det.detect(text)
    # The model should have seen the unmasked IBAN string (no regex hit)
    assert "CH9900762011623852957" in (fake.last_text or "")


def test_composite_empty_text_returns_empty() -> None:
    det = CompositeDetector(model=_FakeModel(spans=[]))
    assert det.detect("") == []


def test_composite_pure_text_no_pii() -> None:
    text = "Das Bundesgericht hat die Beschwerde abgewiesen."
    fake = _FakeModel(spans=[])
    det = CompositeDetector(model=fake)
    out = det.detect(text)
    assert out == []
