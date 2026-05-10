"""Real-model end-to-end tests for LocalDetector + plain + openai surfaces.

These download `openai/privacy-filter` (~3 GB on first run) and run inference on
CPU. Skipped by default — set ``GHEIM_RUN_LIVE=1`` to enable.
"""
from __future__ import annotations

import pytest
from gheim import LocalDetector, Session, anonymize_text, deanonymize_stream, deanonymize_text

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def detector() -> LocalDetector:
    return LocalDetector(device="cpu")


def test_local_detector_finds_name(detector: LocalDetector) -> None:
    spans = detector.detect("Hi, my name is Alice Smith.")
    persons = [s for s in spans if s.label == "private_person"]
    assert persons, f"expected a private_person span, got: {spans}"
    assert "Alice" in "".join(p.text for p in persons)


def test_local_detector_finds_email(detector: LocalDetector) -> None:
    spans = detector.detect("Email me at alice@example.com please.")
    emails = [s for s in spans if s.label == "private_email"]
    assert emails, f"expected a private_email span, got: {spans}"


def test_anonymize_text_round_trip(detector: LocalDetector) -> None:
    session = Session(detector=detector)
    text = "Hi, my name is Alice and my email is alice@example.com."
    redacted = anonymize_text(text, session)
    # At least one sentinel should have been allocated.
    assert "<PERSON_" in redacted or "<EMAIL_" in redacted
    # And restoring the redacted text should reproduce the original.
    restored = deanonymize_text(redacted, session)
    assert restored == text


def test_deanonymize_stream_with_real_session(detector: LocalDetector) -> None:
    session = Session(detector=detector)
    redacted = anonymize_text("Hi, my name is Alice.", session)
    person_sentinel = next(s for s in session.mapping if s.startswith("<PERSON_"))
    # Simulate an LLM that emits the sentinel split across chunks.
    chunks = ["Nice to meet you, ", person_sentinel[:4], person_sentinel[4:], "!"]
    restored = "".join(deanonymize_stream(chunks, session))
    assert "Alice" in restored
    assert person_sentinel not in restored
    _ = redacted  # keep the redacted form alive for debug-on-failure clarity
