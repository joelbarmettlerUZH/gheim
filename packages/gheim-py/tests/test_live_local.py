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


def test_long_input_is_not_silently_truncated(detector: LocalDetector) -> None:
    """Regression: pre-chunking, anything past ~512 tokens was silently
    dropped because the underlying transformers pipeline truncates by
    default. Plant PII near the start, middle, and end of a long
    document and assert all three are detected."""
    pad = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 80  # ~4.6k chars
    text = (
        f"Hi, my name is Alice Smith. {pad}"
        f"My colleague Bob Jones is also on this. {pad}"
        f"Please CC carol@example.com on the response."
    )
    assert len(text) > 9000  # comfortably past the 512-token limit
    spans = detector.detect(text)
    persons_text = " ".join(s.text for s in spans if s.label == "private_person")
    emails_text = " ".join(s.text for s in spans if s.label == "private_email")
    assert "Alice" in persons_text, f"missed PII near start: {spans}"
    assert "Bob" in persons_text, f"missed PII near middle: {spans}"
    assert "carol" in emails_text, f"missed PII near end: {spans}"


def test_detect_batch_processes_multiple_inputs(detector: LocalDetector) -> None:
    """Cross-input batching: three short texts go through one batched
    forward pass and each gets its own span list."""
    texts = [
        "Hi, my name is Alice Smith.",
        "Email me at bob@example.com please.",
        "Nothing personal here.",
    ]
    results = detector.detect_batch(texts)
    assert len(results) == 3
    assert any(s.label == "private_person" for s in results[0])
    assert any(s.label == "private_email" for s in results[1])
    assert results[2] == []  # negative — no PII present
