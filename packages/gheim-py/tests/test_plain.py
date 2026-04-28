"""End-to-end round-trip tests using a fake detector — no model required."""
from gheim import (
    Session,
    Span,
    anonymize_messages,
    anonymize_text,
    deanonymize_stream,
    deanonymize_text,
)
from gheim.detectors.base import Detector


class FakeDetector:
    """Returns spans for a fixed set of surface forms — deterministic, no model."""

    def __init__(self, surfaces: dict[str, str]):
        # surfaces: {surface_form: label}
        self._surfaces = surfaces

    def detect(self, text: str, *, model: str | None = None) -> list[Span]:
        spans: list[Span] = []
        for surface, label in self._surfaces.items():
            start = 0
            while True:
                idx = text.find(surface, start)
                if idx == -1:
                    break
                spans.append(Span(label=label, start=idx, end=idx + len(surface), text=surface))
                start = idx + len(surface)
        return spans


def test_protocol_check():
    # Sanity: FakeDetector satisfies the Detector protocol structurally.
    d: Detector = FakeDetector({})
    assert hasattr(d, "detect")


def test_anonymize_text_basic():
    det = FakeDetector({"Joel": "private_person"})
    s = Session(detector=det)
    out = anonymize_text("My name is Joel.", s)
    assert out == "My name is <PERSON_1>."
    assert s.mapping == {"<PERSON_1>": "Joel"}


def test_anonymize_text_repeated_surface_coalesces():
    det = FakeDetector({"Joel": "private_person"})
    s = Session(detector=det)
    out = anonymize_text("Joel met Joel.", s)
    assert out == "<PERSON_1> met <PERSON_1>."


def test_deanonymize_text_roundtrip():
    det = FakeDetector({"Joel": "private_person", "joel@x.ch": "private_email"})
    s = Session(detector=det)
    redacted = anonymize_text("Joel at joel@x.ch", s)
    # Simulated LLM response references the same sentinels.
    response = f"Hi {redacted.split()[0]}, your email {redacted.split()[-1]} is registered."
    restored = deanonymize_text(response, s)
    assert restored == "Hi Joel, your email joel@x.ch is registered."


def test_anonymize_messages_string_content():
    det = FakeDetector({"Joel": "private_person"})
    s = Session(detector=det)
    msgs = [
        {"role": "system", "content": "You speak to Joel."},
        {"role": "user", "content": "Hi from Joel"},
    ]
    out = anonymize_messages(msgs, s)
    assert out[0]["content"] == "You speak to <PERSON_1>."
    assert out[1]["content"] == "Hi from <PERSON_1>"
    # Original list is not mutated.
    assert msgs[0]["content"] == "You speak to Joel."


def test_anonymize_messages_multimodal_content():
    det = FakeDetector({"Joel": "private_person"})
    s = Session(detector=det)
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is Joel"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
            ],
        }
    ]
    out = anonymize_messages(msgs, s)
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "This is <PERSON_1>"}
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/x.png"},
    }


def test_anonymize_messages_passes_through_extra_fields():
    det = FakeDetector({"Joel": "private_person"})
    s = Session(detector=det)
    msgs = [{"role": "tool", "content": "Joel called", "tool_call_id": "abc-123", "name": "fn"}]
    out = anonymize_messages(msgs, s)
    assert out[0]["tool_call_id"] == "abc-123"
    assert out[0]["name"] == "fn"
    assert out[0]["content"] == "<PERSON_1> called"


def test_deanonymize_stream_full_roundtrip():
    det = FakeDetector({"Joel": "private_person"})
    s = Session(detector=det)
    anonymize_text("Hi Joel", s)
    chunks = ["Hi ", "<PER", "SON_1>", ", how can I help?"]
    restored = "".join(deanonymize_stream(chunks, s))
    assert restored == "Hi Joel, how can I help?"


def test_session_persists_across_calls():
    det = FakeDetector({"Joel": "private_person", "Alice": "private_person"})
    s = Session(detector=det)
    anonymize_text("Joel", s)
    anonymize_text("Alice met Joel", s)
    # Joel keeps its sentinel; Alice gets a new one.
    assert s.mapping == {"<PERSON_1>": "Joel", "<PERSON_2>": "Alice"}
