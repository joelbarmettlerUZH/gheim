"""Correctness edge cases (Batch B) — unicode, overflow, empty input,
cross-language session JSON, repeated surfaces across calls."""
from __future__ import annotations

import json

import pytest
from gheim import Session, Span, anonymize_text, deanonymize_text
from gheim.core.stream import MAX_SENTINEL_LEN, StreamDeanonymizer

# ---------- B.1 multibyte unicode in stream ----------

def test_stream_handles_cjk_around_sentinel():
    mapping = {"<PERSON_1>": "Joel"}
    chunks = ["你好 ", "<PER", "SON_1>", " 안녕"]
    out = "".join(StreamDeanonymizer(mapping).transform(chunks))
    assert out == "你好 Joel 안녕"


def test_stream_handles_emoji_around_sentinel():
    mapping = {"<PERSON_1>": "Joel"}
    chunks = ["🚀 hello ", "<PERSON_1>", " 🎉"]
    out = "".join(StreamDeanonymizer(mapping).transform(chunks))
    assert out == "🚀 hello Joel 🎉"


def test_stream_handles_unicode_in_restored_surface():
    mapping = {"<PERSON_1>": "André"}
    chunks = ["Bonjour ", "<PER", "SON_1>", " — ravi"]
    out = "".join(StreamDeanonymizer(mapping).transform(chunks))
    assert out == "Bonjour André — ravi"


def test_anonymize_unicode_text_round_trip():
    """Detected surface is unicode; spans use Python char offsets, must round-trip cleanly."""
    text = "Hello 你好 André at andré@exämple.com."
    # Span offsets are character (code point) indices in Python.
    person_start = text.index("André")
    email_start = text.index("andré@exämple.com")
    spans = [
        Span(label="private_person", start=person_start, end=person_start + len("André"), text="André"),
        Span(label="private_email", start=email_start, end=email_start + len("andré@exämple.com"), text="andré@exämple.com"),
    ]
    sess = Session()
    redacted = sess.apply_spans(text, spans)
    assert redacted == "Hello 你好 <PERSON_1> at <EMAIL_1>."
    restored = sess.restore(redacted)
    assert restored == text


# ---------- B.2 '<' followed by long non-sentinel proves overflow path ----------

def test_stream_long_non_sentinel_starting_with_lt_emits_verbatim():
    """A `<` followed by content that exceeds MAX_SENTINEL_LEN must emit verbatim
    rather than holding the buffer indefinitely."""
    junk = "<" + "A" * (MAX_SENTINEL_LEN + 16)  # exceed by a comfortable margin
    out = "".join(StreamDeanonymizer({"<PERSON_1>": "Joel"}).transform([junk]))
    assert out == junk


def test_stream_lt_followed_by_huge_block_then_real_sentinel():
    """Pathological mix: `<` + long block of A's + `>` + real sentinel afterwards."""
    pre = "<" + "A" * (MAX_SENTINEL_LEN + 4) + ">"
    text = pre + " then <PERSON_1>"
    out = "".join(StreamDeanonymizer({"<PERSON_1>": "Joel"}).transform(list(text)))
    assert out == pre + " then Joel"


def test_stream_many_consecutive_left_angles():
    """`<<<<<<<<<<<<` should not stall; each `<` is emitted as soon as the next char disqualifies."""
    out = "".join(StreamDeanonymizer({}).transform(["<<<<<<<<"]))
    assert out == "<<<<<<<<"


# ---------- B.3 empty / whitespace / pathological inputs ----------

def test_anonymize_empty_text():
    sess = Session()
    assert anonymize_text("", sess, detector=_NoopDetector()) == ""
    assert sess.mapping == {}


def test_anonymize_whitespace_only_text():
    sess = Session()
    out = anonymize_text("   \n\t  ", sess, detector=_NoopDetector())
    assert out == "   \n\t  "


def test_deanonymize_empty_text():
    sess = Session()
    sess.allocate("private_person", "Joel")
    assert deanonymize_text("", sess) == ""


def test_stream_only_a_single_left_angle_across_chunks():
    out = "".join(StreamDeanonymizer({}).transform(["<", " ", "x"]))
    assert out == "< x"


def test_stream_lone_lt_at_end_of_stream_flushed_verbatim():
    out = "".join(StreamDeanonymizer({}).transform(["text <"]))
    assert out == "text <"


# ---------- B.4 repeated surface across multiple anonymize_text calls ----------

def test_repeated_surface_across_anonymize_calls_keeps_same_sentinel():
    det = _SurfaceDetector({"Joel": "private_person"})
    sess = Session(detector=det)
    a = anonymize_text("Joel was here.", sess)
    b = anonymize_text("Then Joel left.", sess)
    c = anonymize_text("Joel is back.", sess)
    assert a == "<PERSON_1> was here."
    assert b == "Then <PERSON_1> left."
    assert c == "<PERSON_1> is back."
    assert sess.mapping == {"<PERSON_1>": "Joel"}


def test_repeated_surface_distinct_labels_get_distinct_sentinels():
    det = _SurfaceDetector({"Joel": "private_person", "joel@x.ch": "private_email"})
    sess = Session(detector=det)
    out = anonymize_text("Joel at joel@x.ch wrote", sess)
    assert out == "<PERSON_1> at <EMAIL_1> wrote"


# ---------- B.5 cross-language Session JSON round-trip ----------

def test_session_to_json_is_a_simple_dict_for_cross_language_use():
    sess = Session()
    sess.allocate("private_person", "Joel")
    sess.allocate("private_email", "joel@example.com")
    sess.allocate("private_person", "Alice")
    raw = sess.to_json()
    parsed = json.loads(raw)
    # Stable shape — JS gheim expects exactly this.
    assert set(parsed.keys()) == {"mapping"}
    assert parsed["mapping"] == {
        "<PERSON_1>": "Joel",
        "<EMAIL_1>": "joel@example.com",
        "<PERSON_2>": "Alice",
    }


def test_session_from_json_accepts_cross_language_payload():
    """A payload produced by another language (e.g. JS Session.toJSON()) restores cleanly."""
    foreign_payload = json.dumps({
        "mapping": {
            "<PERSON_1>": "Joel",
            "<EMAIL_1>": "joel@example.com",
        }
    })
    sess = Session.from_json(foreign_payload)
    assert sess.mapping == {"<PERSON_1>": "Joel", "<EMAIL_1>": "joel@example.com"}
    # Counters recovered → next allocation does not collide with restored sentinels.
    assert sess.allocate("private_person", "Alice") == "<PERSON_2>"
    assert sess.allocate("private_person", "Joel") == "<PERSON_1>"  # coalesce


def test_session_from_json_ignores_unknown_keys():
    foreign_payload = json.dumps({
        "mapping": {"<PERSON_1>": "Joel"},
        "version": "0.1.0",
        "extra_field_we_dont_know": True,
    })
    sess = Session.from_json(foreign_payload)
    assert sess.mapping == {"<PERSON_1>": "Joel"}


def test_session_json_round_trip_with_unicode():
    sess = Session()
    sess.allocate("private_person", "André")
    sess.allocate("private_email", "anna@bürgerportal.ch")
    rebuilt = Session.from_json(sess.to_json())
    assert rebuilt.mapping == sess.mapping


# ---------- helpers ----------

class _NoopDetector:
    def detect(self, text: str, *, model: str | None = None) -> list[Span]:
        return []


class _SurfaceDetector:
    def __init__(self, surfaces: dict[str, str]):
        self._surfaces = surfaces

    def detect(self, text: str, *, model: str | None = None) -> list[Span]:
        out: list[Span] = []
        for surface, label in self._surfaces.items():
            i = 0
            while True:
                idx = text.find(surface, i)
                if idx == -1:
                    break
                out.append(Span(label=label, start=idx, end=idx + len(surface), text=surface))
                i = idx + len(surface)
        return out


# Sanity: quick self-check that the helpers conform to the Detector protocol.
def test_helpers_are_detectors():
    pytest.importorskip("typing_extensions")  # pure structural check; no import really needed
    n = _NoopDetector()
    s = _SurfaceDetector({"x": "private_person"})
    assert n.detect("anything") == []
    assert len(s.detect("xx")) == 2
