from gheim import Session, Span


def test_allocate_returns_stable_sentinel():
    s = Session()
    a = s.allocate("private_person", "Joel")
    b = s.allocate("private_person", "Joel")
    assert a == b == "<PERSON_1>"


def test_allocate_increments_per_label():
    s = Session()
    assert s.allocate("private_person", "Alice") == "<PERSON_1>"
    assert s.allocate("private_person", "Bob") == "<PERSON_2>"
    assert s.allocate("private_email", "a@b.com") == "<EMAIL_1>"
    assert s.allocate("private_person", "Carol") == "<PERSON_3>"


def test_apply_spans_basic():
    s = Session()
    text = "Hi, my name is Joel and my email is joel@example.com."
    spans = [
        Span("private_person", start=15, end=19, text="Joel"),
        Span("private_email", start=36, end=52, text="joel@example.com"),
    ]
    out = s.apply_spans(text, spans)
    assert out == "Hi, my name is <PERSON_1> and my email is <EMAIL_1>."
    assert s.mapping == {"<PERSON_1>": "Joel", "<EMAIL_1>": "joel@example.com"}


def test_apply_spans_unsorted_input():
    s = Session()
    text = "Joel and Alice"
    spans = [
        Span("private_person", start=9, end=14, text="Alice"),
        Span("private_person", start=0, end=4, text="Joel"),
    ]
    assert s.apply_spans(text, spans) == "<PERSON_1> and <PERSON_2>"


def test_apply_spans_merges_adjacent_same_label():
    s = Session()
    text = "Email me at alice@example.com please."
    # Simulate the BPE pipeline quirk: email split into two adjacent spans.
    spans = [
        Span("private_email", start=12, end=25, text="alice@example"),
        Span("private_email", start=25, end=29, text=".com"),
    ]
    out = s.apply_spans(text, spans)
    assert out == "Email me at <EMAIL_1> please."
    # One sentinel, full email surface preserved.
    assert s.mapping == {"<EMAIL_1>": "alice@example.com"}


def test_apply_spans_does_not_merge_different_labels():
    s = Session()
    text = "Alicejoel@example.com"
    spans = [
        Span("private_person", start=0, end=5, text="Alice"),
        Span("private_email", start=5, end=21, text="joel@example.com"),
    ]
    out = s.apply_spans(text, spans)
    assert out == "<PERSON_1><EMAIL_1>"
    assert s.mapping == {"<PERSON_1>": "Alice", "<EMAIL_1>": "joel@example.com"}


def test_apply_spans_skips_overlap():
    s = Session()
    text = "abcdef"
    spans = [
        Span("private_person", start=0, end=4, text="abcd"),
        Span("private_person", start=2, end=6, text="cdef"),  # overlaps — should be dropped
    ]
    out = s.apply_spans(text, spans)
    assert out == "<PERSON_1>ef"


def test_apply_spans_coalesces_repeated_surface():
    s = Session()
    text = "Joel told Joel"
    spans = [
        Span("private_person", start=0, end=4, text="Joel"),
        Span("private_person", start=10, end=14, text="Joel"),
    ]
    out = s.apply_spans(text, spans)
    assert out == "<PERSON_1> told <PERSON_1>"
    assert len(s.mapping) == 1


def test_restore_replaces_known_sentinels():
    s = Session()
    s.allocate("private_person", "Joel")
    text = "Hello <PERSON_1>, your code is <SECRET_99>."
    # SECRET_99 was never allocated → passes through.
    assert s.restore(text) == "Hello Joel, your code is <SECRET_99>."


def test_restore_handles_repeated_sentinels():
    s = Session()
    s.allocate("private_person", "Joel")
    assert s.restore("<PERSON_1> told <PERSON_1>") == "Joel told Joel"


def test_apply_spans_detailed_recovers_duplicate_in_merged_list():
    """Regression test: prior to v0.1.6, duplicate-surface recovery only
    patched the redacted string. UI consumers (bars view, highlighters)
    consume `merged` and only saw the ONE detected span, so the second
    occurrence rendered without a redaction overlay.
    """
    s = Session()
    text = (
        "Hi Team,\nmein Name ist Lukas Brunner.\n"
        "Freundliche Grüsse,\nLukas Brunner"
    )
    spans = [
        Span("private_person", start=23, end=36, text="Lukas Brunner"),
    ]
    result = s.apply_spans_detailed(text, spans)
    assert "<PERSON_1>" in result.redacted
    assert len(result.merged) == 2
    for m in result.merged:
        assert text[m.start:m.end] == "Lukas Brunner"
        assert m.label == "private_person"
    assert result.merged[0].start < result.merged[1].start


def test_apply_spans_detailed_no_double_count_when_detector_caught_all():
    """If the detector already found both occurrences, recovery should
    be a no-op on the merged list (no duplicated entries)."""
    s = Session()
    text = "Hi Lukas Brunner. Best, Lukas Brunner."
    spans = [
        Span("private_person", start=3, end=16, text="Lukas Brunner"),
        Span("private_person", start=24, end=37, text="Lukas Brunner"),
    ]
    result = s.apply_spans_detailed(text, spans)
    assert len(result.merged) == 2


def test_session_roundtrip_json():
    s = Session()
    s.allocate("private_person", "Joel")
    s.allocate("private_email", "a@b.com")
    s.allocate("private_person", "Alice")

    raw = s.to_json()
    rebuilt = Session.from_json(raw)
    assert rebuilt.mapping == s.mapping
    # Counters survive: next allocation should not collide.
    new_sentinel = rebuilt.allocate("private_person", "Bob")
    assert new_sentinel == "<PERSON_3>"
