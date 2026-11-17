import json

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


# -- sentinels already present in the input ----------------------------------
# Text a caller anonymised earlier carries sentinels this session never issued.
# They must pass through untouched, and their indices must not be reused, or
# restore() would replace the caller's placeholder with a new entity's value.


def test_preexisting_sentinel_index_is_not_reused():
    s = Session()
    text = "<PERSON_1> forwarded this from Anna"
    out = s.apply_spans(text, [Span("private_person", 31, 35, "Anna")])
    assert out == "<PERSON_1> forwarded this from <PERSON_2>"
    assert s.mapping == {"<PERSON_2>": "Anna"}
    assert s.restore(out) == text


def test_span_overlapping_preexisting_sentinel_is_dropped():
    s = Session()
    text = "ask <PERSON_1> again"
    # A detector may read the sentinel's own letters as a name.
    out = s.apply_spans(text, [Span("private_person", 5, 13, "PERSON_1")])
    assert out == text
    assert s.mapping == {}


def test_reserve_covers_text_anonymised_later():
    s = Session()
    s.reserve("second message quotes <PERSON_1> and <EMAIL_3>")
    assert s.allocate("private_person", "Anna") == "<PERSON_2>"
    assert s.allocate("private_email", "a@b.ch") == "<EMAIL_4>"


def test_reserve_ignores_sentinels_this_session_issued():
    s = Session()
    assert s.allocate("private_person", "Anna") == "<PERSON_1>"
    s.reserve("<PERSON_1>")
    assert s.allocate("private_person", "Bob") == "<PERSON_2>"


def test_span_running_into_preexisting_sentinel_is_trimmed_not_dropped():
    # A detector that bleeds across "<...>" must not cause the real PII next
    # to the sentinel to go out in clear text.
    s = Session()
    text = "Dr. <PERSON_1> Meier called"
    out = s.apply_spans(text, [Span("private_person", 4, 20, "<PERSON_1> Meier")])
    assert out == "Dr. <PERSON_1> <PERSON_2> called"
    assert s.mapping == {"<PERSON_2>": "Meier"}


def test_span_around_preexisting_sentinel_keeps_both_sides():
    s = Session()
    text = "Anna <PERSON_1> Meier"
    out = s.apply_spans(text, [Span("private_person", 0, len(text), text)])
    assert out == "<PERSON_2> <PERSON_1> <PERSON_3>"
    assert s.mapping == {"<PERSON_2>": "Anna", "<PERSON_3>": "Meier"}


def test_unrelated_markup_token_does_not_unredact_adjacent_pii():
    s = Session()
    text = "<TICKET_4711>Anna Meier"
    out = s.apply_spans(text, [Span("private_person", 0, len(text), text)])
    assert out == "<TICKET_4711><PERSON_1>"
    assert s.mapping == {"<PERSON_1>": "Anna Meier"}


def test_reserve_walks_nested_structures():
    s = Session()
    s.reserve({"messages": [{"name": "<PERSON_4>", "content": ("<EMAIL_2>",)}], "n": 3})
    assert s.allocate("private_person", "Anna") == "<PERSON_5>"
    assert s.allocate("private_email", "a@b.ch") == "<EMAIL_3>"



# -- reserved indexes across to_json / from_json -------------------------------
# A reserved index above everything the session issued is not implied by the
# mapping, so it has to be written out, or a restored session hands it out again.


def test_reserved_index_survives_json_round_trip():
    s = Session()
    s.reserve("forwarded by <EMAIL_2>")
    rebuilt = Session.from_json(s.to_json())
    assert rebuilt.allocate("private_email", "a@b.ch") == "<EMAIL_3>"
    assert rebuilt.allocate("private_email", "c@d.ch") == "<EMAIL_4>"


def test_reservation_of_index_one_survives_json_round_trip():
    s = Session()
    s.reserve("<EMAIL_1>")
    rebuilt = Session.from_json(s.to_json())
    assert rebuilt.allocate("private_email", "a@b.ch") == "<EMAIL_2>"


def test_reservation_above_an_index_zero_entry_survives_json_round_trip():
    # Only a hand-written or foreign payload carries index 0; it implies nothing.
    s = Session.from_json(json.dumps({"mapping": {"<PERSON_0>": "Anna"}}))
    s.reserve("<PERSON_1>")
    rebuilt = Session.from_json(s.to_json())
    assert rebuilt.allocate("private_person", "Bob") == "<PERSON_2>"


def test_to_json_writes_only_reservations_the_mapping_does_not_imply():
    s = Session()
    s.reserve("<PERSON_2> and <EMAIL_5>")
    s.allocate("private_person", "Anna")  # <PERSON_3> already implies 2
    assert json.loads(s.to_json()) == {
        "mapping": {"<PERSON_3>": "Anna"},
        "reserved": {"EMAIL": 5},
    }


def test_from_json_ignores_malformed_reservations():
    raw = json.dumps({
        "mapping": {"<PERSON_1>": "Anna"},
        "reserved": {"PERSON": 4, "EMAIL": "7", "PHONE": True, "DATE": -3, "URL": 2.5},
    })
    s = Session.from_json(raw)
    assert s.allocate("private_person", "Bob") == "<PERSON_5>"
    assert s.allocate("private_email", "a@b.ch") == "<EMAIL_1>"
    assert s.allocate("private_phone", "044 000 00 00") == "<PHONE_1>"
    assert s.allocate("private_date", "1. Mai") == "<DATE_1>"
    assert s.allocate("private_url", "x.ch") == "<URL_1>"


def test_from_json_ignores_a_reserved_field_that_is_not_an_object():
    s = Session.from_json(json.dumps({"mapping": {}, "reserved": [["PERSON", 3]]}))
    assert s.allocate("private_person", "Anna") == "<PERSON_1>"
