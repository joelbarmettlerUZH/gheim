"""Variant-name dedup tests for ``Session.allocate``.

These would have caught the dedup gap that motivated the
NFKC+casefold normalization: ``Joel`` / ``JOEL`` / ``joel`` previously
got three different sentinels in the same session, which produced
confusing LLM responses where one human appeared as three placeholders.

The mapping itself still stores the *first-seen* casing so restored
text reads naturally; only the lookup key is normalized.
"""
from __future__ import annotations

import pytest
from gheim import Session


def test_casing_variants_collapse_to_one_sentinel():
    s = Session()
    assert s.allocate("private_person", "Joel") == "<PERSON_1>"
    assert s.allocate("private_person", "JOEL") == "<PERSON_1>"
    assert s.allocate("private_person", "joel") == "<PERSON_1>"
    # Mapping preserves the first-seen casing so restoration reads naturally.
    assert s.mapping == {"<PERSON_1>": "Joel"}


def test_whitespace_variants_collapse():
    s = Session()
    a = s.allocate("private_person", "Joel  Barmettler")        # double space
    b = s.allocate("private_person", "Joel\tBarmettler")        # tab
    c = s.allocate("private_person", "  Joel Barmettler  ")     # leading/trailing
    d = s.allocate("private_person", "Joel\nBarmettler")        # newline
    assert a == b == c == d == "<PERSON_1>"


def test_nfkc_collapses_decomposed_unicode():
    s = Session()
    # NFC composed Müller vs NFD decomposed M + combining diaeresis.
    composed = "Müller"
    decomposed = "Müller"
    assert composed != decomposed
    assert s.allocate("private_person", composed) == "<PERSON_1>"
    assert s.allocate("private_person", decomposed) == "<PERSON_1>"
    assert s.allocate("private_person", "MÜLLER") == "<PERSON_1>"


def test_nfkc_collapses_compatibility_ligatures():
    s = Session()
    # Compatibility-decomposable: "ﬁ" (FI ligature, U+FB01) vs "fi" (two letters).
    assert s.allocate("private_person", "Soﬁa") == "<PERSON_1>"
    assert s.allocate("private_person", "Sofia") == "<PERSON_1>"


def test_german_eszett_expands_under_casefold():
    """Python's casefold() expands ``ß`` to ``ss`` so address variants
    collapse. JS's toLowerCase doesn't — see the gheim-js parallel test."""
    s = Session()
    assert s.allocate("private_address", "Strasse 12, Zürich") == "<ADDRESS_1>"
    assert s.allocate("private_address", "Straße 12, Zürich") == "<ADDRESS_1>"
    assert s.allocate("private_address", "STRASSE 12, ZÜRICH") == "<ADDRESS_1>"


def test_email_case_insensitive():
    s = Session()
    assert s.allocate("private_email", "Alice@Example.com") == "<EMAIL_1>"
    assert s.allocate("private_email", "ALICE@EXAMPLE.COM") == "<EMAIL_1>"
    assert s.allocate("private_email", "alice@example.com") == "<EMAIL_1>"


def test_truly_distinct_surfaces_get_distinct_sentinels():
    s = Session()
    assert s.allocate("private_person", "Joel") == "<PERSON_1>"
    assert s.allocate("private_person", "Alice") == "<PERSON_2>"
    assert s.allocate("private_person", "Bob") == "<PERSON_3>"


def test_normalization_does_not_cross_label_boundaries():
    """Same normalized string under different labels must still get distinct sentinels."""
    s = Session()
    assert s.allocate("private_person", "joel") == "<PERSON_1>"
    assert s.allocate("private_email", "joel") == "<EMAIL_1>"
    assert s.allocate("private_person", "JOEL") == "<PERSON_1>"  # collapses with PERSON
    assert s.allocate("private_email", "JOEL") == "<EMAIL_1>"   # collapses with EMAIL


def test_round_trip_preserves_first_seen_casing():
    """If the model picks up ``JOEL`` first and ``Joel`` second, the LLM's
    response containing ``<PERSON_1>`` should restore to ``JOEL``."""
    s = Session()
    s.allocate("private_person", "JOEL")
    s.allocate("private_person", "Joel")
    restored = s.restore("Hello <PERSON_1>!")
    assert restored == "Hello JOEL!"


def test_from_json_preserves_normalized_dedup():
    """Serialising and reloading a session should not break dedup of variants."""
    s1 = Session()
    s1.allocate("private_person", "Joel")
    s2 = Session.from_json(s1.to_json())
    # Reusing a casing variant must hit the same sentinel after round-trip.
    assert s2.allocate("private_person", "JOEL") == "<PERSON_1>"
    assert s2.allocate("private_person", "joel") == "<PERSON_1>"
    # Mapping is still keyed on the original surface form.
    assert s2.mapping == {"<PERSON_1>": "Joel"}


# ---------------------------------------------------------------------------
# Known limitations: aliasing / pluralization / format normalization.
# These are intentionally xfail until we add the opt-in normalizers
# (Ask B in the upcoming work). Marking them documents what the cheap
# NFKC+casefold approach explicitly does NOT cover.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Person-name aliasing not yet implemented (would need nameparser).")
def test_first_name_vs_full_name_should_collapse():
    s = Session()
    a = s.allocate("private_person", "Joel")
    b = s.allocate("private_person", "Joel Barmettler")
    c = s.allocate("private_person", "J. Barmettler")
    assert a == b == c


@pytest.mark.xfail(reason="Phone canonicalisation not yet implemented (would need phonenumbers).")
def test_phone_format_variants_should_collapse():
    s = Session()
    a = s.allocate("private_phone", "+41 44 268 12 34")
    b = s.allocate("private_phone", "0041 44 268 12 34")
    c = s.allocate("private_phone", "044 268 12 34")
    assert a == b == c


@pytest.mark.xfail(reason="Date canonicalisation not yet implemented (would need dateparser).")
def test_date_format_variants_should_collapse():
    s = Session()
    a = s.allocate("private_date", "1990-01-02")
    b = s.allocate("private_date", "2. Januar 1990")
    c = s.allocate("private_date", "Jan 2, 1990")
    assert a == b == c


@pytest.mark.xfail(reason="Transliteration normalization not yet implemented.")
def test_german_transliteration_variants_should_collapse():
    """``Müller`` and ``Mueller`` are the same person in German bureaucracy.
    Casefold doesn't catch this; it'd require unidecode or similar."""
    s = Session()
    assert s.allocate("private_person", "Müller") == s.allocate("private_person", "Mueller")
