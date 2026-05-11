"""Tests for opt-in surface-form normalizers.

These cover the cases that NFKC+casefold deliberately doesn't catch
(format-level variants of phone numbers and dates) and verify the
fallback behaviour when a normalizer can't parse its input.
"""
from __future__ import annotations

import pytest
from gheim import Session
from gheim.normalizers import e164, iso_date, resolve_normalizer


# ---------------------------------------------------------------------------
# e164: phone-number canonicalization
# ---------------------------------------------------------------------------

def test_e164_collapses_swiss_phone_format_variants():
    """The xfail counterpart in test_session_normalize.py shows these
    surfaces produce three different sentinels by default; with the e164
    normalizer (and a Swiss default region), all three collapse."""
    s = Session(normalizers={"private_phone": e164(region="CH")})
    a = s.allocate("private_phone", "+41 44 268 12 34")    # fully qualified
    b = s.allocate("private_phone", "0041 44 268 12 34")   # 00 prefix
    c = s.allocate("private_phone", "044 268 12 34")       # local Swiss form
    d = s.allocate("private_phone", "(044) 268-12-34")     # punctuation variant
    assert a == b == c == d == "<PHONE_1>"


def test_e164_does_not_collapse_truly_different_numbers():
    s = Session(normalizers={"private_phone": e164(region="CH")})
    a = s.allocate("private_phone", "+41 44 268 12 34")
    b = s.allocate("private_phone", "+41 44 268 12 35")  # one digit off
    assert a != b


def test_e164_falls_back_to_default_normalizer_on_garbage():
    """An un-parseable phone surface falls back to NFKC+casefold so it
    still gets a sentinel — the user's not silently swallowed."""
    s = Session(normalizers={"private_phone": e164(region="CH")})
    a = s.allocate("private_phone", "call me later")          # not a phone
    b = s.allocate("private_phone", "Call Me Later")          # case variant
    # Both fall back to NFKC+casefold which collapses them.
    assert a == b == "<PHONE_1>"
    # And a different unparseable string gets its own sentinel.
    c = s.allocate("private_phone", "different garbage")
    assert c == "<PHONE_2>"


def test_e164_short_string_unparseable_without_region():
    """Without a default region, a local-form number can't be E.164'd; it
    falls back to the default normalizer."""
    s_no_region = Session(normalizers={"private_phone": e164()})
    a = s_no_region.allocate("private_phone", "044 268 12 34")
    b = s_no_region.allocate("private_phone", "+41 44 268 12 34")
    # No region → "044 268 12 34" can't parse → falls back to NFKC key
    # → doesn't collapse with the +41 form (which DOES parse to E.164).
    assert a != b


# ---------------------------------------------------------------------------
# iso_date: date canonicalization
# ---------------------------------------------------------------------------

def test_iso_date_collapses_format_variants():
    s = Session(normalizers={"private_date": iso_date(languages=["en", "de"])})
    a = s.allocate("private_date", "1990-01-02")
    b = s.allocate("private_date", "2. Januar 1990")
    c = s.allocate("private_date", "Jan 2, 1990")
    d = s.allocate("private_date", "January 2 1990")
    assert a == b == c == d == "<DATE_1>"


def test_iso_date_distinguishes_different_days():
    s = Session(normalizers={"private_date": iso_date()})
    a = s.allocate("private_date", "2024-01-02")
    b = s.allocate("private_date", "2024-01-03")
    assert a != b


def test_iso_date_falls_back_on_unparseable_input():
    s = Session(normalizers={"private_date": iso_date()})
    a = s.allocate("private_date", "not a date at all")
    b = s.allocate("private_date", "NOT A DATE AT ALL")
    assert a == b == "<DATE_1>"  # NFKC+casefold collapse on the fallback


# ---------------------------------------------------------------------------
# Composition with the default key
# ---------------------------------------------------------------------------

def test_normalizers_only_apply_to_their_label():
    """Configuring a phone normalizer doesn't affect dedup of other labels."""
    s = Session(normalizers={"private_phone": e164(region="CH")})
    # Phone collapse via E.164.
    s.allocate("private_phone", "+41 44 268 12 34")
    s.allocate("private_phone", "0041 44 268 12 34")
    assert len([k for k in s.mapping if k.startswith("<PHONE_")]) == 1
    # Person dedup still uses NFKC+casefold.
    s.allocate("private_person", "Joel")
    s.allocate("private_person", "JOEL")
    assert len([k for k in s.mapping if k.startswith("<PERSON_")]) == 1


def test_user_supplied_callable_normalizer():
    """Users can pass any callable, not just built-in names."""
    def upcase(s: str) -> str:
        return s.upper()
    s = Session(normalizers={"private_person": upcase})
    s.allocate("private_person", "Joel")
    s.allocate("private_person", "joel")
    s.allocate("private_person", "JOEL")
    assert len([k for k in s.mapping if k.startswith("<PERSON_")]) == 1
    assert s.mapping["<PERSON_1>"] == "Joel"  # first-seen surface preserved


def test_resolve_normalizer_string_shortcut():
    fn = resolve_normalizer("e164")
    assert callable(fn)
    # Without region, the local Swiss form doesn't parse.
    assert fn("044 268 12 34") is None
    # The fully qualified form does.
    assert fn("+41 44 268 12 34") == "+41442681234"


def test_resolve_normalizer_unknown_raises():
    with pytest.raises(ValueError, match="unknown built-in normalizer"):
        resolve_normalizer("does_not_exist")


# ---------------------------------------------------------------------------
# Persistence: from_json keeps dedup working when the same normalizers
# are supplied on reload.
# ---------------------------------------------------------------------------

def test_from_json_with_normalizers_continues_to_dedup_phone_variants():
    s1 = Session(normalizers={"private_phone": e164(region="CH")})
    s1.allocate("private_phone", "+41 44 268 12 34")
    raw = s1.to_json()
    # Re-pass the same normalizers on reload so dedup keeps working.
    s2 = Session.from_json(raw, normalizers={"private_phone": e164(region="CH")})
    # A different format of the same number reuses the existing sentinel.
    assert s2.allocate("private_phone", "044 268 12 34") == "<PHONE_1>"
    assert s2.allocate("private_phone", "0041 44 268 12 34") == "<PHONE_1>"


def test_from_json_without_normalizers_falls_back_to_default():
    """If the user forgets to pass normalizers on reload, dedup of new
    format-variants degrades to NFKC+casefold but the existing mapping
    is preserved."""
    s1 = Session(normalizers={"private_phone": e164(region="CH")})
    s1.allocate("private_phone", "+41 44 268 12 34")
    s2 = Session.from_json(s1.to_json())  # no normalizers!
    # The reloaded mapping is intact for restoration purposes.
    assert s2.mapping == {"<PHONE_1>": "+41 44 268 12 34"}
    # But dedup of a new format variant doesn't work — gets PHONE_2.
    assert s2.allocate("private_phone", "044 268 12 34") == "<PHONE_2>"


# ---------------------------------------------------------------------------
# Round-trip: dedup is invisible to .restore (the mapping is keyed on
# rendered sentinels, not normalized surfaces).
# ---------------------------------------------------------------------------

def test_restore_returns_first_seen_phone_surface():
    s = Session(normalizers={"private_phone": e164(region="CH")})
    s.allocate("private_phone", "+41 44 268 12 34")
    s.allocate("private_phone", "044 268 12 34")  # collapses to <PHONE_1>
    # The LLM emits the sentinel; we restore to the first-seen casing.
    assert s.restore("Call <PHONE_1> please.") == "Call +41 44 268 12 34 please."
