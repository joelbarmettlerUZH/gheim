from gheim.core.sentinels import (
    SENTINEL_RE,
    Sentinel,
    find_sentinels,
    is_possible_sentinel_prefix,
    label_tag,
)


def test_label_tag_known():
    assert label_tag("private_person") == "PERSON"
    assert label_tag("private_email") == "EMAIL"
    assert label_tag("account_number") == "ACCOUNT"
    assert label_tag("secret") == "SECRET"


def test_label_tag_unknown_falls_back_to_upper():
    assert label_tag("custom_thing") == "CUSTOM_THING"


def test_sentinel_render_and_parse():
    s = Sentinel("PERSON", 7)
    assert s.render() == "<PERSON_7>"
    assert Sentinel.parse("<PERSON_7>") == s


def test_sentinel_parse_rejects_non_sentinels():
    assert Sentinel.parse("PERSON_7") is None
    assert Sentinel.parse("<PERSON>") is None
    assert Sentinel.parse("<person_7>") is None
    assert Sentinel.parse("<PERSON_>") is None
    assert Sentinel.parse("<PERSON_7> extra") is None


def test_sentinel_re_matches_in_text():
    text = "Hello <PERSON_1>, please email <EMAIL_2> by <DATE_3>."
    matches = list(SENTINEL_RE.finditer(text))
    assert [m.group(0) for m in matches] == ["<PERSON_1>", "<EMAIL_2>", "<DATE_3>"]


def test_find_sentinels_yields_offsets():
    text = "<PERSON_1> ok <PERSON_2>"
    out = list(find_sentinels(text))
    assert len(out) == 2
    assert out[0][0] == 0 and out[0][1] == 10
    assert out[1][0] == 14 and out[1][1] == 24


def test_is_possible_sentinel_prefix_positive():
    for s in ["<", "<P", "<PE", "<PERSON", "<PERSON_", "<PERSON_1", "<PERSON_12"]:
        assert is_possible_sentinel_prefix(s), s


def test_is_possible_sentinel_prefix_negative():
    # Already complete:
    assert not is_possible_sentinel_prefix("<PERSON_1>")
    # Lowercase letter — disqualified:
    assert not is_possible_sentinel_prefix("<person")
    # Space:
    assert not is_possible_sentinel_prefix("<PERSON ")
    # Doesn't start with '<':
    assert not is_possible_sentinel_prefix("PERSON_1>")
    # Empty:
    assert not is_possible_sentinel_prefix("")
    # Stuff after a complete sentinel:
    assert not is_possible_sentinel_prefix("<PERSON_1>x")
