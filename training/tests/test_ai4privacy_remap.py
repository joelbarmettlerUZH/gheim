"""Tests for the unified AI4Privacy REMAP.

These tests pin down the v1, v2, and v3-1m label coverage so we can't
silently regress to the gheim-1 bug where v2 labels were dropped because
the REMAP only knew v1.
"""
from __future__ import annotations

import pytest
from gheim_training.data import ai4privacy_remap as r
from gheim_training.data.label_space import CATEGORIES


@pytest.fixture(autouse=True)
def _reset() -> None:
    r.reset_warnings()


def test_remap_targets_are_canonical_categories() -> None:
    """Every non-None value in REMAP must be one of the 8 categories."""
    allowed = set(CATEGORIES)
    for label, target in r.REMAP.items():
        if target is not None:
            assert target in allowed, f"REMAP[{label!r}] = {target!r} is not a category"


@pytest.mark.parametrize("label,expected", [
    # v1 schema (pii-masking-65k)
    ("FIRSTNAME", "private_person"),
    ("LASTNAME", "private_person"),
    ("PHONE_NUMBER", "private_phone"),
    ("BUILDINGNUMBER", "private_address"),
    ("ZIPCODE", "private_address"),
    # IPV4 / IPV6 normalize to "IPV" (trailing digits stripped) and
    # both should bucket to private_url
    ("IPV4", "private_url"),
    ("IPV6", "private_url"),
    ("DOB", "private_date"),
    ("PASSWORD", "secret"),
    ("SOCIALNUM", "account_number"),
    ("ACCOUNTNUMBER", "account_number"),
])
def test_v1_labels_resolve(label: str, expected: str) -> None:
    assert r.remap(label, source="v1") == expected


@pytest.mark.parametrize("label,expected", [
    # v2 schema (pii-masking-300k)
    ("GIVENNAME", "private_person"),
    ("LASTNAME", "private_person"),
    ("USERNAME", "private_person"),
    ("EMAIL", "private_email"),
    ("TEL", "private_phone"),
    ("IP", "private_url"),
    ("BUILDING", "private_address"),
    ("POSTCODE", "private_address"),
    ("SECADDRESS", "private_address"),
    ("COUNTRY", "private_address"),
    ("BOD", "private_date"),
    ("DATE", "private_date"),
    ("SOCIALNUMBER", "account_number"),
    ("IDCARD", "account_number"),
    ("PASSPORT", "account_number"),
    ("DRIVERLICENSE", "account_number"),
    ("PASS", "secret"),
])
def test_v2_labels_resolve(label: str, expected: str) -> None:
    assert r.remap(label, source="v2") == expected


@pytest.mark.parametrize("label,expected", [
    # v3-1m schema (pii-masking-openpii-1m)
    ("GIVENNAME", "private_person"),
    ("SURNAME", "private_person"),
    ("TELEPHONENUM", "private_phone"),
    ("BUILDINGNUM", "private_address"),
    ("ZIPCODE", "private_address"),
    ("REGION", "private_address"),
    ("BIRTHDATE", "private_date"),
    ("DATE", "private_date"),
    ("PASSPORTNUM", "account_number"),
    ("IDCARDNUM", "account_number"),
    ("DRIVERLICENSENUM", "account_number"),
    ("SOCIALNUM", "account_number"),
    ("CREDITCARDNUMBER", "account_number"),
    ("TAXNUM", "account_number"),
])
def test_v3_1m_labels_resolve(label: str, expected: str) -> None:
    assert r.remap(label, source="v3_1m") == expected


@pytest.mark.parametrize("label", [
    # Things we deliberately drop, in any schema
    "TITLE", "PREFIX", "SUFFIX",  # honorifics
    "SEX", "AGE", "GENDER",        # demographics
    "TIME",                         # bare time-of-day
    "JOBTITLE", "COMPANYNAME",     # org/employment
    "GEOCOORD", "NEARBYGPSCOORDINATE",  # GPS
    "CARDISSUER",                   # bank brand
    "CURRENCY", "AMOUNT",           # money
    "MAC", "PHONEIMEI",             # device IDs
])
def test_known_drops_resolve_to_none(label: str) -> None:
    """Known-but-dropped labels return None silently — no warning emitted."""
    assert r.remap(label, source="drop_test") is None


def test_unknown_label_warns_once_then_silences(capsys: pytest.CaptureFixture) -> None:
    """Unknown label triggers a stderr warning the first time, then stays
    silent for repeated calls with the same (source, label) pair."""
    label = "TOTALLY_FAKE_LABEL_XYZ"
    assert r.remap(label, source="warn_test") is None
    out1 = capsys.readouterr()
    assert "TOTALLY_FAKE_LABEL_XYZ" in out1.err
    # Second call: same source, same label → no new warning
    assert r.remap(label, source="warn_test") is None
    out2 = capsys.readouterr()
    assert out2.err == ""


def test_unknown_label_strict_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """GHEIM_REMAP_STRICT=1 makes unknown labels a hard error.

    This is what we'll set in the build script so a new dataset version
    with unmapped labels fails the build immediately rather than silently
    dropping data — exactly the failure mode that hit gheim-1.
    """
    monkeypatch.setenv("GHEIM_REMAP_STRICT", "1")
    with pytest.raises(ValueError, match="ANOTHER_FAKE_LABEL"):
        r.remap("ANOTHER_FAKE_LABEL", source="strict_test")


def test_normalize_strips_trailing_digits() -> None:
    """LASTNAME1 / LASTNAME2 / LASTNAME → LASTNAME."""
    assert r.normalize("LASTNAME1") == "LASTNAME"
    assert r.normalize("LASTNAME99") == "LASTNAME"
    assert r.normalize("lastname") == "LASTNAME"


def test_all_categories_are_reachable_from_remap() -> None:
    """Every gheim category should have at least one source label that
    maps to it — otherwise we can't train that category from this data."""
    targets = {v for v in r.REMAP.values() if v is not None}
    missing = set(CATEGORIES) - targets
    assert not missing, (
        f"Categories with no AI4Privacy source label: {missing}. "
        "Either add a mapping or document why training data must come "
        "from elsewhere (e.g. Layer 1 synthetic)."
    )
