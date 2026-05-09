"""Unified AI4Privacy → gheim REMAP across all known schema versions.

AI4Privacy has shipped at least three different label taxonomies across its
``pii-masking-65k/200k/300k/400k/openpii-1m`` releases. The earlier
``ai4privacy.py`` REMAP only knew the v1 names and silently dropped any
record whose labels matched v2 or v3 — half the labels by token count
disappeared into ``O``, training the model to ignore (e.g.) phone numbers,
IPs, and ID-card numbers. We caught this in Day 15 of gheim-1.

This module is the single source of truth for that mapping. Use it from
both the (legacy) ``ai4privacy.py`` 300k loader and the (new)
``openpii_1m.py`` loader.

Schemas observed in the wild
----------------------------
- **v1** (early pii-masking-65k): ``FIRSTNAME``, ``LASTNAME``,
  ``PHONE_NUMBER``, ``BUILDINGNUMBER``, ``ZIPCODE``, ``IPV4``,
  ``IPV6``, ``DOB``, ``PASSWORD``, ``SOCIALNUM``, ``ACCOUNTNUMBER``.
- **v2** (pii-masking-300k): ``GIVENNAME``, ``LASTNAME``, ``TEL``,
  ``BUILDING``, ``POSTCODE``, ``IP``, ``BOD``, ``PASS``, ``SOCIALNUMBER``,
  ``IDCARD``, ``PASSPORT``, ``DRIVERLICENSE``, ``COUNTRY``, ``SECADDRESS``.
- **v3-1m** (pii-masking-openpii-1m): ``GIVENNAME``, ``SURNAME``,
  ``TELEPHONENUM``, ``BUILDINGNUM``, ``ZIPCODE``, ``IDCARDNUM``,
  ``PASSPORTNUM``, ``DRIVERLICENSENUM``, ``SOCIALNUM``, ``BIRTHDATE``,
  ``TAXNUM``, ``CREDITCARDNUMBER``.

Unknown labels go through ``warn_unknown()`` so silent drops surface as
log lines (configurable via ``GHEIM_REMAP_STRICT=1`` to raise instead).

Run the smoke check:
    uv run python -m gheim_training.data.ai4privacy_remap
"""
from __future__ import annotations

import os
import sys
from typing import Final

from .label_space import CATEGORIES

# Single REMAP covering all three schemas. Keys are normalised (uppercase,
# trailing digits stripped). Values are gheim's 8 canonical categories or
# ``None`` to drop the span entirely.
REMAP: Final[dict[str, str | None]] = {
    # --- Persons ---
    # v1 / v2 / v3 all share LASTNAME; given-name varies
    "FIRSTNAME": "private_person",       # v1
    "GIVENNAME": "private_person",       # v2/v3
    "LASTNAME": "private_person",        # all
    "SURNAME": "private_person",         # v3-1m
    "MIDDLENAME": "private_person",      # v1
    "USERNAME": "private_person",        # often a real name; v2/v3
    "PREFIX": None,
    "SUFFIX": None,
    "TITLE": None,                       # "Dr.", "Mr." — not PII per our scheme
    "NRP": None,                         # nationality/religious/political — not PII per scheme

    # --- Email + URL ---
    "EMAIL": "private_email",
    "URL": "private_url",
    # IPV4/IPV6 normalize to "IPV" because trailing digits are stripped
    # (treats IPV4 and IPV6 as a single bucket — fine, both → private_url)
    "IPV": "private_url",                # v1 (normalized from IPV4/IPV6)
    "IP": "private_url",                 # v2/v3
    "MAC": None,
    "USERAGENT": None,                   # browser strings, sometimes appear

    # --- Phone ---
    "PHONE_NUMBER": "private_phone",     # v1
    "TEL": "private_phone",              # v2
    "TELEPHONENUM": "private_phone",     # v3-1m
    "PHONEIMEI": None,                   # device-id, debatable; drop

    # --- Address (collapsed; the model sees char-spans, not substructure) ---
    "STREET": "private_address",
    "STREETADDRESS": "private_address",  # v1 alias
    "BUILDINGNUMBER": "private_address", # v1
    "BUILDING": "private_address",       # v2
    "BUILDINGNUM": "private_address",    # v3-1m
    "CITY": "private_address",
    "STATE": "private_address",
    "REGION": "private_address",         # v3-1m
    "COUNTY": "private_address",         # v1
    "COUNTRY": "private_address",        # v2/v3
    "ZIPCODE": "private_address",        # v1/v3
    "POSTCODE": "private_address",       # v2
    "SECONDARYADDRESS": "private_address", # v1
    "SECADDRESS": "private_address",     # v2
    "GEOCOORD": None,                    # GPS coords — drop (not in 8 categories)
    "ORDINALDIRECTION": None,
    "NEARBYGPSCOORDINATE": None,

    # --- Date ---
    "DATE": "private_date",
    "DOB": "private_date",               # v1
    "BOD": "private_date",               # v2
    "BIRTHDATE": "private_date",         # v3-1m
    "TIME": None,                        # bare time-of-day — sparse signal, drop

    # --- Account-like (collapsed into one bucket per gheim's 8 categories) ---
    "IBAN": "account_number",
    "BIC": "account_number",
    "ACCOUNTNUMBER": "account_number",   # v1
    "ACCOUNTNAME": None,                 # name on account; covered by FIRSTNAME/LASTNAME
    "CREDITCARDNUMBER": "account_number",
    "CREDITCARDISSUER": None,
    "CARDISSUER": None,                  # v3-1m
    "CREDITCARDCVV": "secret",
    "PIN": "secret",
    "SSN": "account_number",             # US analog
    "SOCIALNUM": "account_number",       # v1/v3
    "SOCIALNUMBER": "account_number",    # v2
    "TAXNUM": "account_number",          # v3-1m
    # Doc numbers — all map to account_number per gheim's collapsed taxonomy
    "IDCARD": "account_number",          # v2
    "IDCARDNUM": "account_number",       # v3-1m
    "PASSPORT": "account_number",        # v2
    "PASSPORTNUM": "account_number",     # v3-1m
    "DRIVERLICENSE": "account_number",   # v2
    "DRIVERLICENSENUM": "account_number",# v3-1m
    "VEHICLEVIN": None,
    "VEHICLEVRM": None,

    # --- Secrets ---
    "PASSWORD": "secret",                # v1
    "PASS": "secret",                    # v2

    # --- Currency / catch-all (drop) ---
    "CURRENCY": None,
    "CURRENCYCODE": None,
    "CURRENCYNAME": None,
    "CURRENCYSYMBOL": None,
    "AMOUNT": None,
    "PRICE": None,

    # --- Org / job (not in our 8 categories) ---
    "COMPANYNAME": None,
    "JOBTITLE": None,
    "JOBTYPE": None,
    "JOBAREA": None,
    "ORGANIZATION": None,                # presidio alias

    # --- Demographics (drop; out of scope for our 8 categories) ---
    "AGE": None,
    "GENDER": None,
    "SEX": None,                         # v2/v3
    "ETHNICCATEGORY": None,
    "HEIGHT": None,
    "EYECOLOR": None,
    "BLOODTYPE": None,

    # --- Misc ---
    "MASKEDNUMBER": None,
    "SCRIPT": None,
}

# Sanity: every mapped target must be one of our 8 canonical categories.
_ALLOWED = set(CATEGORIES) | {None}
_bad = {k: v for k, v in REMAP.items() if v not in _ALLOWED}
assert not _bad, f"REMAP targets not in CATEGORIES: {_bad}"


def normalize(label: str) -> str:
    """Uppercase + strip trailing digits (AI4Privacy uses LASTNAME1, LASTNAME2…)."""
    return label.upper().rstrip("0123456789")


# Track first occurrence of each unknown label per (source, label) so the
# same unknown doesn't spam logs millions of times during a build.
_warned_unknown: set[tuple[str, str]] = set()


def warn_unknown(label: str, *, source: str) -> None:
    """Record the first sighting of an unknown label per source.

    Set ``GHEIM_REMAP_STRICT=1`` to raise instead — useful in tests so a
    new dataset version with unmapped labels fails CI loudly rather than
    silently dropping data.
    """
    norm = normalize(label)
    if norm in REMAP:
        return  # known, including known-drops (None)
    key = (source, norm)
    if key in _warned_unknown:
        return
    _warned_unknown.add(key)
    msg = (f"[ai4privacy_remap] unknown label {norm!r} (source={source!r}) "
           f"— span will be dropped. Add it to REMAP if it should map to "
           f"one of {sorted(CATEGORIES)}.")
    if os.environ.get("GHEIM_REMAP_STRICT") == "1":
        raise ValueError(msg)
    print(msg, file=sys.stderr, flush=True)


def remap(label: str, *, source: str) -> str | None:
    """Look up a label, warning on unknowns. Returns None for known-drop or unknown."""
    norm = normalize(label)
    if norm in REMAP:
        return REMAP[norm]
    warn_unknown(label, source=source)
    return None


def reset_warnings() -> None:
    """For tests."""
    _warned_unknown.clear()


if __name__ == "__main__":
    # Smoke check: the labels we discovered in 300k + 1m all resolve.
    discovered_v2 = ["LASTNAME", "USERNAME", "IDCARD", "EMAIL", "GIVENNAME",
                     "SOCIALNUMBER", "PASSPORT", "DRIVERLICENSE", "BOD", "IP",
                     "CITY", "SEX", "STATE", "TEL", "BUILDING", "TITLE",
                     "STREET", "POSTCODE", "DATE", "PASS", "COUNTRY",
                     "SECADDRESS", "GEOCOORD", "CARDISSUER", "TIME"]
    discovered_v3_1m = ["GIVENNAME", "SURNAME", "TITLE", "DATE", "BIRTHDATE",
                        "AGE", "SEX", "PASSPORTNUM", "IDCARDNUM",
                        "DRIVERLICENSENUM", "SOCIALNUM", "CREDITCARDNUMBER",
                        "EMAIL", "TELEPHONENUM", "CITY", "STREET",
                        "BUILDINGNUM", "ZIPCODE", "REGION", "TAXNUM"]
    print("=== v2 schema labels ===")
    for lab in discovered_v2:
        print(f"  {lab:<20} → {remap(lab, source='smoke')!r}")
    print("\n=== v3-1m schema labels ===")
    for lab in discovered_v3_1m:
        print(f"  {lab:<20} → {remap(lab, source='smoke')!r}")
