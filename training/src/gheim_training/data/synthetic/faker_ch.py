"""Swiss-locale PII value generators with valid checksums.

Each generator returns a single surface string. Templates compose these into
realistic sentences. We deliberately produce *structurally correct* values
(IBAN-CH passes mod-97, AHV passes EAN-13 mod-10, VAT passes ISO 7064 mod-11)
because the model needs to learn that *real* identifiers look like these, not
random digits.
"""
from __future__ import annotations

import random
import string
from typing import Annotated

from faker import Faker

# One Faker per locale; reused for thread-cheap calls.
_FAKER_DE = Faker("de_CH")
_FAKER_FR = Faker("fr_CH")
_FAKER_IT = Faker("it_CH")
# Faker has no rm_CH — we improvise from a small surname pool in templates_rm.
_FAKER_EN = Faker("en_US")


def seed_all(seed: int) -> None:
    Faker.seed(seed)
    random.seed(seed)


# --------- names ---------

def name_de() -> str:
    return _FAKER_DE.name()


def name_fr() -> str:
    return _FAKER_FR.name()


def name_it() -> str:
    return _FAKER_IT.name()


_RM_FIRST = (
    "Andri", "Reto", "Gian", "Flurin", "Curdin", "Ladina", "Annina",
    "Selina", "Nina", "Ursina", "Lia",
)
_RM_LAST = (
    "Caduff", "Cadalbert", "Tschuor", "Jenni-Tomaschett", "Camenisch",
    "Solèr", "Casanova", "Capeder", "Derungs",
)


def name_rm() -> str:
    return f"{random.choice(_RM_FIRST)} {random.choice(_RM_LAST)}"


# --- first / last name extractors for partial-name synthetic templates ---
# Used by synth_name_patterns.py to construct "Hallo {first}, ..." and
# "{last} hat unterzeichnet..." patterns that the model fails on without
# explicit training data (see edge-case probe, v2 baseline).

def first_name_de() -> str:
    return _FAKER_DE.first_name()


def first_name_fr() -> str:
    return _FAKER_FR.first_name()


def first_name_it() -> str:
    return _FAKER_IT.first_name()


def first_name_rm() -> str:
    return random.choice(_RM_FIRST)


def first_name_en() -> str:
    return _FAKER_EN.first_name()


def last_name_de() -> str:
    return _FAKER_DE.last_name()


def last_name_fr() -> str:
    return _FAKER_FR.last_name()


def last_name_it() -> str:
    return _FAKER_IT.last_name()


def last_name_rm() -> str:
    return random.choice(_RM_LAST)


def last_name_en() -> str:
    return _FAKER_EN.last_name()


# --------- addresses ---------

# Faker_CH's street names fall back to en_US ("Adams Key", "Miller Walks").
# We use a dedicated Swiss generator built on real Geonames PLZ data + per-
# language street pools. See swiss_address.py and swiss_geo.py.
from . import swiss_address as _swiss_address  # noqa: E402


def address_de() -> str:
    return _swiss_address.address_de()


def address_fr() -> str:
    return _swiss_address.address_fr()


def address_it() -> str:
    return _swiss_address.address_it()


def address_rm() -> str:
    return _swiss_address.address_rm()


# --------- phones ---------

def phone_ch(format_kind: str | None = None) -> str:
    """Swiss phone in one of several common formats."""
    fmt = format_kind or random.choice(("e164_spaced", "e164_compact", "national_spaced", "national_compact"))
    # Mobile prefixes 75..79; landline area codes from a small pool.
    is_mobile = random.random() < 0.5
    if is_mobile:
        ndc = random.choice((75, 76, 77, 78, 79))
    else:
        ndc = random.choice((21, 22, 24, 26, 27, 31, 32, 33, 41, 43, 44, 52, 55, 56, 61, 62, 71, 81, 91))
    n = f"{random.randint(100, 999)} {random.randint(10, 99)} {random.randint(10, 99)}"
    n_compact = n.replace(" ", "")
    if fmt == "e164_spaced":
        return f"+41 {ndc} {n}"
    if fmt == "e164_compact":
        return f"+41{ndc}{n_compact}"
    if fmt == "national_spaced":
        return f"0{ndc} {n}"
    return f"0{ndc}{n_compact}"


# --------- IBAN-CH ---------

def _iban_ch(bank_code: int | None = None, account: str | None = None) -> str:
    """Generate a structurally valid Swiss IBAN.

    Format: CH + 2 check digits + 5-digit bank code + 12-char account.
    Check digits computed via ISO 13616 mod-97-10.
    """
    bc = f"{bank_code if bank_code is not None else random.randint(0, 99999):05d}"
    acc = account or "".join(random.choices(string.digits, k=12))
    bban = bc + acc
    # Move country code + "00" to the end, convert letters to digits, mod 97.
    rearranged = bban + "CH" + "00"
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    check = 98 - (int(numeric) % 97)
    return f"CH{check:02d}{bban}"


def iban_ch(spaced: bool | None = None) -> str:
    iban = _iban_ch()
    if spaced is None:
        spaced = random.random() < 0.5
    if spaced:
        # Standard 4-char grouping
        return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))
    return iban


# --------- AHV / AVS (Swiss social security) ---------

def ahv(dotted: bool | None = None) -> str:
    """Generate a structurally valid 13-digit AHV number (756 prefix, EAN-13 check).

    Format: 756.XXXX.XXXX.CC where CC is the EAN-13 check digit over 756 + 9 digits.
    """
    body = "756" + "".join(random.choices(string.digits, k=9))
    # EAN-13 check: weights 1,3,1,3,... over the first 12 digits, sum, take mod 10, complement.
    s = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(body))
    check = (10 - (s % 10)) % 10
    full = body + str(check)
    if dotted is None:
        dotted = random.random() < 0.85
    if dotted:
        return f"{full[0:3]}.{full[3:7]}.{full[7:11]}.{full[11:13]}"
    return full


# --------- VAT (UID) ---------

def vat_che(suffix: str | None = None) -> str:
    """Generate a structurally valid Swiss UID/VAT number (CHE-XXX.XXX.XXX [MWST]).

    Check digit: ISO 7064 mod-11-10 over the 8 leading digits.
    """
    body = "".join(random.choices(string.digits, k=8))
    weights = (5, 4, 3, 2, 7, 6, 5, 4)
    s = sum(int(d) * w for d, w in zip(body, weights, strict=True))
    check = 11 - (s % 11)
    if check == 11:
        check = 0
    elif check == 10:
        # In rare cases the algorithm yields 10; resample for simplicity.
        return vat_che(suffix=suffix)
    full = body + str(check)
    formatted = f"CHE-{full[0:3]}.{full[3:6]}.{full[6:9]}"
    if suffix is None:
        suffix = random.choice(("", " MWST", " IVA", " TVA", ""))
    return formatted + suffix


# --------- credit card (Luhn-valid) ---------

def credit_card(spaced: bool | None = None) -> str:
    """Generate a Luhn-valid 16-digit Visa/Mastercard-shaped number."""
    prefix = random.choice(("4",) + ("5" + str(random.randint(1, 5)),))
    body = prefix + "".join(random.choices(string.digits, k=15 - len(prefix)))
    # Luhn check digit
    digits = [int(d) for d in body]
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    check = (10 - (s % 10)) % 10
    full = body + str(check)
    if spaced is None:
        spaced = random.random() < 0.5
    if spaced:
        return " ".join(full[i:i + 4] for i in range(0, 16, 4))
    return full


# --------- dates ---------

def date_de() -> str:
    """German-Swiss dates: DD.MM.YYYY or 'D. <Month> YYYY' (rarer)."""
    d = _FAKER_DE.date_object()
    if random.random() < 0.85:
        return d.strftime("%d.%m.%Y")
    months = ("Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
              "August", "September", "Oktober", "November", "Dezember")
    return f"{d.day}. {months[d.month - 1]} {d.year}"


def date_fr() -> str:
    d = _FAKER_FR.date_object()
    if random.random() < 0.85:
        return d.strftime("%d/%m/%Y")
    months = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
              "août", "septembre", "octobre", "novembre", "décembre")
    return f"{d.day} {months[d.month - 1]} {d.year}"


def date_it() -> str:
    d = _FAKER_IT.date_object()
    if random.random() < 0.85:
        return d.strftime("%d.%m.%Y")
    months = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
              "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre")
    return f"{d.day} {months[d.month - 1]} {d.year}"


# --------- emails / URLs / secrets ---------

def email_ch(name_hint: Annotated[str | None, "If given, derive local-part from it."] = None) -> str:
    domains = ("uzh.ch", "ethz.ch", "epfl.ch", "swisscom.ch", "ubs.com",
               "credit-suisse.com", "post.ch", "sbb.ch", "bluewin.ch",
               "kanton-zuerich.ch", "ti.ch", "vd.ch", "ge.ch")
    if name_hint:
        local = name_hint.lower()
        for ch in (" ", "-", "."):
            local = local.replace(ch, ".")
        local = "".join(c for c in local if c.isalnum() or c == ".")
    else:
        local = _FAKER_DE.user_name()
    return f"{local}@{random.choice(domains)}"


def url_ch() -> str:
    paths = ("", "/de", "/fr/start.html", "/it/news", "/contatto",
             "/about", "/blog/2024-03-01")
    domains = ("admin.ch", "zh.ch", "vd.ch", "ti.ch", "sbb.ch", "post.ch",
               "ubs.com", "swisscom.ch", "ricardo.ch")
    return f"https://www.{random.choice(domains)}{random.choice(paths)}"


def secret_token() -> str:
    kind = random.choice(("openai", "github", "slack", "jwt", "aws"))
    if kind == "openai":
        return "sk-proj-" + "".join(random.choices(string.ascii_letters + string.digits + "_-", k=43))
    if kind == "github":
        return "ghp_" + "".join(random.choices(string.ascii_letters + string.digits, k=36))
    if kind == "slack":
        return f"xoxb-{random.randint(10**11, 10**12 - 1)}-{random.randint(10**11, 10**12 - 1)}-{''.join(random.choices(string.ascii_letters + string.digits, k=24))}"
    if kind == "aws":
        return "AKIA" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
    # JWT
    def seg(n: int) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits + "_-", k=n))
    return f"{seg(32)}.{seg(64)}.{seg(43)}"
