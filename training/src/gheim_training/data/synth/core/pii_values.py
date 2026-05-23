"""Format-diverse PII value generators.

The first synthetic generators (synth_name_patterns, synth_rm_secrets) used
one format per PII type — every date was `30. Juni 2026`, every phone
was Faker's default `+41 44 555 12 34`. The model learned those specific
formats and missed the others.

This module exposes generators that pick a format variant at random for
each call, so templates that say `{phone}` produce a mix of:

    +41 44 555 12 34   (intl with spaces)
    +41 (0)44 555 12 34 (intl with national-trunk paren)
    044 555 12 34       (national with spaces)
    044/555 12 34       (national with slash separator)
    +41-44-555-12-34    (hyphen separator)
    044 555 1234        (compact pair)

Every generator takes an optional ``fmt`` arg to pin a specific format
when a template needs determinism (e.g. an IBAN field in a bank form
template).
"""
from __future__ import annotations

import random
import string

from ...synth import faker_ch as _fk


# ============================================================ DATES ===

# Names of months for spelled-out date formats per language. We use these
# explicitly (rather than relying on Faker locale's strftime) because
# Faker_CH's DE locale uses German month names but ours need Swiss
# spelling ("Jänner" vs "Januar"; here we stick to the common form).
_MONTHS_DE = ("Januar", "Februar", "März", "April", "Mai", "Juni",
              "Juli", "August", "September", "Oktober", "November", "Dezember")
_MONTHS_FR = ("janvier", "février", "mars", "avril", "mai", "juin",
              "juillet", "août", "septembre", "octobre", "novembre", "décembre")
_MONTHS_IT = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
              "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre")
_MONTHS_RM = ("schaner", "favrer", "mars", "avrigl", "matg", "zercladur",
              "fanadur", "avust", "settember", "october", "november", "december")
_MONTHS_EN = ("January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December")


def _rand_date_components(rng: random.Random) -> tuple[int, int, int]:
    """Random (day, month_1based, year) with year in 1965-2027."""
    return rng.randint(1, 28), rng.randint(1, 12), rng.randint(1965, 2027)


def gen_date(language: str, fmt: str | None = None,
             rng: random.Random | None = None) -> str:
    """Generate a date in one of several format variants.

    Formats:
      - ``dot``     : ``30.06.2026``
      - ``slash``   : ``30/06/2026``
      - ``slash2``  : ``30/6/26``  (compact)
      - ``iso``     : ``2026-06-30``
      - ``spelled`` : ``30. Juni 2026`` / ``30 juin 2026`` / ``30 giugno 2026``
                       / ``30 da zercladur 2026`` / ``30 June 2026``
      - ``short``   : language-specific short form, e.g. ``30. Juni``
      - ``ddmmyy``  : ``30.06.26``  (2-digit year)

    Default behavior: random format, weighted toward the common ones
    (dot, slash, spelled) on Swiss langs.
    """
    rng = rng or random
    d, m, y = _rand_date_components(rng)
    if fmt is None:
        fmt = rng.choices(
            ("dot", "slash", "slash2", "iso", "spelled", "short", "ddmmyy"),
            weights=[28, 16, 6, 8, 28, 8, 6],
        )[0]

    if fmt == "dot":
        return f"{d:02d}.{m:02d}.{y}"
    if fmt == "slash":
        return f"{d:02d}/{m:02d}/{y}"
    if fmt == "slash2":
        return f"{d}/{m}/{y % 100:02d}"
    if fmt == "iso":
        return f"{y}-{m:02d}-{d:02d}"
    if fmt == "ddmmyy":
        return f"{d:02d}.{m:02d}.{y % 100:02d}"
    if fmt == "short":
        months = _months_for(language)
        if language in ("fr_ch", "it_ch"):
            return f"{d} {months[m - 1]}"
        return f"{d}. {months[m - 1]}"
    # spelled
    months = _months_for(language)
    if language == "de_ch":
        return f"{d}. {months[m - 1]} {y}"
    if language == "fr_ch":
        if d == 1:
            return f"1er {months[m - 1]} {y}"
        return f"{d} {months[m - 1]} {y}"
    if language == "it_ch":
        return f"{d} {months[m - 1]} {y}"
    if language == "rm":
        return f"{d} da {months[m - 1]} {y}"
    return f"{d} {months[m - 1]} {y}"  # en


def _months_for(language: str) -> tuple[str, ...]:
    return {
        "de_ch": _MONTHS_DE, "fr_ch": _MONTHS_FR, "it_ch": _MONTHS_IT,
        "rm": _MONTHS_RM, "en": _MONTHS_EN,
    }.get(language, _MONTHS_EN)


# ============================================================ PHONES ==

def gen_phone_ch(fmt: str | None = None,
                 rng: random.Random | None = None) -> str:
    """Generate a Swiss phone in one of several format variants.

    Formats: ``intl_spaced``, ``intl_paren``, ``intl_hyphen``, ``intl_compact``,
    ``natl_spaced``, ``natl_slash``, ``natl_paired``.
    """
    rng = rng or random
    if fmt is None:
        fmt = rng.choices(
            ("intl_spaced", "intl_paren", "intl_hyphen", "intl_compact",
             "natl_spaced", "natl_slash", "natl_paired"),
            weights=[28, 8, 5, 8, 25, 13, 13],
        )[0]
    # 2-digit area code drawn from common Swiss prefixes; remaining 7 digits
    # synthesised. (real Faker_CH does the same but with one fixed format)
    area = rng.choice(("21", "22", "26", "27", "31", "32", "33", "41",
                       "43", "44", "52", "55", "56", "61", "62", "71",
                       "76", "78", "79", "81", "91"))
    rest = f"{rng.randint(2_000_000, 9_999_999):07d}"  # 7 digits
    a, b, c = rest[:3], rest[3:5], rest[5:7]

    if fmt == "intl_spaced":
        return f"+41 {area} {a} {b} {c}"
    if fmt == "intl_paren":
        return f"+41 (0){area} {a} {b} {c}"
    if fmt == "intl_hyphen":
        return f"+41-{area}-{a}-{b}-{c}"
    if fmt == "intl_compact":
        return f"+41{area}{rest}"
    if fmt == "natl_spaced":
        return f"0{area} {a} {b} {c}"
    if fmt == "natl_slash":
        return f"0{area}/{a} {b} {c}"
    if fmt == "natl_paired":
        return f"0{area} {rest[:3]} {rest[3:]}"
    return f"+41 {area} {a} {b} {c}"


# ============================================================ EMAILS ==

_EMAIL_LOCAL_TEMPLATES = (
    "{first}.{last}", "{first}_{last}", "{first}-{last}",
    "{first_initial}.{last}", "{first}{last_initial}",
    "{first}.{last}+{tag}", "{last}.{first}", "{first}",
    "{first}.{last}{year}",
)

_EMAIL_TAGS = ("billing", "support", "noreply", "info", "newsletter",
               "marketing", "kontakt", "service")

_EMAIL_DOMAINS_CH = (
    "example.ch", "firma-ag.ch", "swisscompany.ch", "kanzlei-mueller.ch",
    "praxis-bern.ch", "spital-zh.ch", "bluewin.ch", "gmx.ch", "hispeed.ch",
    "post.ch", "swisscom.ch", "ubs.com", "credit-suisse.com",
)


def gen_email_ch(first: str | None = None, last: str | None = None,
                 fmt: str | None = None,
                 rng: random.Random | None = None) -> str:
    """Generate a Swiss-looking email. If ``first``/``last`` provided,
    derive local part from them; else use Faker."""
    rng = rng or random
    first = first or _fk.first_name_de()
    last = last or _fk.last_name_de()
    if fmt is None:
        fmt = rng.choice(_EMAIL_LOCAL_TEMPLATES)
    local = fmt.format(
        first=first.lower(),
        last=last.lower(),
        first_initial=first[0].lower() if first else "x",
        last_initial=last[0].lower() if last else "x",
        tag=rng.choice(_EMAIL_TAGS),
        year=str(rng.randint(70, 99)),
    )
    # Strip non-ASCII for the email local-part since some MTAs reject it;
    # real Swiss emails do this for "ü"→"ue" etc.
    local = (local.replace("ü", "ue").replace("ö", "oe")
                  .replace("ä", "ae").replace("ß", "ss"))
    domain = rng.choice(_EMAIL_DOMAINS_CH)
    return f"{local}@{domain}"


# ============================================================ URLs ====

_URL_DOMAINS = (
    "example.ch", "firma-ag.ch", "info.bern.ch", "swissinfo.ch",
    "ricardo.ch", "srf.ch", "20min.ch", "blick.ch", "tagesanzeiger.ch",
    "nzz.ch", "letemps.ch", "rsi.ch",
)

_URL_PATHS = ("", "/", "/dashboard", "/login", "/api/v1/users",
              "/portal/account", "/help/contact", "/de/news",
              "/news/2026/01", "/blog/post-123")

_URL_QUERIES = ("", "", "", "?lang=de", "?ref=email", "?utm_source=newsletter",
                "?id=42", "?token=xyz&lang=fr")


def gen_url_ch(fmt: str | None = None,
               rng: random.Random | None = None) -> str:
    """Generate a Swiss-looking URL. Formats: ``bare`` (domain only),
    ``http``, ``https`` (with/without path + query)."""
    rng = rng or random
    if fmt is None:
        fmt = rng.choices(("bare", "https_root", "https_path", "https_query"),
                          weights=[20, 25, 35, 20])[0]
    domain = rng.choice(_URL_DOMAINS)
    if fmt == "bare":
        return domain
    if fmt == "https_root":
        return f"https://{domain}"
    if fmt == "https_path":
        return f"https://{domain}{rng.choice(_URL_PATHS)}"
    return f"https://{domain}{rng.choice(_URL_PATHS)}{rng.choice(_URL_QUERIES)}"


# ============================================================ ACCOUNTS

def gen_iban(spaced: bool | None = None,
             rng: random.Random | None = None) -> str:
    """Swiss IBAN (Faker_CH's verified checksum)."""
    rng = rng or random
    if spaced is None:
        spaced = rng.random() < 0.5
    return _fk.iban_ch(spaced=spaced)


def gen_ahv(dotted: bool | None = None,
            rng: random.Random | None = None) -> str:
    """AHV number (EAN-13 checksum-valid)."""
    rng = rng or random
    if dotted is None:
        dotted = rng.random() < 0.7
    return _fk.ahv(dotted=dotted)


def gen_vat(rng: random.Random | None = None) -> str:
    """CHE VAT number (ISO 7064 checksum-valid)."""
    return _fk.vat_che()


def gen_credit_card(spaced: bool | None = None,
                    rng: random.Random | None = None) -> str:
    """Luhn-valid credit card number."""
    rng = rng or random
    if spaced is None:
        spaced = rng.random() < 0.5
    return _fk.credit_card(spaced=spaced)


# ============================================================ SECRETS =

def gen_secret(kind: str | None = None,
               rng: random.Random | None = None) -> str:
    """Generate a secret token. Wraps faker_ch.secret_token() which
    already produces 5 format variants (OpenAI, GitHub, Slack, AWS, JWT).

    Additional formats added here: generic 32-char base64-like password,
    .env style ``KEY=value`` body (just the value)."""
    rng = rng or random
    if kind in (None, "fk"):
        return _fk.secret_token()
    if kind == "password_b64":
        # base64-like generic password, ~24 chars
        alphabet = string.ascii_letters + string.digits + "+/"
        return "".join(rng.choices(alphabet, k=24))
    if kind == "stripe_live":
        return "sk_live_" + "".join(
            rng.choices(string.ascii_letters + string.digits, k=24))
    if kind == "stripe_test":
        return "sk_test_" + "".join(
            rng.choices(string.ascii_letters + string.digits, k=24))
    if kind == "anthropic":
        return "sk-ant-api03-" + "".join(
            rng.choices(string.ascii_letters + string.digits + "-_", k=80))
    return _fk.secret_token()


# ============================================================ ADDRESSES

def gen_address(language: str, fmt: str | None = None,
                rng: random.Random | None = None) -> str:
    """Swiss address. Wraps swiss_address generators (Geonames-backed).

    Formats:
      - ``full``         : ``Bahnhofstrasse 10, 8001 Zürich``  (default in faker_ch)
      - ``with_canton``  : ``Bahnhofstrasse 10, 8001 Zürich ZH``
      - ``street_only``  : ``Bahnhofstrasse 10``
      - ``city_first``   : ``8001 Zürich, Bahnhofstrasse 10``
    """
    rng = rng or random
    # Reuse faker_ch's per-language address generator (returns "full"-style)
    if language == "de_ch":
        full = _fk.address_de()
    elif language == "fr_ch":
        full = _fk.address_fr()
    elif language == "it_ch":
        full = _fk.address_it()
    elif language == "rm":
        full = _fk.address_rm()
    else:
        full = _fk.address_de()

    if fmt is None:
        fmt = rng.choices(("full", "with_canton", "street_only", "city_first"),
                          weights=[60, 15, 15, 10])[0]
    if fmt == "full":
        return full
    if fmt == "street_only":
        # take everything before the first comma
        return full.split(",", 1)[0].strip()
    if fmt == "with_canton":
        # Append a 2-letter canton code; best-effort (we don't know the
        # canton, so just pick a plausible one from the city we generated).
        # Real-text training does this from context; for synth it's enough
        # to format-match.
        canton = rng.choice(("ZH", "BE", "VD", "GE", "TI", "GR", "SG",
                             "LU", "BS", "AG", "VS", "FR", "SO", "TG"))
        return f"{full} {canton}"
    # city_first
    parts = full.split(",", 1)
    if len(parts) == 2:
        return f"{parts[1].strip()}, {parts[0].strip()}"
    return full
