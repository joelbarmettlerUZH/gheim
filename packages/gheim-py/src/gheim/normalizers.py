"""Optional surface-form normalizers for ``Session``.

A normalizer canonicalises one specific category of PII so that
trivially-different formats of the same identity collapse to a single
sentinel — beyond what NFKC + casefold + whitespace-collapse alone
catches.

Examples that the default NFKC+casefold key does NOT collapse and
that an opt-in normalizer here does:

  - Phone formats: ``+41 44 268 12 34`` / ``0041 44 268 1234`` →
    both ``+41442681234`` under :func:`e164`.
  - Date formats: ``1990-01-02`` / ``2. Januar 1990`` / ``Jan 2 1990``
    → all ``1990-01-02`` under :func:`iso_date`.

Wire it up by passing a ``{label: normalizer}`` map to ``Session``::

    from gheim import Session
    from gheim.normalizers import e164, iso_date

    session = Session(normalizers={
        "private_phone": e164(region="CH"),
        "private_date":  iso_date(languages=["de", "fr", "it", "en"]),
    })

A normalizer returns ``None`` for un-parseable input (e.g. a malformed
phone number); when that happens, the session falls back to the default
NFKC+casefold key so the surface still gets a sentinel — it just won't
collapse with format variants.

Both ``phonenumbers`` and ``dateparser`` are optional dependencies. They
are only imported when the corresponding factory is actually called, and
fail with an actionable ``ImportError`` if the user forgot the extra::

    uv add 'gheim[normalizers]'
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

# A normalizer takes a raw surface and returns a canonical form, or None
# if the surface is unparseable (caller falls back to the default
# NFKC+casefold key in that case).
Normalizer = Callable[[str], str | None]


def e164(
    region: Annotated[
        str | None,
        "Default region (ISO 3166-1 alpha-2, e.g. 'CH') for phone numbers "
        "without an explicit country code. None disables parsing of "
        "non-fully-qualified numbers.",
    ] = None,
) -> Normalizer:
    """Phone-number canonicaliser via Google's ``phonenumbers``.

    Returns the E.164 form (``+<country><subscriber>``) of valid phone
    numbers; returns ``None`` for invalid or unparseable input. Useful
    so that ``+41 44 268 12 34``, ``0041 44 268 1234`` and (with
    ``region='CH'``) ``044 268 12 34`` all collapse to the same sentinel.
    """
    try:
        import phonenumbers
    except ImportError as e:
        raise ImportError(
            "the e164 normalizer requires the 'phonenumbers' package. "
            "Install with: uv add 'gheim[normalizers]'"
        ) from e

    def norm(s: str) -> str | None:
        try:
            n = phonenumbers.parse(s, region)
        except phonenumbers.NumberParseException:
            return None
        if not phonenumbers.is_valid_number(n):
            return None
        return phonenumbers.format_number(n, phonenumbers.PhoneNumberFormat.E164)

    return norm


def iso_date(
    languages: Annotated[
        list[str] | None,
        "Languages to consider when parsing ambiguous dates "
        "(e.g. ['de', 'en']). None lets dateparser auto-detect.",
    ] = None,
    settings: Annotated[
        dict | None,
        "Extra dateparser settings (e.g. {'DATE_ORDER': 'DMY'} to "
        "disambiguate '01/02/2024' as 1 February).",
    ] = None,
) -> Normalizer:
    """Date canonicaliser to ISO-8601 (``YYYY-MM-DD``) via ``dateparser``.

    Returns ``None`` for input that doesn't parse as a date.
    """
    try:
        import dateparser
    except ImportError as e:
        raise ImportError(
            "the iso_date normalizer requires the 'dateparser' package. "
            "Install with: uv add 'gheim[normalizers]'"
        ) from e

    def norm(s: str) -> str | None:
        d = dateparser.parse(s, languages=languages, settings=settings)
        if d is None:
            return None
        return d.date().isoformat()

    return norm


# String-name shortcuts so callers can pass ``"e164"`` / ``"iso_date"``
# without importing the factories. Built-in factories take no required
# arguments here; callers that need parameters (e.g. ``region='CH'``)
# should pass the factory result directly.
_BUILTINS: dict[str, Callable[[], Normalizer]] = {
    "e164": e164,
    "iso_date": iso_date,
}


def resolve_normalizer(
    spec: Annotated[
        "str | Normalizer",
        "A built-in name like 'e164' / 'iso_date' or any callable "
        "matching the Normalizer signature.",
    ],
) -> Normalizer:
    """Turn a built-in name or a callable into a concrete normalizer."""
    if callable(spec):
        return spec
    if spec in _BUILTINS:
        return _BUILTINS[spec]()
    raise ValueError(
        f"unknown built-in normalizer: {spec!r}. "
        f"Available: {sorted(_BUILTINS)}, or pass a callable."
    )
