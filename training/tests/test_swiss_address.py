"""Validate the Phase 1a Swiss address generator.

Gate: 1k sampled addresses contain zero non-Swiss surface forms (no en_US
street types like "Walks", "Causeway", "Trail" that Faker_CH was emitting),
and every (PLZ, city) pair maps to a real Swiss canton.
"""
from __future__ import annotations

import random
import re

from gheim_training.data.synth import swiss_address, swiss_geo

# Surface forms that en_US Faker emits but never appear in real Swiss
# addresses. Word-bounded so they don't false-positive inside legitimate
# Swiss city names (e.g. "Walk" inside "Walkringen", a real BE municipality).
# Boulevard is intentionally NOT here — it's a real French-Swiss street type.
_EN_US_STREET_TYPES = {
    "Walks", "Causeway", "Trail", "Highway", "Pike", "Parkway", "Turnpike",
    "Throughway", "Freeway", "Crossing", "Crescent", "Junction",
    "Heights", "Springs", "Ranch", "Mews", "Cove", "Loaf", "Stravenue",
    "Skyway",
}

# PLZ is the 4-digit number that follows a comma (separating street from
# city). This avoids false-matching Postfach numbers, house numbers, etc.
_PLZ_RE = re.compile(r",\s*(\d{4})\s+")


def setup_module() -> None:
    random.seed(2026)


def test_no_en_us_street_types_in_1k_samples() -> None:
    # Word-bounded match: "Walk" inside "Walkringen" doesn't count.
    word_re = {tok: re.compile(rf"\b{re.escape(tok)}\b") for tok in _EN_US_STREET_TYPES}
    bad: list[tuple[str, str]] = []
    for _ in range(1000):
        addr = swiss_address.address()
        for tok, pat in word_re.items():
            if pat.search(addr):
                bad.append((tok, addr))
                break
    assert not bad, f"{len(bad)} addresses contain en_US street tokens; e.g. {bad[:3]}"


def test_plz_resolves_to_real_swiss_canton() -> None:
    places = swiss_geo.all_places()
    by_plz: dict[str, swiss_geo.Place] = {p.plz: p for p in places}
    matched = 0
    for _ in range(500):
        addr = swiss_address.address()
        m = _PLZ_RE.search(addr)
        assert m, f"address has no PLZ: {addr!r}"
        plz = m.group(1)
        assert plz in by_plz, f"PLZ {plz} from {addr!r} is not a real Swiss PLZ"
        matched += 1
    assert matched == 500


def test_per_language_address_dispatch() -> None:
    # Sample 100 of each; the city should come from a canton we tagged with
    # that language (allowing GR overrides for it/rm).
    expected_cantons = {
        "de": {c for c, lang in swiss_geo.CANTON_LANG.items() if lang == "de"},
        "fr": {c for c, lang in swiss_geo.CANTON_LANG.items() if lang == "fr"},
        "it": {c for c, lang in swiss_geo.CANTON_LANG.items() if lang == "it"} | {"GR"},
        "rm": {"GR"},
    }
    by_city = {p.city: p for p in swiss_geo.all_places()}
    for lang in ("de", "fr", "it", "rm"):
        for _ in range(100):
            addr = swiss_address.address(lang)
            # The city is the trailing token after the PLZ.
            m = re.search(r"\d{4}\s+(.+)$", addr)
            assert m, addr
            city = m.group(1)
            place = by_city.get(city)
            assert place is not None, f"{lang}: city {city!r} not in geonames"
            assert place.canton in expected_cantons[lang], (
                f"{lang}: address {addr!r} resolved to canton {place.canton}, "
                f"expected one of {expected_cantons[lang]}"
            )


def test_addresses_have_house_number_or_postfach() -> None:
    has_num = re.compile(r"\b\d+[a-f]?\b")
    has_postfach = re.compile(r"(Postfach|Case postale|Casella postale|Chascha postala)\s+\d+")
    for _ in range(200):
        addr = swiss_address.address()
        line1 = addr.split(",")[0].strip()
        ok = bool(has_num.search(line1)) or bool(has_postfach.search(line1))
        assert ok, f"address line1 missing number/postfach: {addr!r}"
