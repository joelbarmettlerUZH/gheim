"""Slot-bag sampling: pre-generate PII values for Apertus to weave around.

Returns a list of (category, value) tuples spanning multiple categories so each
generated example exercises diverse entity types.
"""
from __future__ import annotations

import random

from .. import synthetic  # noqa: F401  (sibling import to keep package wiring simple)
from ..synthetic import faker_ch as F

# How likely each category is to appear in a slot bag.
CATEGORY_WEIGHTS: dict[str, float] = {
    "private_person": 0.95,
    "private_address": 0.55,
    "private_phone": 0.45,
    "private_email": 0.55,
    "private_date": 0.50,
    "account_number": 0.45,
    "private_url": 0.20,
    "secret": 0.10,
}


def _gen_for_category(cat: str, language: str) -> str:
    if cat == "private_person":
        if language == "fr_ch":
            return F.name_fr()
        if language == "it_ch":
            return F.name_it()
        if language == "rm":
            return F.name_rm()
        return F.name_de()
    if cat == "private_address":
        if language == "fr_ch":
            return F.address_fr()
        if language == "it_ch":
            return F.address_it()
        return F.address_de()
    if cat == "private_phone":
        return F.phone_ch()
    if cat == "private_email":
        return F.email_ch()
    if cat == "private_date":
        if language == "fr_ch":
            return F.date_fr()
        if language == "it_ch":
            return F.date_it()
        return F.date_de()
    if cat == "account_number":
        return random.choice((F.iban_ch, F.ahv, F.vat_che, F.credit_card))()
    if cat == "private_url":
        return F.url_ch()
    if cat == "secret":
        return F.secret_token()
    raise ValueError(f"unknown category {cat!r}")


def sample_slot_bag(language: str, *, n: int | None = None) -> list[tuple[str, str]]:
    """Sample a coherent slot bag for one generation request.

    By default samples 3-7 slots, weighted by CATEGORY_WEIGHTS. ``n`` overrides.
    """
    target = n if n is not None else random.randint(3, 7)
    cats = list(CATEGORY_WEIGHTS.keys())
    weights = [CATEGORY_WEIGHTS[c] for c in cats]
    chosen: list[str] = []
    while len(chosen) < target:
        c = random.choices(cats, weights=weights, k=1)[0]
        chosen.append(c)
    bag: list[tuple[str, str]] = []
    for c in chosen:
        bag.append((c, _gen_for_category(c, language)))
    return bag
