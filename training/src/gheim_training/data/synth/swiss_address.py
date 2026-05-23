"""Real Swiss address generator. Replaces Faker_CH's en_US street fallback.

Strategy: pick a real Swiss (canton, plz, city) triple from Geonames, then
build a street name from a per-language pool of common Swiss street types
(Bahnhofstrasse, Hauptstrasse / Avenue de la Gare, Rue du Lac / Via Nassa,
Piazza Grande / Plaz Cumin) optionally combined with a real-feeling local
qualifier. The result is always parseable as a Swiss address by a human
reader and never includes en_US street types like "Walks" or "Causeway".

Output format follows the most common Swiss convention: ``Street + number,
PLZ City``. The street can also be a Postfach/Case postale variant.
"""
from __future__ import annotations

import random
from typing import Annotated, Literal

from . import swiss_geo

Lang = Literal["de", "fr", "it", "rm"]

# --- Generic street roots ---

_DE_STREET_TYPES = (
    "strasse", "weg", "gasse", "platz", "allee", "ring", "steig", "hof",
)
# French street prefixes are encoded as (singular_prefix, plural_prefix) so we
# can pick the correct article ("Rue du Lac" vs "Rue des Alpes"). The qualifier
# pool below tags each item as singular or plural.
_FR_PREFIX_PAIRS = (
    ("Rue du", "Rue des"),
    ("Rue de la", "Rue des"),
    ("Avenue du", "Avenue des"),
    ("Avenue de la", "Avenue des"),
    ("Boulevard du", "Boulevard des"),
    ("Boulevard de la", "Boulevard des"),
    ("Chemin du", "Chemin des"),
    ("Chemin de la", "Chemin des"),
    ("Place du", "Place des"),
    ("Place de la", "Place des"),
    ("Impasse du", "Impasse des"),
    ("Impasse de la", "Impasse des"),
    ("Quai du", "Quai des"),
)
_IT_STREET_PREFIXES = (
    "Via", "Viale", "Vicolo", "Piazza", "Largo", "Salita", "Stradone",
)
_RM_STREET_PREFIXES = ("Plaz", "Via", "Stradun", "Veia", "Plazza")

# --- Place-feeling qualifiers per language ---

_DE_QUALIFIERS = (
    "Bahnhof", "Haupt", "Markt", "Kirch", "Schul", "Dorf", "Mühle", "Linden",
    "Eichen", "Blumen", "Berg", "Tal", "Brunnen", "Wiesen", "Garten",
    "Sonnen", "Schloss", "Park", "Kloster", "Schmieden", "Müller", "Stein",
    "Rosen", "Buchen", "Birken", "Tannen", "Aaren", "Limmat", "Reuss",
    "Sihl", "Töss", "Glatt",
)
# (qualifier, is_plural). Vowel-initial singulars are handled by elision in
# _fr_street ("Rue de l'Église" not "Rue de la Église").
_FR_QUALIFIERS: tuple[tuple[str, bool], ...] = (
    ("Gare", False), ("Lac", False), ("Mont", False), ("Jardin", False),
    ("Église", False), ("Marché", False), ("Vieux Bourg", False),
    ("Forêt", False), ("Vigne", False), ("Vallée", False), ("Rivière", False),
    ("Pont", False), ("Hôtel-de-Ville", False), ("Léman", False),
    ("Jura", False),
    ("Champs", True), ("Tilleuls", True), ("Acacias", True),
    ("Marronniers", True), ("Cèdres", True), ("Roses", True), ("Lilas", True),
    ("Cinq Continents", True), ("Alpes", True),
)
# Treat H as triggering elision: most relevant French nouns (hôtel, hôpital,
# homme, herbe) have h muet. Misses h-aspiré edge cases (Hollande, héros)
# but those don't appear in our qualifier pool.
_FR_VOWELS = set("AEIOUHÉÈÊÀÂÎÔÛaeiouhéèêàâîôû")
_IT_QUALIFIERS = (
    "Nassa", "Stazione", "Cattedrale", "Lago", "Ponte", "Chiesa",
    "Mercato", "Castello", "Ospedale", "Comune", "Municipio", "Vecchia",
    "Nuova", "Indipendenza", "Monte Bre", "Ceresio", "Verbano",
    "Tre Re", "San Gottardo", "Maggiore", "San Lorenzo", "Garibaldi",
    "Cattaneo", "Manzoni",
)
_RM_QUALIFIERS = (
    "Cumin", "Vegl", "Nov", "Baselgia", "Staziun", "Mar", "Munt",
    "Plaun", "Mistral", "Curtin", "Ferrera", "Surplaz",
)


def _de_street() -> str:
    qualifier = random.choice(_DE_QUALIFIERS)
    type_ = random.choice(_DE_STREET_TYPES)
    return qualifier + type_


def _fr_street() -> str:
    qualifier, is_plural = random.choice(_FR_QUALIFIERS)
    sing_prefix, plur_prefix = random.choice(_FR_PREFIX_PAIRS)
    if is_plural:
        return f"{plur_prefix} {qualifier}"
    # Singular: elide "de la"/"de" → "de l'" before vowel-initial qualifier.
    starts_with_vowel = qualifier[0] in _FR_VOWELS
    if starts_with_vowel:
        # "Rue de la X" / "Rue du X" → "Rue de l'X"
        # "Place de la X" / "Place du X" → "Place de l'X"
        # "Avenue ..." → "Avenue de l'X" (Faubourg/Quai/etc same pattern)
        head = sing_prefix.split(" ", 1)[0]  # "Rue", "Place", "Avenue", ...
        return f"{head} de l'{qualifier}"
    return f"{sing_prefix} {qualifier}"


def _it_street() -> str:
    return f"{random.choice(_IT_STREET_PREFIXES)} {random.choice(_IT_QUALIFIERS)}"


def _rm_street() -> str:
    return f"{random.choice(_RM_STREET_PREFIXES)} {random.choice(_RM_QUALIFIERS)}"


def _street_for_lang(lang: Lang) -> str:
    if lang == "de":
        return _de_street()
    if lang == "fr":
        return _fr_street()
    if lang == "it":
        return _it_street()
    if lang == "rm":
        return _rm_street()
    raise ValueError(f"unknown lang {lang!r}")


# --- House numbers and Postfach ---

def _house_number() -> str:
    # Mostly small numbers; occasional letter suffix; occasional bigger number.
    n = random.choice((
        random.randint(1, 30),
        random.randint(1, 99),
        random.randint(1, 200),
    ))
    if random.random() < 0.08:  # ~8% of Swiss addresses use a letter suffix
        return f"{n}{random.choice('abcdef')}"
    return str(n)


def _postfach(lang: Lang) -> str:
    n = random.randint(10, 5000)
    if lang == "de":
        return f"Postfach {n}"
    if lang == "fr":
        return f"Case postale {n}"
    if lang == "it":
        return f"Casella postale {n}"
    if lang == "rm":
        return f"Chascha postala {n}"
    raise ValueError(lang)


# --- Public API ---

def address(
    lang: Annotated[Lang | None, "Linguistic register; if None, weighted by canton."] = None,
    *,
    postfach_prob: Annotated[float, "Probability of a Postfach instead of a street address."] = 0.05,
) -> str:
    place = swiss_geo.sample_place(lang)
    # When the caller asked for a specific language, use that for street naming
    # even if the place's tagged language differs (e.g. RM places sit in GR
    # which we tag as DE — but a Romansh address should use Romansh street
    # words). When called with no lang preference, route by the place's lang.
    street_lang: Lang = lang if lang is not None else place.lang
    if random.random() < postfach_prob:
        line1 = _postfach(street_lang)
    else:
        line1 = f"{_street_for_lang(street_lang)} {_house_number()}"
    return f"{line1}, {place.plz} {place.city}"


def address_de() -> str:
    return address("de")


def address_fr() -> str:
    return address("fr")


def address_it() -> str:
    return address("it")


def address_rm() -> str:
    return address("rm")
