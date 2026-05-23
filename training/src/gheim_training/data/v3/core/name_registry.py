"""Name registry: bigger pools + format variants.

v2 used Faker_CH for names, which has ~1000 unique values per locale.
For training diversity we mix Faker_DE + Faker_CH (and FR/IT
equivalents) to widen the pool. We also hand-curate a list of
**common-word surnames** — German surnames that are also everyday
nouns — because the model needs explicit training to distinguish
"Herr Bach" (person) from "Am Bach steht ein Baum" (creek).

We also expose name *format variants* (last-first, initials, title+last,
etc.) since v2 only ever rendered "First Last" and missed CV-style
or formal correspondence formats.
"""
from __future__ import annotations

import random
from typing import Literal

from faker import Faker

from ...synthetic import faker_ch as _fk


# Faker instances. We use multiple locales per target language to widen
# the pool (Faker_DE_CH has ~200 first names; mixing in Faker_DE_AT and
# Faker_DE gets ~2000 unique names with proper diacritics).
_FAKERS_DE = [Faker("de_CH"), Faker("de_AT"), Faker("de_DE")]
_FAKERS_FR = [Faker("fr_CH"), Faker("fr_FR")]
_FAKERS_IT = [Faker("it_CH"), Faker("it_IT")]


def first_name(language: str, rng: random.Random | None = None) -> str:
    """Pick a first name from the language's expanded pool."""
    rng = rng or random
    if language == "de_ch":
        return rng.choice(_FAKERS_DE).first_name()
    if language == "fr_ch":
        return rng.choice(_FAKERS_FR).first_name()
    if language == "it_ch":
        return rng.choice(_FAKERS_IT).first_name()
    if language == "rm":
        return _fk.first_name_rm()
    return _fk.first_name_en()


def last_name(language: str, rng: random.Random | None = None) -> str:
    """Pick a last name from the language's expanded pool."""
    rng = rng or random
    if language == "de_ch":
        return rng.choice(_FAKERS_DE).last_name()
    if language == "fr_ch":
        return rng.choice(_FAKERS_FR).last_name()
    if language == "it_ch":
        return rng.choice(_FAKERS_IT).last_name()
    if language == "rm":
        return _fk.last_name_rm()
    return _fk.last_name_en()


def full_name(language: str, rng: random.Random | None = None) -> str:
    return f"{first_name(language, rng)} {last_name(language, rng)}"


# ============================================================ COMMON_WORD

# German surnames that are ALSO everyday common nouns. These are the
# surnames that v2.2 misclassifies as their common-noun meaning when
# not explicitly disambiguated by context. Hand-curated from Swiss
# surname-frequency data + my own knowledge of common DE polysemy.
#
# Each entry below is also a real word in DE (creek, mountain, stone,
# king, fisher, farmer, ...). The model needs explicit positive examples
# of these as people, paired with negative examples (the noun usage),
# to learn context-sensitive disambiguation.
COMMON_WORD_LASTNAMES_DE: tuple[str, ...] = (
    # nature
    "Bach", "Berg", "Stein", "See", "Wald", "Wiese", "Acker", "Garten",
    "Sommer", "Winter", "Frühling", "Herbst", "Sonne", "Mond", "Stern",
    "Blume", "Baum", "Strauch", "Tau", "Wolke",
    # animals
    "Wolf", "Fuchs", "Adler", "Hahn", "Vogel", "Fisch", "Hirsch", "Reh",
    "Bär", "Storch", "Lerche", "Spatz", "Pfau",
    # occupations (these are extremely common Swiss surnames)
    "Müller", "Schmidt", "Schmid", "Schmied", "Bauer", "Fischer", "Schneider",
    "Schreiner", "Bäcker", "Becker", "Wagner", "Schuster", "Maurer",
    "Schlosser", "Weber", "Hofer", "Gerber", "Färber", "Jäger", "Förster",
    "Imker", "Koch", "Zimmermann",
    # titles / status
    "König", "Kaiser", "Fürst", "Graf", "Herzog", "Ritter", "Knecht",
    "Meister", "Junker",
    # household / abstract
    "Frank", "Reich", "Gross", "Klein", "Lang", "Kurz", "Jung", "Alt",
    "Neumann", "Hofmann", "Hoffmann", "Friedmann", "Lehmann",
    # body / nature continued
    "Kopf", "Hand", "Fuss",
    # plants / food
    "Apfel", "Birne", "Rebe", "Weizen", "Korn", "Roggen",
    # weather / phenomena
    "Sturm", "Regen", "Schnee", "Blitz", "Donner", "Eis", "Frost",
)


# Common-word surnames in FR (parallel set, smaller because the
# common-word-surname pattern is less prevalent in French naming).
COMMON_WORD_LASTNAMES_FR: tuple[str, ...] = (
    "Roche", "Mont", "Champs", "Forêt", "Rivière", "Lac",
    "Bois", "Pré", "Vallée", "Pierre",
    "Loup", "Oiseau", "Lièvre",
    "Boulanger", "Fournier", "Charpentier", "Maçon", "Pêcheur",
    "Berger", "Meunier", "Tisserand", "Cordonnier",
    "Roi", "Comte", "Duc", "Évêque",
    "Grand", "Petit", "Jeune", "Vieux", "Riche", "Pauvre",
    "Tempête", "Neige", "Vent",
)


# Common-word surnames in IT.
COMMON_WORD_LASTNAMES_IT: tuple[str, ...] = (
    "Monte", "Bosco", "Fiume", "Lago", "Valle", "Roccia",
    "Lupo", "Volpe", "Aquila", "Uccello",
    "Fornaio", "Falegname", "Muratore", "Sarto", "Pescatore",
    "Re", "Conte", "Duca", "Vescovo",
    "Bianco", "Nero", "Grande", "Piccolo", "Forte",
    "Pioggia", "Sole", "Luna", "Vento",
    "Rossi", "Bianchi", "Verdi",  # super-common IT surnames that are color words
)


def common_word_lastname(language: str,
                         rng: random.Random | None = None) -> str:
    """Pick a common-word surname for the given language. These are
    the surnames v2.2 mislabels as common nouns; v3 trains the model
    on positive (= person) + negative (= common noun) pairs on the
    same surface form."""
    rng = rng or random
    if language == "de_ch":
        return rng.choice(COMMON_WORD_LASTNAMES_DE)
    if language == "fr_ch":
        return rng.choice(COMMON_WORD_LASTNAMES_FR)
    if language == "it_ch":
        return rng.choice(COMMON_WORD_LASTNAMES_IT)
    # RM/EN: fall back to regular last name (no curated common-word pool)
    return last_name(language, rng)


# ============================================================ TITLES ==

# Per-language honorific titles. ``abbrev`` and ``full`` forms exposed
# separately so templates can pick which they need.
_TITLES_M = {
    "de_ch": {"abbrev": ("Hr.", "Dr.", "Prof.", "Ing.", "Dipl.-Ing.",
                          "lic. iur.", "lic. phil.", "M.A."),
              "full":   ("Herr", "Doktor", "Professor", "Dr.")},
    "fr_ch": {"abbrev": ("M.", "Dr", "Prof.", "Me", "Mtre"),
              "full":   ("Monsieur", "Docteur", "Professeur", "Maître")},
    "it_ch": {"abbrev": ("Sig.", "Dott.", "Prof.", "Ing.", "Avv."),
              "full":   ("Signor", "Dottor", "Professor", "Avvocato")},
    "rm":    {"abbrev": ("Sgnr.", "Dr.", "Prof."),
              "full":   ("Signur", "Dottur", "Professur")},
    "en":    {"abbrev": ("Mr.", "Dr.", "Prof.", "Hon."),
              "full":   ("Mr", "Doctor", "Professor", "The Honorable")},
}
_TITLES_F = {
    "de_ch": {"abbrev": ("Fr.", "Dr.", "Prof.", "Dr. med.", "Dipl.-Ing."),
              "full":   ("Frau", "Doktorin", "Professorin")},
    "fr_ch": {"abbrev": ("Mme", "Dr", "Prof.", "Me"),
              "full":   ("Madame", "Docteure", "Professeure", "Maître")},
    "it_ch": {"abbrev": ("Sig.ra", "Dott.ssa", "Prof.ssa", "Avv."),
              "full":   ("Signora", "Dottoressa", "Professoressa")},
    "rm":    {"abbrev": ("Sgnra.", "Dr.", "Prof."),
              "full":   ("Signura", "Dottura", "Professura")},
    "en":    {"abbrev": ("Ms.", "Mrs.", "Dr.", "Prof."),
              "full":   ("Ms", "Mrs", "Doctor", "Professor")},
}


def title(language: str, gender: Literal["m", "f"], style: str = "auto",
          rng: random.Random | None = None) -> str:
    """Pick an honorific title. ``style`` ∈ ``abbrev``, ``full``, ``auto`` (mix)."""
    rng = rng or random
    pool = (_TITLES_M if gender == "m" else _TITLES_F).get(
        language, _TITLES_M["de_ch"])
    if style == "auto":
        style = rng.choice(("abbrev", "full"))
    return rng.choice(pool[style])


# ============================================================ FORMATS =

# Name surface-form styles. Templates pick one (or randomise) so we
# train the model on more than "First Last".
NameStyle = Literal[
    "first_last",       # "Anna Müller"
    "last_first",       # "Müller, Anna"
    "last_first_upper", # "MÜLLER, Anna"
    "first_initial",    # "A. Müller"
    "title_last",       # "Hr. Müller" / "Herr Müller"
    "title_first_last", # "Dr. Anna Müller"
    "title_initial_last",  # "Hr. A. Müller"
    "last_only",        # "Müller"
    "first_only",       # "Anna"
]


def format_name(first: str, last: str, style: NameStyle,
                language: str = "de_ch", gender: Literal["m", "f"] = "m",
                title_style: str = "abbrev",
                rng: random.Random | None = None) -> str:
    """Render a name in the requested surface format. Returns the
    string that should appear in the chunk text; the caller is
    responsible for emitting a single span over the whole returned
    string (including title, if any)."""
    rng = rng or random
    if style == "first_last":
        return f"{first} {last}"
    if style == "last_first":
        return f"{last}, {first}"
    if style == "last_first_upper":
        return f"{last.upper()}, {first}"
    if style == "first_initial":
        return f"{first[0]}. {last}"
    if style == "title_last":
        t = title(language, gender, title_style, rng)
        return f"{t} {last}"
    if style == "title_first_last":
        t = title(language, gender, title_style, rng)
        return f"{t} {first} {last}"
    if style == "title_initial_last":
        t = title(language, gender, "abbrev", rng)
        return f"{t} {first[0]}. {last}"
    if style == "last_only":
        return last
    if style == "first_only":
        return first
    return f"{first} {last}"
