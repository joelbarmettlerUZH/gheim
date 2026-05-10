"""Regression tests for `lang_detect.detect`.

Locks in the FP fixes discovered during the test_v1 build (Day 3):

  - Italian text containing the common word "che" must not be tagged "rm"
    (the previous RM marker set included "che" → ~94% of FP RM bucket).
  - French text containing "ils" must not be tagged "rm" (the FR pronoun
    is too common to use as an RM marker).
  - Real Romansh content must still be tagged "rm" — the override cannot
    mask the marker heuristic for genuinely Romansh-shaped text.

These cases are deliberate: if anyone re-adds "che" or "ils" to
`_RM_MARKERS`, or lowers `_LINGUA_OVERRIDE_CONF` below 0.80, this suite
fails. The strings here are real samples drawn from the test_v1 build.
"""
from __future__ import annotations

from gheim_training.data.lang_detect import detect


def test_italian_text_with_repeated_che_not_classified_as_rm() -> None:
    """The first line of test_v1 chunk T000601-as-was — Italian text where
    "che" appears 3+ times. Pre-fix this returned "rm"."""
    text = (
        "La causa principale di questa situazione è l'imposizione dell'IAP "
        "che ogni azienda agricola allevi solo gli animali che può nutrire "
        "con i mangimi che riesce a produrre. Niente più indennità per la "
        "cura delle terre coltivate."
    )
    assert detect(text) == "it_ch"


def test_french_text_with_ils_not_classified_as_rm() -> None:
    """French sentence with "ils" used as 3rd-person pronoun. Pre-fix
    (after the "che" fix) this still returned "rm"."""
    text = (
        "Quand il donna le coup d'envoi, tous les enfants se sont donné "
        "la main et ont couru ensemble pour trouver le panier, puis ils "
        "se sont assis tous ensemble pour déguster les fruits."
    )
    assert detect(text) == "fr_ch"


def test_german_text_classified_as_de_ch() -> None:
    """Sanity: standard DE text routes to de_ch."""
    text = (
        "Das Bundesgericht hat die Beschwerde abgewiesen. Die "
        "Vorinstanz hatte den Anspruch der Beschwerdeführerin auf "
        "Schadenersatz und Genugtuung verneint. Die Begründung erscheint "
        "im Lichte der bundesgerichtlichen Rechtsprechung als haltbar."
    )
    assert detect(text) == "de_ch"


def test_real_romansh_classified_as_rm_via_markers() -> None:
    """Real Rumantsch Grischun content with ≥3 distinctive markers.
    The marker count is what saves us when lingua mistakes RM for IT
    (lingua doesn't ship a Romansh model — Romansh shares many
    Italian-derived cognates so lingua often picks IT)."""
    text = (
        "In adesiun a la NATO na fiss betg cumpatibla cun la "
        "neutralitad svizra. Relaziuns tranter la Svizra ed il "
        "Principadi da Liechtenstein. Las relaziuns vegnan regladas "
        "dapi il 1923 en in contract da duana. Il chantun Grischun "
        "ha decidì da sustegnair il rumantsch sco lingua naziunala."
    )
    assert detect(text) == "rm"


def test_real_romansh_classified_as_rm_via_subset_hint() -> None:
    """Real Rumantsch Grischun content where lingua isn't confident enough
    to override (≤0.80) and marker count is low. Subset hint must save it."""
    text = (
        "Las relaziuns tranter las pievels e las nazius datteran "
        "dapi millennis. Il chantun maina ina politica da sustegn "
        "per las communitads pli pintschnas, surtut en la regiun "
        "rumantscha dal cantun grischun."
    )
    # Without subset hint: insufficient signal, falls back to default.
    # With subset hint: explicit RM context.
    assert detect(text, subset="romansh") == "rm"


def test_subset_romansh_hint_routes_to_rm_for_short_text() -> None:
    """When text is too short for lingua (< ~30 chars) and the subset
    hint says romansh, return "rm"."""
    assert detect("Bun di!", subset="romansh") == "rm"


def test_short_text_falls_back_to_subset_default() -> None:
    """No lingua signal, no marker hits, no helpful subset → de_ch."""
    assert detect("hi") == "de_ch"
    assert detect("hi", subset="entscheidsuche") == "de_ch"
