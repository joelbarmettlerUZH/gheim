"""Tests for the commercial-register chunk identifier."""
from __future__ import annotations

from gheim_training.data.v2.cr_identify import (
    filter_cr,
    is_commercial_register,
)


def test_de_cr_listing_detected():
    text = (
        "Das Unternehmen Rubeli Bau Holding GmbH ist eine Gesellschaft mit "
        "beschränkter Haftung mit Sitz in Gampelen, im Kanton Bern, die im "
        "Jahr 2019 gegründet wurde. Personen mit Entscheidungsbefugnis: "
        "Reto Rubeli (Gesellschafter, Präsident und Geschäftsführer) und "
        "Fabienne Rubeli-Riedo (Geschäftsführerin)."
    )
    assert is_commercial_register(text, "de_ch")


def test_fr_cr_listing_detected():
    text = (
        "François Roullet (Membre du conseil d'administration) et Karim "
        "Lassueur (Directeur), avec signature individuelle."
    )
    assert is_commercial_register(text, "fr_ch")


def test_it_cr_listing_detected():
    text = (
        "Marone, Claudio, da Bellinzona, in Muralto, presidente, con firma "
        "collettiva a due, con Cattaneo, Carlo, da Riva San Vitale, in "
        "Lugano, amministratore."
    )
    assert is_commercial_register(text, "it_ch")


def test_en_cr_listing_detected():
    text = (
        "The company Example Holding Ltd is a limited liability company "
        "with its registered office in Zurich. Board of Directors: Anna "
        "Mueller (chair), John Schmid (managing director)."
    )
    assert is_commercial_register(text, "en")


def test_passing_mention_still_flagged_per_design():
    """A prose chunk that mentions 'Verwaltungsrat' in passing still
    matches — the identifier is intentionally generous so the recall
    fix doesn't miss the actual problem cases. False-positive prose
    chunks just get labelled by the CR-specialised prompt, which
    falls back to the standard rules when no contact-block is present."""
    text = (
        "Der Verwaltungsrat hat heute eine Stellungnahme zur neuen "
        "Regulierung veröffentlicht."
    )
    assert is_commercial_register(text, "de_ch")


def test_pure_prose_not_flagged():
    text = "Es war ein schöner Tag in Zürich. Anna ging spazieren."
    assert not is_commercial_register(text, "de_ch")


def test_empty_text():
    assert not is_commercial_register("", "de_ch")


def test_unknown_language_falls_back_to_all_patterns():
    """A chunk with an unrecognised language label still gets checked
    against every language's marker set."""
    text = "Personen mit Entscheidungsbefugnis: Anna Müller."
    assert is_commercial_register(text, "unknown_lang")


def test_filter_cr_yields_only_cr():
    chunks = [
        {"text": "Personen mit Entscheidungsbefugnis: Anna.", "language": "de_ch"},
        {"text": "Es war ein schöner Tag.", "language": "de_ch"},
        {"text": "Conseil d'administration: Marie.", "language": "fr_ch"},
        {"text": "Hello world.", "language": "en"},
    ]
    out = list(filter_cr(chunks))
    assert len(out) == 2
    assert out[0]["language"] == "de_ch"
    assert out[1]["language"] == "fr_ch"
