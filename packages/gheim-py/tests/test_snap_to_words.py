"""Word-boundary snapping: spans that start or end inside a word are widened to the whole word,
and same-label spans that overlap or touch are merged."""
from __future__ import annotations

from gheim.detectors.base import Span, snap_to_words
from gheim.detectors.calibrated import CalibratedDetector


def _span(text: str, part: str, label: str = "private_address", score: float = 1.0, nth: int = 0) -> Span:
    start = -1
    for _ in range(nth + 1):
        start = text.index(part, start + 1)
    return Span(label=label, start=start, end=start + len(part), text=part, score=score)


def _texts(spans: list[Span]) -> list[str]:
    return [s.text for s in spans]


def test_a_fragment_at_the_end_of_a_word_is_widened() -> None:
    text = "Treuhand, Lugano."
    assert _texts(snap_to_words(text, [_span(text, "gano")])) == ["Lugano"]


def test_a_fragment_at_the_start_of_a_word_is_widened() -> None:
    text = "Herr Tschudi kam."
    assert _texts(snap_to_words(text, [_span(text, "Tsch", "private_person")])) == ["Tschudi"]


def test_both_edges_inside_words_are_widened() -> None:
    text = "Herr Zimmermann kam."
    assert _texts(snap_to_words(text, [_span(text, "mmerm", "private_person")])) == ["Zimmermann"]


def test_edges_at_spaces_and_punctuation_are_left_alone() -> None:
    text = "Dr. Gay-Crosier, Bern."
    assert _texts(snap_to_words(text, [_span(text, "Gay", "private_person")])) == ["Gay"]
    assert _texts(snap_to_words(text, [_span(text, "Dr. Gay-Crosier", "private_person")])) == ["Dr. Gay-Crosier"]


def test_touching_fragments_of_one_word_become_one_span() -> None:
    text = "Frau Käthi Zbinden kam."
    first = _span(text, "Frau Käthi Zbind", "private_person", 0.9)
    second = _span(text, "en", "private_person", 0.5)  # the "en" of "Zbinden"
    out = snap_to_words(text, [first, second])
    assert _texts(out) == ["Frau Käthi Zbinden"]
    assert 0.5 < out[0].score < 0.9  # length-weighted, dominated by the long part


def test_spans_separated_by_a_space_are_not_merged() -> None:
    text = "Anna Keller"
    out = snap_to_words(text, [_span(text, "Anna", "private_person"), _span(text, "Keller", "private_person")])
    assert _texts(out) == ["Anna", "Keller"]


def test_different_labels_are_never_merged() -> None:
    text = "Meier-Bern"
    out = snap_to_words(text, [_span(text, "Meier", "private_person"), _span(text, "Bern", "private_address")])
    assert _texts(out) == ["Meier", "Bern"]


def test_overlapping_same_label_spans_merge() -> None:
    text = "Frau Käthi Zbinden"
    out = snap_to_words(text, [_span(text, "Frau Käthi", "private_person"), _span(text, "Käthi Zbinden", "private_person")])
    assert _texts(out) == ["Frau Käthi Zbinden"]


def test_digits_count_as_word_characters() -> None:
    text = "Tel 0311234567 bitte"
    assert _texts(snap_to_words(text, [_span(text, "1234567", "private_phone")])) == ["0311234567"]


def test_empty_input_and_untouched_spans() -> None:
    assert snap_to_words("", []) == []
    text = "Anna kam."
    span = _span(text, "Anna", "private_person")
    assert snap_to_words(text, [span]) == [span]


def test_a_whitespace_only_span_is_dropped_not_a_crash() -> None:
    det = CalibratedDetector()
    assert det._make_span("a   b", 1, 4, "private_person", 0, [0.9, 0.9, 0.9]) is None
