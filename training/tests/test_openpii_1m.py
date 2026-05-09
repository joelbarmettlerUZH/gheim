"""Unit tests for the openpii-1m builder.

These exercise the deterministic logic — record-to-Span conversion,
overlap resolution, language collapsing — without hitting HuggingFace.
The full network-bound build is exercised by F3 (the actual layer build)
which the user runs explicitly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from gheim_training.data import ai4privacy_remap as remap_mod
from gheim_training.data import openpii_1m as m


@pytest.fixture(autouse=True)
def _reset_warnings() -> None:
    remap_mod.reset_warnings()


def _record(text: str, mask: list[dict]) -> dict:
    return {"source_text": text, "privacy_mask": mask}


def test_spans_from_record_v3_labels() -> None:
    """1m emits v3 labels (TELEPHONENUM, BUILDINGNUM, ZIPCODE, etc.) —
    they must all resolve via the unified REMAP."""
    text = "Call John at +41 79 555 12 34 from 2025-01-15."
    rec = _record(text, [
        {"label": "GIVENNAME", "start": 5, "end": 9, "value": "John"},
        {"label": "TELEPHONENUM", "start": 13, "end": 30, "value": "+41 79 555 12 34"},
        {"label": "DATE", "start": 36, "end": 46, "value": "2025-01-15"},
    ])
    spans = m._spans_from_record(rec)
    labels = sorted(s.label for s in spans)
    assert labels == ["private_date", "private_person", "private_phone"]


def test_spans_from_record_drops_unknown_silently_with_warning(
    capsys: pytest.CaptureFixture,
) -> None:
    """Unmapped labels get logged once, then dropped from the output."""
    rec = _record("Hello WORLD.", [
        {"label": "GIVENNAME", "start": 6, "end": 11, "value": "WORLD"},
        {"label": "FUTURISTIC_LABEL_Z", "start": 0, "end": 5, "value": "Hello"},
    ])
    spans = m._spans_from_record(rec)
    assert len(spans) == 1 and spans[0].label == "private_person"
    err = capsys.readouterr().err
    assert "FUTURISTIC_LABEL_Z" in err


def test_spans_from_record_skips_invalid_offsets() -> None:
    """Spans with end <= start, or end > len(text), or empty surface form
    are silently skipped (don't raise)."""
    text = "Hi Anna."
    rec = _record(text, [
        {"label": "GIVENNAME", "start": 3, "end": 7, "value": "Anna"},  # valid
        {"label": "GIVENNAME", "start": 5, "end": 5, "value": ""},       # zero-width
        {"label": "GIVENNAME", "start": 0, "end": 200, "value": "x"},    # past EOL
    ])
    spans = m._spans_from_record(rec)
    assert len(spans) == 1


def test_resolve_overlaps_first_wins() -> None:
    from gheim_training.data.schema import Span
    spans = [
        Span(start=0, end=10, label="private_person"),
        Span(start=5, end=15, label="private_address"),  # overlaps first
        Span(start=20, end=25, label="private_date"),
    ]
    out = m._resolve_overlaps(spans)
    labels = [s.label for s in out]
    assert labels == ["private_person", "private_date"]


def test_lang_to_gheim_collapses_regions() -> None:
    """All DE regions collapse to de_ch, etc."""
    assert m._LANG_TO_GHEIM["de"] == "de_ch"
    assert m._LANG_TO_GHEIM["fr"] == "fr_ch"
    assert m._LANG_TO_GHEIM["it"] == "it_ch"
    assert m._LANG_TO_GHEIM["en"] == "en"


def test_hash_collision_exclusion(tmp_path: Path) -> None:
    """A pre-existing layer file's texts are hash-excluded from a new build."""
    layer4 = tmp_path / "layer4.jsonl"
    layer4.write_text(
        '{"text": "Hello there.", "spans": []}\n'
        '{"text": "Another text.", "spans": []}\n',
        encoding="utf-8",
    )
    hashes = m._load_text_hashes(layer4)
    assert m._hash("Hello there.") in hashes
    assert m._hash("Another text.") in hashes
    assert m._hash("A new text.") not in hashes


def test_hash_is_stable() -> None:
    """sha256(text)[:16] is deterministic across calls."""
    a = m._hash("repeatable")
    b = m._hash("repeatable")
    assert a == b
    assert len(a) == 16
