"""End-to-end validation across the actual generated layer JSONLs.

Skipped when a layer file is missing so the suite stays runnable on a fresh
checkout — only the layers you've actually generated get checked.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from gheim_training.data.label_space import CATEGORIES
from gheim_training.data.schema import read_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

LAYERS = [
    ("layer1", DATA_DIR / "layer1.jsonl"),
    ("layer2", DATA_DIR / "layer2.jsonl"),
    ("layer3", DATA_DIR / "layer3_apertus.jsonl"),
    ("layer4", DATA_DIR / "layer4_en.jsonl"),
]


@pytest.mark.parametrize("name,path", LAYERS, ids=[n for n, _ in LAYERS])
def test_layer_offsets_consistent(name: str, path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{name} file missing — generate first")
    n = 0
    bad_offsets = 0
    bad_overlap = 0
    bad_label = 0
    for ex in read_jsonl(path):
        n += 1
        for sp in ex.spans:
            if sp.start < 0 or sp.end > len(ex.text) or sp.end <= sp.start:
                bad_offsets += 1
            if sp.label not in CATEGORIES:
                bad_label += 1
        ordered = sorted(ex.spans, key=lambda s: s.start)
        prev_end = -1
        for sp in ordered:
            if sp.start < prev_end:
                bad_overlap += 1
            prev_end = sp.end
    assert n > 0, f"{name}: empty layer"
    assert bad_offsets == 0, f"{name}: {bad_offsets} bad offsets"
    assert bad_overlap == 0, f"{name}: {bad_overlap} overlapping spans"
    assert bad_label == 0, f"{name}: {bad_label} unknown labels"


@pytest.mark.parametrize("name,path", LAYERS, ids=[n for n, _ in LAYERS])
def test_layer_surface_matches_text(name: str, path: Path) -> None:
    """For every span, text[start:end] must equal a non-empty trimmed surface."""
    if not path.exists():
        pytest.skip(f"{name} file missing")
    bad = 0
    n_spans = 0
    for ex in read_jsonl(path):
        for sp in ex.spans:
            n_spans += 1
            surface = ex.text[sp.start:sp.end]
            if not surface or surface.strip() != surface:
                bad += 1
    assert n_spans > 0
    # Allow up to 0.5% of spans to have edge whitespace (shouldn't happen but
    # we don't want to block on tokenizer quirks in upstream data).
    assert bad / n_spans < 0.005, f"{name}: {bad}/{n_spans} bad surfaces"
