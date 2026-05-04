"""Unit tests for the 4-way labeler aggregator.

Locks in the agreement-threshold semantics, the overlap-resolution rule,
the verbatim relocation behavior, the ambiguity counter, and the strict
filename-pattern validation. The aggregator is the most consequential
module in the test_v1 pipeline (it decides what becomes gold), so any
regression here invalidates the held-out set.
"""
from __future__ import annotations

import json
from pathlib import Path

from gheim_training.data.schema import Span
from gheim_training.eval.aggregate_labels import (
    Claim,
    _aggregate_chunk,
    _load_per_labeler,
    _locate,
    _resolve_overlaps,
    aggregate,
)

# --- _locate: returns (offsets, n_occurrences) ---

def test_locate_returns_offsets_and_count_for_single_occurrence() -> None:
    text = "Hello Anna Müller wrote this."
    offsets, n = _locate(text, "Anna Müller")
    assert offsets == (6, 17)
    assert n == 1


def test_locate_counts_multiple_occurrences_but_returns_first() -> None:
    text = "Anna Müller wrote. Later Anna Müller said hi."
    offsets, n = _locate(text, "Anna Müller")
    assert offsets == (0, 11)  # first occurrence only
    assert n == 2  # but caller can detect ambiguity


def test_locate_trims_surrounding_whitespace() -> None:
    text = "x  Joel  y"
    offsets, n = _locate(text, "  Joel  ")
    assert offsets == (3, 7)  # whitespace trimmed
    assert n == 1


def test_locate_returns_none_for_missing_value() -> None:
    offsets, n = _locate("Hello world", "foobar")
    assert offsets is None
    assert n == 0


def test_locate_returns_none_for_empty_value() -> None:
    offsets, n = _locate("Hello world", "")
    assert offsets is None
    assert n == 0


# --- _resolve_overlaps: greedy earlier-start, longer-on-tie ---

def test_resolve_overlaps_keeps_non_overlapping() -> None:
    spans = [
        Span(start=0, end=5, label="private_person"),
        Span(start=10, end=20, label="private_address"),
    ]
    out = _resolve_overlaps(spans)
    assert len(out) == 2


def test_resolve_overlaps_drops_later_overlap() -> None:
    spans = [
        Span(start=0, end=10, label="private_person"),
        Span(start=5, end=15, label="private_person"),  # overlaps the first
    ]
    out = _resolve_overlaps(spans)
    assert len(out) == 1
    assert out[0].start == 0


def test_resolve_overlaps_longer_wins_on_equal_start() -> None:
    spans = [
        Span(start=0, end=5, label="private_person"),
        Span(start=0, end=11, label="private_person"),  # longer at same start
    ]
    out = _resolve_overlaps(spans)
    assert len(out) == 1
    assert out[0].end == 11


# --- _aggregate_chunk: threshold semantics ---

def test_aggregate_chunk_3_of_4_accepts() -> None:
    text = "Anna Müller wrote a paper."
    by_labeler = {
        "A": [Claim(label="private_person", value="Anna Müller")],
        "B": [Claim(label="private_person", value="Anna Müller")],
        "C": [Claim(label="private_person", value="Anna Müller")],
        "D": [],  # 4th labeler did NOT propose
    }
    accepted, disputed = _aggregate_chunk(text, by_labeler, threshold=3)
    assert len(accepted) == 1
    assert accepted[0].label == "private_person"
    assert text[accepted[0].start:accepted[0].end] == "Anna Müller"
    assert disputed == []


def test_aggregate_chunk_2_of_4_disputes() -> None:
    text = "Some Müller person."
    by_labeler = {
        "A": [Claim(label="private_person", value="Müller")],
        "B": [Claim(label="private_person", value="Müller")],
        "C": [],
        "D": [],
    }
    accepted, disputed = _aggregate_chunk(text, by_labeler, threshold=3)
    assert accepted == []
    assert len(disputed) == 1
    claim, labelers = disputed[0]
    assert claim.value == "Müller"
    assert labelers == ("A", "B")


def test_aggregate_chunk_1_of_4_drops() -> None:
    text = "Foo bar baz."
    by_labeler = {
        "A": [Claim(label="private_person", value="Foo")],
        "B": [],
        "C": [],
        "D": [],
    }
    accepted, disputed = _aggregate_chunk(text, by_labeler, threshold=3)
    assert accepted == []
    assert disputed == []


def test_aggregate_chunk_within_labeler_dedup() -> None:
    """Same labeler emitting the same (label, value) twice still counts once."""
    text = "Anna Müller mentioned. Anna Müller again."
    by_labeler = {
        "A": [
            Claim(label="private_person", value="Anna Müller"),
            Claim(label="private_person", value="Anna Müller"),  # dup
        ],
        "B": [Claim(label="private_person", value="Anna Müller")],
        "C": [],
        "D": [],
    }
    accepted, disputed = _aggregate_chunk(text, by_labeler, threshold=3)
    # A and B together = 2 labelers, not 3 → disputes, not accepts
    assert accepted == []
    assert len(disputed) == 1


def test_aggregate_chunk_drops_unlocatable_value() -> None:
    """A 3+/4 claim whose verbatim value isn't in the chunk gets dropped."""
    text = "Anna wrote here."
    by_labeler = {
        "A": [Claim(label="private_person", value="Beat Brunner")],
        "B": [Claim(label="private_person", value="Beat Brunner")],
        "C": [Claim(label="private_person", value="Beat Brunner")],
        "D": [],
    }
    accepted, disputed = _aggregate_chunk(text, by_labeler, threshold=3)
    assert accepted == []  # not in text → silently dropped


# --- _load_per_labeler: strict filename pattern ---

def test_load_per_labeler_skips_filename_typos(tmp_path: Path) -> None:
    """Only batch_NNN_labeler[A-D].jsonl matches; typos surface in WARN."""
    good = tmp_path / "batch_001_labelerA.jsonl"
    good.write_text(
        json.dumps({"chunk_id": "T01", "labeler": "A", "spans": []}) + "\n",
        encoding="utf-8",
    )
    typo = tmp_path / "batch_001_LabelerB.jsonl"  # capital L — should not match
    typo.write_text(
        json.dumps({"chunk_id": "T01", "labeler": "B", "spans": []}) + "\n",
        encoding="utf-8",
    )
    by_chunk, responded = _load_per_labeler(tmp_path)
    # Only the good file's labeler should appear in the responded set.
    assert responded["T01"] == {"A"}
    assert by_chunk == {}  # no claims emitted


def test_load_per_labeler_records_responses_with_empty_spans(tmp_path: Path) -> None:
    """Empty spans is still a "responded" — distinguishes clean negatives
    from missing data."""
    p = tmp_path / "batch_007_labelerC.jsonl"
    p.write_text(
        json.dumps({"chunk_id": "T07", "labeler": "C", "spans": []}) + "\n",
        encoding="utf-8",
    )
    by_chunk, responded = _load_per_labeler(tmp_path)
    assert responded["T07"] == {"C"}
    assert by_chunk == {}  # no claims, so no entry in by_chunk


def test_load_per_labeler_drops_invalid_labels(tmp_path: Path) -> None:
    """Labels not in CATEGORIES get dropped (with a warning) — guards
    against subagent typos like "private_name" or "person"."""
    p = tmp_path / "batch_001_labelerA.jsonl"
    p.write_text(
        json.dumps({
            "chunk_id": "T01",
            "labeler": "A",
            "spans": [
                {"label": "private_person", "value": "Anna"},   # ok
                {"label": "person", "value": "Bob"},             # invalid
                {"label": "name", "value": "Charlie"},           # invalid
            ],
        }) + "\n",
        encoding="utf-8",
    )
    by_chunk, _ = _load_per_labeler(tmp_path)
    claims = by_chunk["T01"]["A"]
    assert len(claims) == 1
    assert claims[0].value == "Anna"


# --- aggregate(): full pipeline through pool + responded + threshold ---

def test_aggregate_emits_one_record_per_pool_chunk_even_if_unlabeled() -> None:
    """Pool chunks with no labeler responses still appear in consensus
    (with empty spans) so downstream finalize doesn't lose chunk_ids."""
    pool = [
        {"chunk_id": "T1", "text": "x", "language": "de_ch",
         "subset": "fineweb", "doc_id": "d1", "presence": "control"},
        {"chunk_id": "T2", "text": "Anna here.", "language": "fr_ch",
         "subset": "fineweb", "doc_id": "d2", "presence": "pii_dense"},
    ]
    by_chunk: dict = {
        "T2": {
            "A": [Claim(label="private_person", value="Anna")],
            "B": [Claim(label="private_person", value="Anna")],
            "C": [Claim(label="private_person", value="Anna")],
            "D": [],
        }
    }
    responded: dict = {"T2": {"A", "B", "C", "D"}}
    consensus, disagreements, stats = aggregate(
        pool, by_chunk, responded, threshold=3,
    )
    assert len(consensus) == 2  # both pool chunks present
    t1 = next(c for c in consensus if c["chunk_id"] == "T1")
    t2 = next(c for c in consensus if c["chunk_id"] == "T2")
    assert t1["spans"] == []
    assert t1["n_labelers_responded"] == 0
    assert len(t2["spans"]) == 1
    assert stats["chunks_no_labeler_responded"] == 1


def test_aggregate_records_ambiguous_first_occurrence() -> None:
    """Spans whose verbatim value occurs >1 time are flagged in the report."""
    pool = [
        {"chunk_id": "T1", "text": "Anna wrote. Later Anna said.", "language": "de_ch",
         "subset": "fineweb", "doc_id": "d1", "presence": "pii_dense"},
    ]
    by_chunk: dict = {
        "T1": {
            "A": [Claim(label="private_person", value="Anna")],
            "B": [Claim(label="private_person", value="Anna")],
            "C": [Claim(label="private_person", value="Anna")],
            "D": [Claim(label="private_person", value="Anna")],
        }
    }
    responded: dict = {"T1": {"A", "B", "C", "D"}}
    consensus, _, stats = aggregate(pool, by_chunk, responded, threshold=3)
    assert consensus[0]["n_ambiguous_offsets"] == 1
    assert stats["ambiguous_first_occurrence_spans"] == 1
