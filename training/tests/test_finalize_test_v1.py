"""Unit tests for the test_v1 finalizer.

Locks in: refuse-overwrite guard (held-out set should be written once),
atomic write (no half-written file on crash), accept-only resolution
merging, overlap re-resolution after merge, and presence/source field
mapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _run_finalize(consensus: Path, resolved: Path, out: Path,
                  backup: Path) -> None:
    """Invoke finalize_test_v1.main() with the given paths."""
    import sys

    from gheim_training.eval import finalize_test_v1
    saved = sys.argv[:]
    sys.argv = [
        "finalize_test_v1",
        "--consensus", str(consensus),
        "--resolved", str(resolved),
        "--out", str(out),
        "--backup-dir", str(backup),
    ]
    try:
        finalize_test_v1.main()
    finally:
        sys.argv = saved


def test_finalize_refuses_to_overwrite_existing_test_v1(tmp_path: Path) -> None:
    """The held-out set should never be silently overwritten."""
    consensus = tmp_path / "consensus.jsonl"
    resolved = tmp_path / "resolved.jsonl"
    out = tmp_path / "test_v1.jsonl"
    backup = tmp_path / "_backups"
    _write_jsonl(consensus, [])
    _write_jsonl(resolved, [])
    out.write_text("preexisting\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSING TO OVERWRITE"):
        _run_finalize(consensus, resolved, out, backup)
    # File should remain untouched
    assert out.read_text() == "preexisting\n"


def test_finalize_merges_consensus_with_accepted_resolutions(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus.jsonl"
    resolved = tmp_path / "resolved.jsonl"
    out = tmp_path / "test_v1.jsonl"
    backup = tmp_path / "_backups"
    _write_jsonl(consensus, [
        {
            "chunk_id": "T1",
            "text": "Anna Müller and Beat Brunner here.",
            "language": "de_ch",
            "subset": "fineweb",
            "doc_id": "d1",
            "presence": "pii_dense",
            "spans": [{"start": 0, "end": 11, "label": "private_person"}],  # Anna Müller
        },
    ])
    _write_jsonl(resolved, [
        # Accept Beat Brunner (the labelers were 2-of-4 on this)
        {"chunk_id": "T1", "claim": {"label": "private_person", "value": "Beat Brunner"},
         "decision": "accept", "reason": "named individual"},
        # Reject some other claim
        {"chunk_id": "T1", "claim": {"label": "private_person", "value": "here"},
         "decision": "reject", "reason": "not a person"},
    ])
    _run_finalize(consensus, resolved, out, backup)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    spans = rows[0]["spans"]
    surfaces = {rows[0]["text"][sp["start"]:sp["end"]] for sp in spans}
    assert surfaces == {"Anna Müller", "Beat Brunner"}  # accept was merged


def test_finalize_drops_unlocatable_resolutions(tmp_path: Path) -> None:
    """A resolution accepting a value that isn't actually in the chunk
    must be dropped (defensive against subagent typos)."""
    consensus = tmp_path / "consensus.jsonl"
    resolved = tmp_path / "resolved.jsonl"
    out = tmp_path / "test_v1.jsonl"
    backup = tmp_path / "_backups"
    _write_jsonl(consensus, [
        {"chunk_id": "T1", "text": "Anna Müller wrote.", "language": "de_ch",
         "subset": "fineweb", "doc_id": "d", "presence": "pii_dense", "spans": []},
    ])
    _write_jsonl(resolved, [
        # "Hans Schmid" is not in the chunk text
        {"chunk_id": "T1", "claim": {"label": "private_person", "value": "Hans Schmid"},
         "decision": "accept", "reason": "named"},
    ])
    _run_finalize(consensus, resolved, out, backup)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["spans"] == []  # un-locatable accept dropped


def test_finalize_locks_output_chmod_444(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus.jsonl"
    resolved = tmp_path / "resolved.jsonl"
    out = tmp_path / "test_v1.jsonl"
    backup = tmp_path / "_backups"
    _write_jsonl(consensus, [
        {"chunk_id": "T1", "text": "x", "language": "de_ch",
         "subset": "fineweb", "doc_id": "d", "presence": "control", "spans": []},
    ])
    _write_jsonl(resolved, [])
    _run_finalize(consensus, resolved, out, backup)
    mode = out.stat().st_mode & 0o777
    assert mode == 0o444  # read-only for everyone


def test_finalize_creates_backup(tmp_path: Path) -> None:
    consensus = tmp_path / "consensus.jsonl"
    resolved = tmp_path / "resolved.jsonl"
    out = tmp_path / "test_v1.jsonl"
    backup_dir = tmp_path / "_backups"
    _write_jsonl(consensus, [
        {"chunk_id": "T1", "text": "x", "language": "de_ch",
         "subset": "fineweb", "doc_id": "d", "presence": "control", "spans": []},
    ])
    _write_jsonl(resolved, [])
    _run_finalize(consensus, resolved, out, backup_dir)
    bak = backup_dir / "test_v1.jsonl.bak"
    assert bak.exists()
    assert bak.read_text() == out.read_text()
    bak_mode = bak.stat().st_mode & 0o777
    assert bak_mode == 0o444


def test_finalize_maps_subset_to_source_field(tmp_path: Path) -> None:
    """The pool field is `subset`; the harness expects `source`. Mapping
    happens in finalize."""
    consensus = tmp_path / "consensus.jsonl"
    resolved = tmp_path / "resolved.jsonl"
    out = tmp_path / "test_v1.jsonl"
    backup = tmp_path / "_backups"
    _write_jsonl(consensus, [
        {"chunk_id": "T1", "text": "x", "language": "de_ch",
         "subset": "entscheidsuche", "doc_id": "d", "presence": "control", "spans": []},
    ])
    _write_jsonl(resolved, [])
    _run_finalize(consensus, resolved, out, backup)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["source"] == "entscheidsuche"
    assert "subset" not in row  # cleaned up after rename


def test_finalize_resolves_overlaps_after_merging(tmp_path: Path) -> None:
    """If a consensus span and a resolved-accept span overlap, the
    finalizer's overlap resolution must drop one (not double-count)."""
    consensus = tmp_path / "consensus.jsonl"
    resolved = tmp_path / "resolved.jsonl"
    out = tmp_path / "test_v1.jsonl"
    backup = tmp_path / "_backups"
    _write_jsonl(consensus, [
        {"chunk_id": "T1", "text": "Anna Müller wrote.", "language": "de_ch",
         "subset": "fineweb", "doc_id": "d", "presence": "pii_dense",
         "spans": [{"start": 0, "end": 11, "label": "private_person"}]},  # Anna Müller
    ])
    _write_jsonl(resolved, [
        # Accept "Anna" alone — overlaps the existing consensus span
        {"chunk_id": "T1", "claim": {"label": "private_person", "value": "Anna"},
         "decision": "accept", "reason": "first name alone"},
    ])
    _run_finalize(consensus, resolved, out, backup)
    row = json.loads(out.read_text().splitlines()[0])
    # Greedy resolver keeps the longer Anna Müller (start=0, end=11)
    assert len(row["spans"]) == 1
    assert (row["spans"][0]["start"], row["spans"][0]["end"]) == (0, 11)
