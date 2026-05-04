"""Unit tests for the test_v1 surfacer.

Covers the deterministic parts: prod-doc-id loading, romansh JSONL.gz
iteration, stratified sampling. The end-to-end CLI run is exercised
manually (Day 3) and produces real artefacts under data/.
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from gheim_training.data.apertus_label.stream import Chunk
from gheim_training.eval.build_test_v1 import (
    PoolRecord,
    _iter_romansh_candidates,
    _load_prod_doc_ids,
    _stratify_and_sample,
)


def test_load_prod_doc_ids_handles_blank_lines_and_missing_keys(
    tmp_path: Path,
) -> None:
    p = tmp_path / "prod_inputs.jsonl"
    p.write_text(
        json.dumps({"doc_id": "doc-A", "text": "x"}) + "\n"
        + "\n"  # blank line — must not crash
        + json.dumps({"text": "no doc id"}) + "\n"  # no doc_id — skipped
        + json.dumps({"doc_id": "doc-B"}) + "\n",
        encoding="utf-8",
    )
    assert _load_prod_doc_ids(p) == {"doc-A", "doc-B"}


def test_iter_romansh_candidates_skips_overlap_and_emits_chunks(
    tmp_path: Path,
) -> None:
    """rm_wikipedia-like input: dedupe by id, emit chunks tagged rm/romansh."""
    rm_path = tmp_path / "rm.jsonl.gz"
    # Two records: one in prod (skipped), one new (emitted, possibly
    # multi-chunk depending on text length).
    long_text = "Schefcumandant da l'armada è il president dal stadi. " * 12
    records = [
        {"id": "roh-PROD", "text": long_text},
        {"id": "roh-NEW", "text": long_text},
    ]
    with gzip.open(rm_path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    out = list(_iter_romansh_candidates(
        rm_files=(rm_path,),
        prod_doc_ids={"roh-PROD"},
    ))
    # roh-PROD must be filtered; roh-NEW yields ≥1 chunk.
    assert all(c.doc_id == "roh-NEW" for c, _, _ in out)
    assert all(c.subset == "romansh" for c, _, _ in out)
    assert all(lang == "rm" for _, lang, _ in out)


def _mk(text: str, doc_id: str = "d", subset: str = "fineweb") -> Chunk:
    return Chunk(text=text, doc_id=doc_id, subset=subset)  # type: ignore[arg-type]


def test_stratify_and_sample_respects_per_lang_targets() -> None:
    """Per-bucket sampling: each (language, presence) bucket gets up to
    its target; over-target buckets get sampled down."""
    rng = random.Random(42)
    # Pool: 100 de_ch dense, 100 fr_ch dense, 5 rm dense, 30 it_ch control,
    # nothing for it_ch dense or de_ch control.
    candidates: list[tuple[Chunk, str, str]] = []
    for i in range(100):
        candidates.append((_mk(f"de{i}"), "de_ch", "pii_dense"))
        candidates.append((_mk(f"fr{i}"), "fr_ch", "pii_dense"))
    for i in range(5):
        candidates.append((_mk(f"rm{i}"), "rm", "pii_dense"))
    for i in range(30):
        candidates.append((_mk(f"it_c{i}"), "it_ch", "control"))

    sampled = _stratify_and_sample(
        candidates, target_per_lang=20, controls_per_lang=10, rng=rng,  # type: ignore[arg-type]
    )

    # Buckets that meet target → sampled to target
    de_dense = [c for c, lang, p in sampled if lang == "de_ch" and p == "pii_dense"]
    fr_dense = [c for c, lang, p in sampled if lang == "fr_ch" and p == "pii_dense"]
    assert len(de_dense) == 20
    assert len(fr_dense) == 20
    # rm dense was 5 < 20 → all 5 kept
    rm_dense = [c for c, lang, p in sampled if lang == "rm" and p == "pii_dense"]
    assert len(rm_dense) == 5
    # it_ch control was 30 > 10 → sampled to 10
    it_control = [c for c, lang, p in sampled if lang == "it_ch" and p == "control"]
    assert len(it_control) == 10
    # Buckets with 0 candidates → 0 in output (no errors)
    de_control = [c for c, lang, p in sampled if lang == "de_ch" and p == "control"]
    assert len(de_control) == 0


def test_pool_record_round_trips_through_dataclass_asdict() -> None:
    """PoolRecord must be JSON-serialisable via asdict for the JSONL writer."""
    from dataclasses import asdict
    r = PoolRecord(
        chunk_id="T000001",
        text="hello",
        language="de_ch",
        subset="fineweb",
        doc_id="doc-x",
        presence="pii_dense",
    )
    d = asdict(r)
    assert d["chunk_id"] == "T000001"
    assert d["language"] == "de_ch"
    # Round-trip via JSON to confirm no non-serialisable fields.
    assert json.loads(json.dumps(d))["chunk_id"] == "T000001"
