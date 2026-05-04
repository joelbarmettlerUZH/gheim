"""Day 4 batch splitter for the test_v1 labeling pipeline.

Reads ``data/test_v1_pool.jsonl`` (800 chunks) and splits it into ~40
batches of ~20 chunks each. Each batch is a self-contained JSONL that a
single subagent labeler can process in one pass.

Each batch is sized to fit comfortably in a subagent's context window
and to take ~5-10 minutes of subagent labeling effort. Each chunk in a
batch will be labeled by 4 independent subagent invocations
(labelers A, B, C, D) for 4-way ensemble agreement.

Output layout::

    data/test_v1_batches/batch_001.jsonl   (chunks T000001..T000020)
    data/test_v1_batches/batch_002.jsonl   (chunks T000021..T000040)
    ...
    data/test_v1_batches/batch_040.jsonl   (chunks T000781..T000800)

Each batch line is a chunk record from ``test_v1_pool.jsonl`` (chunk_id,
text, language, subset, doc_id, presence). Subagents read the batch +
``prompt.md`` and emit ``data/test_v1_labels/batch_NNN_labeler{A,B,C,D}.jsonl``.

Run:
    uv run python -m gheim_training.eval.split_batches \\
        --pool data/test_v1_pool.jsonl \\
        --out-dir data/test_v1_batches \\
        --batch-size 20
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Annotated


def _read_pool(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _split(records: list[dict], batch_size: int) -> list[list[dict]]:
    """Stable round-robin: keep within-batch language mix proportional to the
    pool. We sort by chunk_id (which is already T-prefixed counter from the
    surfacer), then chunk into fixed-size batches. The surfacer already
    emitted records in stratified order so a sequential split preserves the
    cross-language mix per batch."""
    return [records[i:i + batch_size] for i in range(0, len(records), batch_size)]


def _write_batches(
    batches: list[list[dict]],
    out_dir: Annotated[Path, "Where batch_NNN.jsonl files go."],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(batches, start=1):
        path = out_dir / f"batch_{i:03d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in batch:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")


def _print_report(batches: list[list[dict]]) -> None:
    print(f"Wrote {len(batches)} batches.")
    # Per-batch language mix
    langs_per_batch: list[Counter[str]] = [
        Counter(r["language"] for r in b) for b in batches
    ]
    # Show min/max counts per language to confirm balance
    all_langs = sorted({lang for c in langs_per_batch for lang in c})
    print(f"\nPer-batch language balance ({len(batches)} batches):")
    for lang in all_langs:
        counts = [c.get(lang, 0) for c in langs_per_batch]
        avg = sum(counts) / len(counts)
        print(f"  {lang:<6} per-batch: min={min(counts)}  avg={avg:.1f}  max={max(counts)}")
    # Per-batch presence mix
    presences_per_batch: list[Counter[str]] = [
        Counter(r["presence"] for r in b) for b in batches
    ]
    print("\nPer-batch presence balance:")
    for presence in ("pii_dense", "control"):
        counts = [c.get(presence, 0) for c in presences_per_batch]
        avg = sum(counts) / len(counts)
        print(f"  {presence:<10} min={min(counts)} avg={avg:.1f} max={max(counts)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=Path("data/test_v1_pool.jsonl"),
                    help="Input pool JSONL (T-prefixed chunk records).")
    ap.add_argument("--out-dir", type=Path, default=Path("data/test_v1_batches"),
                    help="Directory to write batch_NNN.jsonl files into.")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="Chunks per batch. Default 20 = ~5-10 min per subagent.")
    args = ap.parse_args()

    records = _read_pool(args.pool)
    print(f"Loaded {len(records)} chunks from {args.pool}")

    batches = _split(records, args.batch_size)
    _write_batches(batches, args.out_dir)
    _print_report(batches)
    print(f"\nBatches written to {args.out_dir}/batch_NNN.jsonl")


if __name__ == "__main__":
    main()
