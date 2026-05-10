"""Combine layer JSONLs into a HuggingFace ``datasets.DatasetDict``.

  - Reads all input JSONLs (typically the gold core: L1 + L2.v2 + L4.v2;
    optionally gap-fill layers from Phase B)
  - Dedupes by exact text match (cheap proxy; cross-layer near-duplicates rare)
  - **Runs the coverage gate** — refuses to write the dataset if any
    (language × category) cell falls below ``--min-per-cell`` (default 1000)
    unless explicitly waived via ``--waive lang/category``
  - Stratified split by (language, source) into train / validation
  - Writes to ``--out-dir`` as a HF dataset directory (load_from_disk-able)

Run (gold core for gheim-2):
    uv run python -m gheim_training.data.build \\
        --in data/layer1.jsonl data/layer2_v2.jsonl data/layer4_v2.jsonl \\
        --out-dir data/built --val-frac 0.05

To skip the coverage gate (e.g. while iterating on data design), pass
``--skip-coverage-gate``. Don't ship a checkpoint that was trained that way.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from . import coverage_gate
from .schema import Example, read_jsonl


def _stratified_split(
    examples: list[Example],
    val_frac: float,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    bucketed: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for ex in examples:
        bucketed[(ex.language, ex.source)].append(ex)

    train: list[Example] = []
    val: list[Example] = []
    for items in bucketed.values():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac))) if items else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _dedupe(examples: list[Example]) -> list[Example]:
    seen: set[str] = set()
    out: list[Example] = []
    for ex in examples:
        if ex.text in seen:
            continue
        seen.add(ex.text)
        out.append(ex)
    return out


def _to_hf_record(ex: Example) -> dict:
    return {
        "text": ex.text,
        "spans": [{"start": s.start, "end": s.end, "label": s.label} for s in ex.spans],
        "language": ex.language,
        "source": ex.source,
        "template_id": ex.template_id or "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--min-per-cell", type=int, default=1000,
                    help="Coverage-gate threshold per (language × category).")
    ap.add_argument("--waive", action="append", default=[],
                    help="Cells to waive: lang/category. Repeatable.")
    ap.add_argument("--skip-coverage-gate", action="store_true",
                    help="Skip the coverage gate (do not ship a model trained this way).")
    args = ap.parse_args()

    examples: list[Example] = []
    for p in args.inputs:
        examples.extend(read_jsonl(p))
    print(f"loaded {len(examples)} examples from {len(args.inputs)} files")

    examples = _dedupe(examples)
    print(f"after dedupe: {len(examples)}")

    if not args.skip_coverage_gate:
        print(f"\nrunning coverage gate (min-per-cell={args.min_per_cell})…")
        waivers: set[tuple[str, str]] = set()
        for w in args.waive:
            try:
                lang, cat = w.split("/", 1)
            except ValueError:
                print(f"--waive expects lang/category, got {w!r}", file=sys.stderr)
                sys.exit(2)
            waivers.add((lang, cat))
        counts = coverage_gate.count_examples(args.inputs)
        passed, failures = coverage_gate.report(
            counts, min_per_cell=args.min_per_cell, waivers=waivers,
        )
        if not passed:
            print(f"\nCOVERAGE-GATE FAIL: {len(failures)} cells below threshold:",
                  file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            print("\nFix: gap-fill the missing cells, or pass --waive lang/category "
                  "for cells you accept as gaps. Use --skip-coverage-gate "
                  "to bypass entirely (don't ship that model).", file=sys.stderr)
            sys.exit(1)
        print("PASS\n")

    train, val = _stratified_split(examples, args.val_frac, args.seed)
    print(f"split: train={len(train)} val={len(val)}")

    # Per-bucket stats so we can spot imbalance pre-training.
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for ex in train:
        counts[(ex.language, ex.source)] += 1
    print("train counts (language, source):")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    from datasets import Dataset, DatasetDict
    dd = DatasetDict({
        "train": Dataset.from_list([_to_hf_record(e) for e in train]),
        "validation": Dataset.from_list([_to_hf_record(e) for e in val]),
    })
    dd.save_to_disk(str(args.out_dir))
    print(f"saved → {args.out_dir}")


if __name__ == "__main__":
    main()
