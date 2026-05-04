"""Combine layer JSONLs into a HuggingFace ``datasets.DatasetDict``.

  - Reads all input JSONLs (any subset of L1/L2/L3/L4)
  - Dedupes by exact text match (cheap proxy; cross-layer near-duplicates rare)
  - Stratified split by (language, source) into train / validation
  - Writes to ``--out-dir`` as a HF dataset directory (load_from_disk-able)

Run:
    uv run python -m gheim_training.data.build \\
        --in data/layer1.jsonl data/layer2.jsonl data/layer3_apertus.jsonl data/layer4_en.jsonl \\
        --out-dir data/built --val-frac 0.05
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

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
    args = ap.parse_args()

    examples: list[Example] = []
    for p in args.inputs:
        examples.extend(read_jsonl(p))
    print(f"loaded {len(examples)} examples from {len(args.inputs)} files")

    examples = _dedupe(examples)
    print(f"after dedupe: {len(examples)}")

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
