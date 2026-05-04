"""Layer 4: English anchor from pii-masking-300k to prevent catastrophic forgetting.

Uses the same REMAP and parsing logic as ai4privacy.py but filters for English
records. Target volume: ~25% of total dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .ai4privacy import _spans_from_record
from .schema import Example, write_jsonl


def load(
    dataset_name: str = "ai4privacy/pii-masking-300k",
    split: str = "train",
    n: int = 30_000,
) -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    out: list[Example] = []
    for rec in ds:
        loc = rec.get("language") or rec.get("locale")
        if loc != "English":
            continue
        ex = _spans_from_record(rec, language="en")
        if ex is None:
            continue
        # mark source so build.py can balance the mix
        ex.source = "english_anchor"  # type: ignore[assignment]
        out.append(ex)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30_000)
    args = ap.parse_args()
    examples = load(n=args.n)
    n = write_jsonl(args.out, examples)
    print(f"wrote {n} English-anchor examples → {args.out}")


if __name__ == "__main__":
    main()
