"""P6 audit step 1: Stratified sample of val + test for subagent quality audit.

Reads:  data/gheim_pii_balanced_validation.jsonl  (read-only, chmod 444)
        data/gheim_pii_balanced_test.jsonl        (read-only, chmod 444)

Writes: data/p6_audit_sample.jsonl       — chunks WITH gold spans (from val/test)
        data/p6_audit_unlabeled.jsonl    — same chunks, spans stripped (for subagents)

Sampling target: 15 chunks per (lang × cat) cell + 80 negatives ≈ 580 total.
A chunk is sampled per cell once even if it contains multiple cats (so the
total is bounded by the union, not the sum). Each sampled chunk records
its original split via `original_split: "validation"` or `"test"` so
we can report F1 per-split later.

Source files are NEVER modified.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

VAL_PATH = Path("data/gheim_pii_balanced_validation.jsonl")
TEST_PATH = Path("data/gheim_pii_balanced_test.jsonl")
OUT_GOLD = Path("data/p6_audit_sample.jsonl")
OUT_UNLABELED = Path("data/p6_audit_unlabeled.jsonl")

LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]
PER_CELL = 15
N_NEG = 80  # split per language

SEED = 2026


def main() -> None:
    rng = random.Random(SEED)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    negatives: dict[str, list[dict]] = defaultdict(list)
    n = 0

    for path, split_name in [(VAL_PATH, "validation"), (TEST_PATH, "test")]:
        print(f"Reading: {path}", flush=True)
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                rec["original_split"] = split_name
                n += 1
                la = rec["language"]
                if la not in LANGS:
                    continue
                spans = rec.get("spans", [])
                if not spans:
                    negatives[la].append(rec)
                    continue
                cats_in_chunk = {sp["label"] for sp in spans}
                for cat in cats_in_chunk:
                    if cat in CATS:
                        by_cell[(la, cat)].append(rec)

    print(f"\nScanned {n:,} chunks across val + test")
    print()
    print("Pool sizes per (lang × cat):")
    for la in LANGS:
        row = " ".join(f"{len(by_cell.get((la, c), [])):>6}" for c in CATS)
        print(f"  {la:<8} {row}")

    sampled: dict[str, dict] = {}  # by chunk id, dedupes if a chunk hits multiple cells

    print()
    print(f"Sampling up to {PER_CELL} per cell ...")
    for la in LANGS:
        for cat in CATS:
            pool = by_cell.get((la, cat), [])
            if not pool:
                continue
            k = min(PER_CELL, len(pool))
            for rec in rng.sample(pool, k):
                sampled[rec["id"]] = rec

    neg_per_lang = max(1, N_NEG // len(LANGS))
    for la in LANGS:
        pool = negatives.get(la, [])
        if not pool:
            continue
        k = min(neg_per_lang, len(pool))
        for rec in rng.sample(pool, k):
            sampled[rec["id"]] = rec

    print(f"Sampled {len(sampled)} unique chunks")
    print()

    # Final per-cell counts in the sample
    cell_sample = defaultdict(int)
    n_neg_sample = 0
    split_sample = {"validation": 0, "test": 0}
    for rec in sampled.values():
        split_sample[rec["original_split"]] += 1
        if not rec.get("spans"):
            n_neg_sample += 1
            continue
        cats = {sp["label"] for sp in rec["spans"]}
        for cat in cats:
            cell_sample[(rec["language"], cat)] += 1
    print("Sampled chunks per (lang × cat) (chunks may count in multiple cells):")
    print(f"  {'lang':<8} " + " ".join(f"{c:>5}" for c in [c[:5] for c in CATS]))
    for la in LANGS:
        row = " ".join(f"{cell_sample.get((la, c), 0):>5}" for c in CATS)
        print(f"  {la:<8} {row}")
    print(f"  negatives: {n_neg_sample}")
    print(f"  by split: validation={split_sample['validation']}  test={split_sample['test']}")

    with OUT_GOLD.open("w") as fl, OUT_UNLABELED.open("w") as fu:
        for rec in sampled.values():
            fl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stripped = {
                "id": rec["id"],
                "text": rec["text"],
                "language": rec["language"],
                "subset": rec.get("subset", ""),
                "original_split": rec["original_split"],
            }
            fu.write(json.dumps(stripped, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote labeled gold:    {OUT_GOLD}  ({len(sampled)} chunks)")
    print(f"Wrote unlabeled input: {OUT_UNLABELED}")


if __name__ == "__main__":
    main()
