"""P2 step 1: Build a stratified sample of ~150 chunks for subagent verification.

Sampling source: data/layer5v4_lang_fix.jsonl (Gemma+regex spans, corrected
langs from honest fasttext re-detect).

Stratification target: 5 target langs × 8 categories = 40 cells. Goal is
4 chunks per cell, but many cells have few chunks (en/secret has 4 total),
so we ceiling-cap per cell and accept thin cells.

Negative chunks: also include 20 "no PII" chunks for negative-class
agreement scoring.

Outputs:
  data/p2_sample.jsonl       — 150-ish chunks WITH Gemma+regex spans (gold for comparison)
  data/p2_unlabeled.jsonl    — same chunks WITHOUT spans (input to subagents)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(2026)

IN_PATH = Path("data/layer5v4_lang_fix.jsonl")
OUT_LABELED = Path("data/p2_sample.jsonl")
OUT_UNLABELED = Path("data/p2_unlabeled.jsonl")

TARGET_LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]
PER_CELL = 4   # chunks per (lang × cat); some cells will fall short
N_NEG = 20     # negative chunks (no PII spans) for negative-class scoring


def main() -> None:
    # Bucket chunks by (lang, primary_cat). A chunk with multiple cats
    # contributes to all of its cats; we sample without replacement at
    # the chunk level, so a chunk only ends up in the sample once.
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    negatives: dict[str, list[dict]] = defaultdict(list)
    n = 0
    n_skipped_other = 0

    print(f"Reading: {IN_PATH}")
    with IN_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            la = rec.get("language", "")
            if la not in TARGET_LANGS:
                n_skipped_other += 1
                continue
            spans = rec.get("spans", [])
            if not spans:
                negatives[la].append(rec)
                continue
            cats_in_chunk = {sp["label"] for sp in spans}
            for cat in cats_in_chunk:
                if cat in CATS:
                    by_cell[(la, cat)].append(rec)
            if n % 500_000 == 0:
                print(f"  scanned {n:,}", flush=True)

    print()
    print(f"Total chunks scanned: {n:,}")
    print(f"  skipped (non-target lang): {n_skipped_other:,}")
    print()
    print("Cell sizes (lang × cat) before sampling:")
    for la in TARGET_LANGS:
        for cat in CATS:
            sz = len(by_cell.get((la, cat), []))
            print(f"  {la:<8} {cat:<22} {sz:>10,}")

    # Stratified sample: PER_CELL per cell, ceiling-capped.
    sampled: dict[str, dict] = {}  # by chunk id, dedup across cells
    print()
    print(f"Sampling up to {PER_CELL} per cell ...")
    for la in TARGET_LANGS:
        for cat in CATS:
            pool = by_cell.get((la, cat), [])
            if not pool:
                continue
            k = min(PER_CELL, len(pool))
            for rec in random.sample(pool, k):
                sampled[rec["id"]] = rec

    # Negative sample: 4 per language (or fewer if pool is tiny).
    neg_per_lang = max(1, N_NEG // len(TARGET_LANGS))
    for la in TARGET_LANGS:
        pool = negatives.get(la, [])
        if not pool:
            continue
        k = min(neg_per_lang, len(pool))
        for rec in random.sample(pool, k):
            sampled[rec["id"]] = rec

    print(f"Sampled {len(sampled)} unique chunks.")
    print()

    # Sanity: count what we ended up with
    cell_sample = defaultdict(int)
    n_neg_sample = 0
    for rec in sampled.values():
        if not rec.get("spans"):
            n_neg_sample += 1
            continue
        cats = {sp["label"] for sp in rec["spans"]}
        for cat in cats:
            cell_sample[(rec["language"], cat)] += 1
    print("Chunks-per-cell in final sample (chunks may count in multiple cells):")
    for la in TARGET_LANGS:
        row = " ".join(f"{cell_sample.get((la, c), 0):>3}" for c in CATS)
        print(f"  {la:<8} {row}")
    print(f"  negatives: {n_neg_sample}")

    # Write outputs.
    with OUT_LABELED.open("w") as fl, OUT_UNLABELED.open("w") as fu:
        for rec in sampled.values():
            fl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stripped = {
                "id": rec["id"],
                "text": rec["text"],
                "language": rec["language"],
                "subset": rec.get("subset", ""),
            }
            fu.write(json.dumps(stripped, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote labeled gold:    {OUT_LABELED}")
    print(f"Wrote unlabeled input: {OUT_UNLABELED}")


if __name__ == "__main__":
    main()
