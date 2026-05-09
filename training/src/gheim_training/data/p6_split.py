"""P6 step 1: Stratified document-level train/val/test split.

Reads:  data/gheim_pii_balanced.jsonl  (read-only)
Writes: data/gheim_pii_balanced_train.jsonl
        data/gheim_pii_balanced_validation.jsonl
        data/gheim_pii_balanced_test.jsonl

Constraints:
  - Doc-level isolation: every doc_id lives in exactly one split. Chunks
    from the same document never leak across splits.
  - Stratification on (language, subset) — 5 langs × 4 subsets = up to
    20 strata. Within each stratum, docs are deterministically shuffled
    (seed=42) and greedily assigned to splits 80/10/10 by chunk count.
  - Negatives (chunks with no spans) inherit their doc's split.

Default ratios: 80% train / 10% validation / 10% test.

Usage:
  uv run --project training python -m gheim_training.data.p6_split
  uv run --project training python -m gheim_training.data.p6_split --write
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

IN_PATH = Path("data/gheim_pii_balanced.jsonl")
OUT_TRAIN = Path("data/gheim_pii_balanced_train.jsonl")
OUT_VAL = Path("data/gheim_pii_balanced_validation.jsonl")
OUT_TEST = Path("data/gheim_pii_balanced_test.jsonl")

LANGS_ORDER = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]

RATIO_TRAIN = 0.80
RATIO_VAL = 0.10
RATIO_TEST = 0.10
SEED = 42


@dataclass
class DocMeta:
    doc_id: str
    lang: str
    subset: str
    n_chunks: int
    n_pos: int  # chunks with ≥1 span
    n_neg: int  # chunks with 0 spans
    cats: set[str]  # union of cats across all chunks of this doc


def scan_docs() -> tuple[dict[str, DocMeta], int]:
    """Pass 1: build per-doc metadata."""
    docs: dict[str, DocMeta] = {}
    n_total = 0
    print(f"Reading: {IN_PATH}", flush=True)
    with IN_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            doc_id = rec.get("doc_id", "")
            lang = rec.get("language", "")
            subset = rec.get("subset") or "unknown"
            spans = rec.get("spans", [])
            if doc_id not in docs:
                docs[doc_id] = DocMeta(
                    doc_id=doc_id, lang=lang, subset=subset,
                    n_chunks=0, n_pos=0, n_neg=0, cats=set(),
                )
            d = docs[doc_id]
            d.n_chunks += 1
            if spans:
                d.n_pos += 1
                for sp in spans:
                    d.cats.add(sp.get("label", ""))
            else:
                d.n_neg += 1
            if n_total % 50_000 == 0:
                print(f"  scanned {n_total:,}", flush=True)
    return docs, n_total


def assign_splits(docs: dict[str, DocMeta]) -> dict[str, str]:
    """Greedy stratified assignment of docs to splits.

    For each (lang, subset) stratum, shuffle docs deterministically and
    fill train → val → test by chunk-count quota until each quota is met.
    """
    rng = random.Random(SEED)
    by_stratum: dict[tuple[str, str], list[DocMeta]] = defaultdict(list)
    for d in docs.values():
        by_stratum[(d.lang, d.subset)].append(d)

    assignment: dict[str, str] = {}
    print()
    print(f"{'lang':<8} {'subset':<22} {'n_docs':>8} {'n_chunks':>10} "
          f"{'q_train':>10} {'q_val':>8} {'q_test':>8}")
    for stratum, dlist in sorted(by_stratum.items()):
        # Sort by doc_id for determinism, then shuffle with seed
        dlist.sort(key=lambda d: d.doc_id)
        rng.shuffle(dlist)
        total_chunks = sum(d.n_chunks for d in dlist)
        q_train = int(total_chunks * RATIO_TRAIN)
        q_val = int(total_chunks * RATIO_VAL)
        # Test gets the remainder so totals match exactly
        q_test = total_chunks - q_train - q_val

        train_so_far = val_so_far = test_so_far = 0
        for d in dlist:
            if train_so_far + d.n_chunks <= q_train:
                assignment[d.doc_id] = "train"
                train_so_far += d.n_chunks
            elif val_so_far + d.n_chunks <= q_val:
                assignment[d.doc_id] = "validation"
                val_so_far += d.n_chunks
            else:
                assignment[d.doc_id] = "test"
                test_so_far += d.n_chunks

        lang, subset = stratum
        print(f"{lang:<8} {subset:<22} {len(dlist):>8,} {total_chunks:>10,} "
              f"{q_train:>10,} {q_val:>8,} {q_test:>8,}")
    return assignment


def report_split_distribution(docs: dict[str, DocMeta], assignment: dict[str, str]) -> None:
    """Per-(lang × cat) chunk counts in each split."""
    pos_in_split: Counter[str] = Counter()
    neg_in_split: Counter[str] = Counter()
    lang_in_split: dict[tuple[str, str], int] = Counter()

    for d in docs.values():
        sp = assignment[d.doc_id]
        pos_in_split[sp] += d.n_pos
        neg_in_split[sp] += d.n_neg
        lang_in_split[(d.lang, sp)] += d.n_chunks

    # For per-cell counts we need to scan again (cell is per chunk, not per doc)
    print()
    print(f"{'=' * 80}")
    print("PER-(lang × cat) CHUNK COUNTS BY SPLIT")
    print(f"{'=' * 80}")
    cell_counts: dict[tuple[str, str, str], int] = Counter()  # (split, lang, cat)
    print("Re-scanning chunks for per-cell counts ...", flush=True)
    n = 0
    with IN_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            sp = assignment.get(rec["doc_id"], "?")
            lang = rec["language"]
            cats_in_chunk = {s["label"] for s in rec.get("spans", [])}
            for cat in cats_in_chunk:
                cell_counts[(sp, lang, cat)] += 1

    print(f"\n{'lang':<8} {'cat':<22} {'train':>10} {'val':>8} {'test':>8} "
          f"{'val%':>6} {'test%':>6}")
    for lang in LANGS_ORDER:
        for cat in CATS:
            tr = cell_counts.get(("train", lang, cat), 0)
            va = cell_counts.get(("validation", lang, cat), 0)
            te = cell_counts.get(("test", lang, cat), 0)
            tot = tr + va + te
            if tot == 0:
                continue
            print(f"{lang:<8} {cat:<22} {tr:>10,} {va:>8,} {te:>8,} "
                  f"{100*va/tot:>5.1f}% {100*te/tot:>5.1f}%")

    # Per-language summary
    print()
    print(f"{'lang':<8} {'train_pos':>10} {'val_pos':>8} {'test_pos':>8} "
          f"{'train_neg':>10} {'val_neg':>8} {'test_neg':>8} {'total':>10}")
    for lang in LANGS_ORDER:
        tr = lang_in_split.get((lang, "train"), 0)
        va = lang_in_split.get((lang, "validation"), 0)
        te = lang_in_split.get((lang, "test"), 0)
        # We don't have pos/neg per (lang, split) directly; approximate
        print(f"{lang:<8} {tr:>10,} {va:>8,} {te:>8,} {'-':>10} {'-':>8} {'-':>8} "
              f"{tr + va + te:>10,}")

    print()
    print(f"OVERALL: train={pos_in_split['train'] + neg_in_split['train']:,}  "
          f"val={pos_in_split['validation'] + neg_in_split['validation']:,}  "
          f"test={pos_in_split['test'] + neg_in_split['test']:,}")
    print(f"  positives: train={pos_in_split['train']:,}  "
          f"val={pos_in_split['validation']:,}  test={pos_in_split['test']:,}")
    print(f"  negatives: train={neg_in_split['train']:,}  "
          f"val={neg_in_split['validation']:,}  test={neg_in_split['test']:,}")


def write_splits(assignment: dict[str, str]) -> None:
    """Pass 2: stream source file, write each chunk to its split."""
    out = {
        "train": OUT_TRAIN.open("w"),
        "validation": OUT_VAL.open("w"),
        "test": OUT_TEST.open("w"),
    }
    counts = Counter()
    n = 0
    with IN_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            sp = assignment.get(rec["doc_id"], "train")  # fallback
            out[sp].write(line)
            counts[sp] += 1
            if n % 50_000 == 0:
                print(f"  wrote {n:,}", flush=True)
    for f in out.values():
        f.close()
    print()
    for sp in ("train", "validation", "test"):
        print(f"  {sp}: {counts[sp]:,} chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Actually write the three split files (default: dry-run)")
    args = parser.parse_args()

    docs, n_total = scan_docs()
    print()
    print(f"Loaded {n_total:,} chunks across {len(docs):,} unique documents")
    print(f"  median chunks/doc: "
          f"{sorted(d.n_chunks for d in docs.values())[len(docs) // 2]}")
    print(f"  max chunks/doc:    {max(d.n_chunks for d in docs.values())}")

    print()
    print(f"--- Stratified assignment (target {RATIO_TRAIN:.0%}/{RATIO_VAL:.0%}/{RATIO_TEST:.0%}) ---")
    assignment = assign_splits(docs)
    report_split_distribution(docs, assignment)

    if not args.write:
        print("\n[DRY-RUN] no files written. Add --write to produce:")
        print(f"  {OUT_TRAIN}")
        print(f"  {OUT_VAL}")
        print(f"  {OUT_TEST}")
        return

    print("\n[WRITE] producing split files ...")
    write_splits(assignment)


if __name__ == "__main__":
    main()
