"""P14: Move a stratified slice of synthetic chunks from train to val + test.

Why: gheim_pii_balanced_validation.jsonl and _test.jsonl have zero real
``secret`` coverage across all 5 languages. ``en`` × {email, address,
account_number, phone} are also near-zero. Without representation in the
held-out sets, we can't measure secret detection or English performance
on those cells, and overall F1 is silently degraded by FPs on cells
where there's no gold to anchor.

Fix: take a stratified slice of synthetic chunks (already in train) and
move them to val + test. Train shrinks accordingly (no leakage — moved
chunks are removed from train, kept in val/test).

Per-cell injection plan:
  dev_chat        × 5 langs  → 200 to val, 200 to test, per lang  (1000 + 1000)
                              Provides secret + person + email
  customer_record × en       → 100 to val, 100 to test            (100 + 100)
                              Provides address + account + email + phone + person
  incident_report × en       → 100 to val, 100 to test            (100 + 100)
                              Provides secret + person + phone + email

Total moved per split: 1,200 → val grows 14,634 → 15,834
                              → test grows 14,661 → 15,861
Train shrinks: 142,041 → 142,041 - 2,400 = 139,641

Output:
  data/gheim_pii_balanced_train.jsonl       (rewritten, smaller)
  data/gheim_pii_balanced_validation.jsonl  (rewritten, with synth)
  data/gheim_pii_balanced_test.jsonl        (rewritten, with synth)

Source files NEVER modified beyond this script's controlled rewrite. Run
once; this script refuses to run again if the train file no longer
contains the expected synthetic chunks.

Usage:
  uv run --project training python -m gheim_training.data.p14_inject_synth_to_eval --dry-run
  uv run --project training python -m gheim_training.data.p14_inject_synth_to_eval --write
"""
from __future__ import annotations

import argparse
import json
import os
import random
import stat
from collections import Counter, defaultdict
from pathlib import Path

TRAIN = Path("data/gheim_pii_balanced_train.jsonl")
VAL = Path("data/gheim_pii_balanced_validation.jsonl")
TEST = Path("data/gheim_pii_balanced_test.jsonl")

SEED = 17
LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]

# (template, lang) → (n_to_val, n_to_test)
PLAN: dict[tuple[str, str], tuple[int, int]] = {
    **{("dev_chat", lang): (200, 200) for lang in LANGS},
    ("customer_record", "en"): (100, 100),
    ("incident_report",  "en"): (100, 100),
}


def _read_all(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _writable(path: Path) -> None:
    """Make path writable (val/test are chmod 444)."""
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Rewrite train/val/test files (default: dry-run only)")
    args = parser.parse_args()
    rng = random.Random(SEED)

    print(f"Reading {TRAIN} ...", flush=True)
    train_records = _read_all(TRAIN)
    print(f"  {len(train_records):,} chunks loaded")
    print(f"Reading {VAL} ...", flush=True)
    val_records = _read_all(VAL)
    print(f"  {len(val_records):,} chunks loaded")
    print(f"Reading {TEST} ...", flush=True)
    test_records = _read_all(TEST)
    print(f"  {len(test_records):,} chunks loaded")
    print()

    # Bucket train synthetic chunks by (template, lang)
    by_bucket: dict[tuple[str, str], list[int]] = defaultdict(list)  # → indices into train_records
    for idx, rec in enumerate(train_records):
        if not rec.get("synthetic"):
            continue
        tmpl = rec.get("synthetic_template")
        lang = rec.get("language")
        if not tmpl or not lang:
            continue
        by_bucket[(tmpl, lang)].append(idx)

    print("Synthetic chunks available in train:")
    for (tmpl, lang), idxs in sorted(by_bucket.items()):
        print(f"  {tmpl:<18} {lang:<6} {len(idxs):>5,}")
    print()

    # Pick which to move
    move_to_val: set[int] = set()
    move_to_test: set[int] = set()
    print("Selecting per-cell ...")
    for (tmpl, lang), (n_val, n_test) in PLAN.items():
        pool = list(by_bucket.get((tmpl, lang), []))
        need = n_val + n_test
        if len(pool) < need:
            raise SystemExit(
                f"Not enough synthetic chunks for {tmpl}/{lang}: "
                f"have {len(pool):,}, need {need:,}"
            )
        rng.shuffle(pool)
        v = pool[:n_val]
        t = pool[n_val:n_val + n_test]
        move_to_val.update(v)
        move_to_test.update(t)
        print(f"  {tmpl:<18} {lang:<6} → val:{n_val:>4} test:{n_test:>4}")
    print()
    print(f"Total chunks to move: {len(move_to_val):,} val + {len(move_to_test):,} test = "
          f"{len(move_to_val) + len(move_to_test):,}")

    # Build new train/val/test
    moved = move_to_val | move_to_test
    new_train = [r for i, r in enumerate(train_records) if i not in moved]
    val_addition = [train_records[i] for i in move_to_val]
    test_addition = [train_records[i] for i in move_to_test]

    new_val = val_records + val_addition
    new_test = test_records + test_addition

    print()
    print("Resulting sizes:")
    print(f"  train: {len(train_records):>8,} → {len(new_train):>8,}  "
          f"(removed {len(train_records) - len(new_train):,})")
    print(f"  val:   {len(val_records):>8,} → {len(new_val):>8,}  "
          f"(added {len(val_addition):,})")
    print(f"  test:  {len(test_records):>8,} → {len(new_test):>8,}  "
          f"(added {len(test_addition):,})")

    # Per-cell preview for val + test
    CATS = ["account_number", "private_address", "private_date", "private_email",
            "private_person", "private_phone", "private_url", "secret"]
    print()
    print("Per-cell counts in NEW val:")
    cells_v: Counter[tuple[str, str]] = Counter()
    for rec in new_val:
        cats = {sp["label"] for sp in rec.get("spans", [])}
        for c in cats:
            cells_v[(rec["language"], c)] += 1
    print(f"  {'lang':<6} " + " ".join(f"{c[:5]:>6}" for c in CATS))
    for la in LANGS:
        row = " ".join(f"{cells_v.get((la, c), 0):>6}" for c in CATS)
        print(f"  {la:<6} {row}")

    print()
    print("Per-cell counts in NEW test:")
    cells_t: Counter[tuple[str, str]] = Counter()
    for rec in new_test:
        cats = {sp["label"] for sp in rec.get("spans", [])}
        for c in cats:
            cells_t[(rec["language"], c)] += 1
    print(f"  {'lang':<6} " + " ".join(f"{c[:5]:>6}" for c in CATS))
    for la in LANGS:
        row = " ".join(f"{cells_t.get((la, c), 0):>6}" for c in CATS)
        print(f"  {la:<6} {row}")

    if not args.write:
        print()
        print("[DRY-RUN] No files modified. Re-run with --write to apply.")
        return

    # ---------- Rewrite ----------
    print()
    print("[WRITE] Rewriting train/val/test ...")
    # train: simple write (was already writable)
    with TRAIN.open("w") as f:
        for rec in new_train:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {len(new_train):,} → {TRAIN}")

    # val + test: chmod +w, write, chmod 444
    _writable(VAL)
    with VAL.open("w") as f:
        for rec in new_val:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _readonly(VAL)
    print(f"  wrote {len(new_val):,} → {VAL} (re-locked)")

    _writable(TEST)
    with TEST.open("w") as f:
        for rec in new_test:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _readonly(TEST)
    print(f"  wrote {len(new_test):,} → {TEST} (re-locked)")

    print()
    print("Next: rebuild HF dataset:")
    print("  uv run --project training python -m gheim_training.data.p11_build_hf")


if __name__ == "__main__":
    main()
