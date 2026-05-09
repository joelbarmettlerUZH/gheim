"""P5: Build the balanced gheim-pii dataset from layer5v4 + regex_new.

Source files are NEVER modified. Default mode is --dry-run (prints
selection statistics only). With --write the script produces a NEW file
at data/gheim_pii_balanced.jsonl.

Pipeline:
  1. Load + filter
       - keep target langs (de_ch/fr_ch/it_ch/rm/en)
       - dedup chunks with identical text (catches the bilingual-corpus
         double-loading)
       - drop literal placeholder spans (value == "<email-pii>")
  2. Build indexes (by_doc, by_value, by_cell)
  3. Phase A — per-doc cap: sample CAP_DOC chunks per doc_id
  4. Phase B — per-value cap: sample CAP_VALUE chunks per (lang, cat,
     normalized value); a chunk is kept if it is selected for at least
     one of its (cat, value) tuples
  5. Phase C — per-cell cap: tier-based per (lang, cat) ceiling
  6. Phase D — negatives at NEG_RATIO of positives, sampled per-language

Inputs (read-only):
  data/layer5v4_lang_fix.jsonl
  data/layer5v4_regex_new_lang_fix.jsonl

Output (only with --write):
  data/gheim_pii_balanced.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

INPUTS = [
    Path("data/layer5v4_lang_fix.jsonl"),
    Path("data/layer5v4_regex_new_lang_fix.jsonl"),
]
OUT_PATH = Path("data/gheim_pii_balanced.jsonl")
TARGET_LANGS = {"de_ch", "fr_ch", "it_ch", "rm", "en"}
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]
LANGS_ORDER = ["de_ch", "fr_ch", "it_ch", "rm", "en"]

CAP_DOC = 30
CAP_VALUE = 30

# Per-cell ceilings (chunks containing at least one span of this cat in this lang)
CAP_CELL_MAJOR = 8000   # de/fr/it × {person, date, address, url, phone}
CAP_CELL_MINOR = 3000   # de/fr/it × {email, account_number}
CAP_CELL_RM = 2000      # rm × *
CAP_CELL_EN = 1500      # en × *
# secret: take all (sparse — < 60 per lang)

NEG_RATIO = 0.10        # 10% of positives, sampled per-language

PLACEHOLDER_VALUE = "<email-pii>"
SEED = 42


def cap_for_cell(lang: str, cat: str) -> int | None:
    if cat == "secret":
        return None
    if lang == "rm":
        return CAP_CELL_RM
    if lang == "en":
        return CAP_CELL_EN
    if cat in ("private_email", "account_number"):
        return CAP_CELL_MINOR
    return CAP_CELL_MAJOR


def normalize_value(value: str, label: str) -> str:
    v = value.strip()
    if label in ("private_person", "private_address"):
        return v.casefold()
    return v


@dataclass
class ChunkMeta:
    cid: str
    lang: str
    subset: str
    doc_id: str
    spans: list[tuple[str, str]]  # (label, normalized_value)
    is_neg: bool


def load_chunks() -> tuple[dict[str, ChunkMeta], dict]:
    """Pass 1: load metadata, filter, dedup. Returns (chunks, stats)."""
    chunks: dict[str, ChunkMeta] = {}
    seen_text_hashes: set[str] = set()
    n_total = n_dropped_lang = n_dropped_dup = n_placeholder = 0

    for path in INPUTS:
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        print(f"Reading: {path}", flush=True)
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                n_total += 1
                lang = rec.get("language", "")
                if lang not in TARGET_LANGS:
                    n_dropped_lang += 1
                    continue
                text = rec.get("text", "")
                th = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                if th in seen_text_hashes:
                    n_dropped_dup += 1
                    continue
                seen_text_hashes.add(th)

                spans_filtered: list[tuple[str, str]] = []
                for sp in rec.get("spans", []):
                    val = sp.get("value", "").strip()
                    if val == PLACEHOLDER_VALUE:
                        n_placeholder += 1
                        continue
                    label = sp.get("label", "")
                    spans_filtered.append((label, normalize_value(val, label)))

                cid = rec["id"]
                chunks[cid] = ChunkMeta(
                    cid=cid,
                    lang=lang,
                    subset=rec.get("subset") or "unknown",
                    doc_id=rec.get("doc_id", ""),
                    spans=spans_filtered,
                    is_neg=(len(spans_filtered) == 0),
                )
                if n_total % 500_000 == 0:
                    print(f"  scanned {n_total:,}", flush=True)

    return chunks, {
        "total": n_total,
        "dropped_lang": n_dropped_lang,
        "dropped_dup": n_dropped_dup,
        "placeholder_spans": n_placeholder,
        "kept": len(chunks),
    }


def build_indexes(chunks: dict[str, ChunkMeta]):
    by_doc: dict[str, list[str]] = defaultdict(list)
    by_value: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    negatives: list[str] = []

    for cid, m in chunks.items():
        if m.doc_id:
            by_doc[m.doc_id].append(cid)
        if m.is_neg:
            negatives.append(cid)
            continue
        cats_in_chunk = set()
        for label, value in m.spans:
            by_value[(m.lang, label, value)].append(cid)
            cats_in_chunk.add(label)
        for cat in cats_in_chunk:
            by_cell[(m.lang, cat)].append(cid)
    return by_doc, by_value, by_cell, negatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Actually write the balanced file (default: dry-run)")
    args = parser.parse_args()

    rng = random.Random(SEED)

    # ---------- Pass 1 ----------
    chunks, stats = load_chunks()
    print()
    print(f"Loaded {stats['total']:,} total → kept {stats['kept']:,}")
    print(f"  dropped (non-target lang):    {stats['dropped_lang']:,}")
    print(f"  dropped (duplicate text):     {stats['dropped_dup']:,}")
    print(f"  filtered (placeholder spans): {stats['placeholder_spans']:,}")

    by_doc, by_value, by_cell, negatives = build_indexes(chunks)
    print(f"\nIndexed: {len(by_doc):,} docs, {len(by_value):,} (lang,cat,value) tuples, "
          f"{len(by_cell)} cells, {len(negatives):,} negatives")

    # ---------- Phase A ----------
    print(f"\n--- Phase A: per-doc cap (CAP_DOC={CAP_DOC}) ---")
    kept_a: set[str] = set()
    docs_capped = chunks_dropped_a = 0
    for doc_id, cids in by_doc.items():
        if len(cids) > CAP_DOC:
            kept_a.update(rng.sample(cids, CAP_DOC))
            docs_capped += 1
            chunks_dropped_a += len(cids) - CAP_DOC
        else:
            kept_a.update(cids)
    no_doc = {cid for cid, m in chunks.items() if not m.doc_id}
    kept_a.update(no_doc)
    print(f"  {docs_capped:,} docs capped, {chunks_dropped_a:,} chunks dropped")
    print(f"  kept: {len(kept_a):,} chunks")

    # ---------- Phase B ----------
    print(f"\n--- Phase B: per-value cap (CAP_VALUE={CAP_VALUE}) ---")
    kept_b: set[str] = set()
    values_capped = 0
    for (lang, cat, value), cids in by_value.items():
        pool = [c for c in cids if c in kept_a]
        if not pool:
            continue
        if len(pool) > CAP_VALUE:
            kept_b.update(rng.sample(pool, CAP_VALUE))
            values_capped += 1
        else:
            kept_b.update(pool)
    print(f"  {values_capped:,} (lang,cat,value) tuples capped")
    print(f"  positives kept: {len(kept_b):,}")

    # ---------- Phase C ----------
    print(f"\n--- Phase C: per-cell cap ---")
    cell_picks: dict[tuple[str, str], set[str]] = {}
    for cell in by_cell:
        pool = [c for c in by_cell[cell] if c in kept_b]
        cap = cap_for_cell(*cell)
        if cap is None or len(pool) <= cap:
            cell_picks[cell] = set(pool)
        else:
            cell_picks[cell] = set(rng.sample(pool, cap))
    kept_c: set[str] = set().union(*cell_picks.values())
    print(f"  positives kept: {len(kept_c):,}")

    # ---------- Phase D ----------
    print(f"\n--- Phase D: negatives ({int(NEG_RATIO * 100)}% of positives, per-language) ---")
    neg_pool_in_a = [c for c in negatives if c in kept_a]
    pos_per_lang = Counter(chunks[c].lang for c in kept_c)
    kept_neg: set[str] = set()
    for lang in LANGS_ORDER:
        target = int(pos_per_lang.get(lang, 0) * NEG_RATIO)
        pool_l = [c for c in neg_pool_in_a if chunks[c].lang == lang]
        if target >= len(pool_l):
            kept_neg.update(pool_l)
            print(f"  {lang:<8} target={target:,}  pool={len(pool_l):,}  used=ALL ({len(pool_l):,})")
        else:
            picked = rng.sample(pool_l, target)
            kept_neg.update(picked)
            print(f"  {lang:<8} target={target:,}  pool={len(pool_l):,}  used={target:,}")

    final = kept_c | kept_neg
    print(f"\nFINAL: {len(final):,} chunks ({len(kept_c):,} pos + {len(kept_neg):,} neg)")

    # ---------- Report ----------
    print(f"\n{'=' * 90}")
    print("PER-CELL FUNNEL (chunks containing at least one span of this cat)")
    print(f"{'=' * 90}")
    print(f"{'lang':<8} {'cat':<22} {'before':>10} {'after_A':>10} {'after_B':>10} {'after_C':>10} {'cap':>8}")
    for lang in LANGS_ORDER:
        for cat in CATS:
            cell = (lang, cat)
            before = len(by_cell.get(cell, []))
            if before == 0:
                continue
            after_a = sum(1 for c in by_cell[cell] if c in kept_a)
            after_b = sum(1 for c in by_cell[cell] if c in kept_b)
            after_c = len(cell_picks.get(cell, set()))
            cap = cap_for_cell(lang, cat)
            cap_s = "all" if cap is None else f"{cap:,}"
            print(f"{lang:<8} {cat:<22} {before:>10,} {after_a:>10,} {after_b:>10,} {after_c:>10,} {cap_s:>8}")

    print()
    print(f"{'lang':<8} {'positives':>12} {'negatives':>12} {'total':>12}")
    for lang in LANGS_ORDER:
        pos = sum(1 for c in kept_c if chunks[c].lang == lang)
        neg = sum(1 for c in kept_neg if chunks[c].lang == lang)
        print(f"{lang:<8} {pos:>12,} {neg:>12,} {pos + neg:>12,}")

    print()
    print("Per-subset distribution (final):")
    by_subset = Counter(chunks[c].subset for c in final)
    for s, n in by_subset.most_common():
        print(f"  {s:<22} {n:>10,}")

    if not args.write:
        print(f"\n[DRY-RUN] no file written. Add --write to produce {OUT_PATH}")
        return

    # ---------- Write ----------
    print(f"\n[WRITE] Writing {len(final):,} chunks to {OUT_PATH}")
    n_written = 0
    seen_written: set[str] = set()
    with OUT_PATH.open("w") as fout:
        for path in INPUTS:
            if not path.exists():
                continue
            with path.open() as fin:
                for line in fin:
                    rec = json.loads(line)
                    cid = rec.get("id")
                    if cid not in final or cid in seen_written:
                        continue
                    seen_written.add(cid)
                    rec["spans"] = [
                        sp for sp in rec.get("spans", [])
                        if sp.get("value", "").strip() != PLACEHOLDER_VALUE
                    ]
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
    print(f"  wrote {n_written:,} chunks")


if __name__ == "__main__":
    main()
