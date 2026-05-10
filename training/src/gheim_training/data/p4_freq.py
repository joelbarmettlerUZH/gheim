"""P4: Entity / date / URL frequency analysis on the post-Phase-1 corpus.

Goal: identify what's over-represented so P5 can balance with informed
down-sampling. Specifically:

  1. Top N most-frequent span values per (lang × cat) — surfaces the
     "Patrick Fischer × 1000" problem.
  2. Per-(lang × cat × subset) chunk distribution — surfaces register
     skew (entscheidsuche dominates, fineweb is heterogeneous).
  3. Doc-level concentration — which docs contribute the most chunks?
     With per_doc_cap removed, some docs may have 100+ chunks.
  4. Quick estimate of duplicate-text density via chunk-text length
     bucketing + cheap exact-match check.

Inputs:
  data/layer5v4_lang_fix.jsonl              — Gemma + corrected langs (2.31M)
  data/layer5v4_regex_new_lang_fix.jsonl    — regex-only new chunks   (1.38M)

Output: human-readable report on stdout.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

INPUTS = [
    Path("data/layer5v4_lang_fix.jsonl"),
    Path("data/layer5v4_regex_new_lang_fix.jsonl"),
]
LANGS_TARGET = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]
TOP_N = 30


def _normalize_value(value: str, label: str) -> str:
    """Light normalization for grouping near-duplicates of the same value."""
    v = value.strip()
    # Person/address: case-fold for grouping ('Patrick Fischer' = 'patrick fischer')
    if label in ("private_person", "private_address"):
        return v.casefold()
    return v


def main() -> None:
    # Aggregations
    span_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    chunks_per_lang_cat_subset: Counter[tuple[str, str, str]] = Counter()
    chunks_per_doc: Counter[str] = Counter()
    chunks_per_lang: Counter[str] = Counter()
    spans_per_lang_cat: Counter[tuple[str, str]] = Counter()
    n_total = 0

    for path in INPUTS:
        if not path.exists():
            print(f"WARNING: missing {path}, skipping")
            continue
        print(f"Reading: {path}")
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                n_total += 1
                lang = rec.get("language", "")
                subset = rec.get("subset", "unknown")
                doc_id = rec.get("doc_id", "")
                chunks_per_lang[lang] += 1
                chunks_per_doc[doc_id] += 1

                spans = rec.get("spans", [])
                cats_in_chunk = set()
                for sp in spans:
                    label = sp["label"]
                    val = _normalize_value(sp["value"], label)
                    span_counts[(lang, label, val)] += 1
                    spans_per_lang_cat[(lang, label)] += 1
                    cats_in_chunk.add(label)
                for cat in cats_in_chunk:
                    chunks_per_lang_cat_subset[(lang, cat, subset)] += 1
                if n_total % 500_000 == 0:
                    print(f"  scanned {n_total:,}", flush=True)

    print()
    print(f"TOTAL chunks across all inputs: {n_total:,}")
    print()

    # --- Section 1: top span values per (lang × cat) ---
    print("=" * 80)
    print(f"TOP {TOP_N} MOST-FREQUENT SPAN VALUES per (lang × cat)")
    print("=" * 80)

    by_lang_cat: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (lang, label, val), c in span_counts.items():
        by_lang_cat[(lang, label)].append((val, c))

    for la in LANGS_TARGET:
        for cat in CATS:
            items = by_lang_cat.get((la, cat), [])
            if not items:
                continue
            items.sort(key=lambda x: -x[1])
            total_spans = sum(c for _, c in items)
            unique_vals = len(items)
            top = items[:TOP_N]
            top_share = sum(c for _, c in top) / total_spans if total_spans else 0
            print()
            print(f"--- {la} × {cat}  ({total_spans:,} spans, {unique_vals:,} unique values; "
                  f"top-{TOP_N} = {top_share*100:.1f}% of all spans) ---")
            for val, c in top:
                short_val = val if len(val) < 60 else val[:57] + "..."
                pct = 100 * c / total_spans
                print(f"  {c:>8,}  ({pct:>5.2f}%)  {short_val}")

    # --- Section 2: per (lang × cat × subset) chunk distribution ---
    print()
    print("=" * 80)
    print("CHUNKS per (lang × cat × subset)  [target langs only]")
    print("=" * 80)
    subsets = sorted({s for _, _, s in chunks_per_lang_cat_subset})
    print(f"\n{'lang':<8} {'cat':<22} " + " ".join(f"{s:>14}" for s in subsets))
    for la in LANGS_TARGET:
        for cat in CATS:
            row = [chunks_per_lang_cat_subset.get((la, cat, s), 0) for s in subsets]
            if sum(row) == 0:
                continue
            print(f"{la:<8} {cat:<22} " + " ".join(f"{v:>14,}" for v in row))

    # --- Section 3: doc-level concentration ---
    print()
    print("=" * 80)
    print("DOC-LEVEL CONCENTRATION (top 30 docs by chunk count)")
    print("=" * 80)
    print(f"Total unique docs: {len(chunks_per_doc):,}")
    median_chunks = sorted(chunks_per_doc.values())[len(chunks_per_doc) // 2] if chunks_per_doc else 0
    p99_chunks = sorted(chunks_per_doc.values())[int(len(chunks_per_doc) * 0.99)] if chunks_per_doc else 0
    print(f"Median chunks per doc: {median_chunks}")
    print(f"P99 chunks per doc: {p99_chunks}")
    print(f"Max chunks per doc: {max(chunks_per_doc.values()) if chunks_per_doc else 0}")
    print()
    print("Top 30 docs by chunk count:")
    for doc_id, c in chunks_per_doc.most_common(30):
        short = doc_id if len(doc_id) < 80 else "..." + doc_id[-77:]
        print(f"  {c:>6,}  {short}")

    # --- Section 4: span-value duplicate density ---
    print()
    print("=" * 80)
    print("SPAN-VALUE DUPLICATION DENSITY per (lang × cat)")
    print("=" * 80)
    print(f"\n{'lang':<8} {'cat':<22} {'n_spans':>10} {'n_unique':>10} {'top1_%':>8} {'top10_%':>8} {'top100_%':>10}")
    for la in LANGS_TARGET:
        for cat in CATS:
            items = sorted(by_lang_cat.get((la, cat), []), key=lambda x: -x[1])
            if not items:
                continue
            total = sum(c for _, c in items)
            unique = len(items)
            top1 = items[0][1] / total * 100
            top10 = sum(c for _, c in items[:10]) / total * 100
            top100 = sum(c for _, c in items[:100]) / total * 100
            print(f"{la:<8} {cat:<22} {total:>10,} {unique:>10,} {top1:>7.2f}% {top10:>7.2f}% {top100:>9.2f}%")


if __name__ == "__main__":
    main()
