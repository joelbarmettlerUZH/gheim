"""V2-8: frequency analysis on the merged v2 span universe.

Reads ``data/v2_assembled.jsonl`` and reports the distributions
that drive the V2-9 balancer:

1. **Span counts per (language × category × confidence-tier)**.
   Confidence tiers:
     - ``high``    = confidence ≥ 0.75 (3+ labellers agreed)
     - ``mid``     = 0.5 ≤ confidence < 0.75 (2 labellers agreed)
     - ``single``  = confidence < 0.5 (1 labeller only)
2. **Signal-strength distribution**: histogram of how many spans have
   1/2/3/4 signals (regex always counts; audit excluded from the
   denominator so the max is the number of labellers that ran).
3. **Per-labeller coverage matrix**: which labellers cover which
   categories. Helps spot categories where one labeller dominates
   (e.g. regex for account_number).
4. **Multi-PII-per-chunk rate**: percentage of chunks with ≥2 distinct
   categories or ≥2 spans. Tracks the v1 finding that 69.5% of
   chunks were mono-category (drove the model's suppression bug).
5. **Per-chunk negatives**: percentage with zero spans, broken down
   by language.

Output
------
- stdout: human-readable tables for sanity-checking
- ``data/v2_freq_report.json``: machine-readable input for the V2-9
  balancer

Run
---
    uv run python -m gheim_training.data.v2.freq_analysis
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .schema import V2Example, V2_SIGNALS

DEFAULT_IN = Path("data/v2_assembled.jsonl")
DEFAULT_OUT = Path("data/v2_freq_report.json")
LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]


def _confidence_tier(conf: float) -> str:
    if conf >= 0.75:
        return "high"
    if conf >= 0.5:
        return "mid"
    return "single"


def main() -> None:
    print(f"Reading {DEFAULT_IN}…", flush=True)

    cell: dict[tuple[str, str, str], int] = Counter()
    n_signals_dist: Counter[int] = Counter()
    per_labeller_per_cat: dict[str, Counter[str]] = {
        s: Counter() for s in V2_SIGNALS
    }
    n_chunks = 0
    n_negative_per_lang: Counter[str] = Counter()
    chunks_per_lang: Counter[str] = Counter()
    n_chunks_multi_cat = 0
    n_chunks_multi_span = 0
    n_chunks_with_any_pii = 0
    spans_per_chunk_dist: Counter[int] = Counter()

    with DEFAULT_IN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = V2Example.from_json(line)
            n_chunks += 1
            chunks_per_lang[ex.language] += 1
            spans = ex.spans
            spans_per_chunk_dist[min(len(spans), 10)] += 1
            if not spans:
                n_negative_per_lang[ex.language] += 1
                continue
            n_chunks_with_any_pii += 1
            if len(spans) >= 2:
                n_chunks_multi_span += 1
            if len({sp.label for sp in spans}) >= 2:
                n_chunks_multi_cat += 1
            for sp in spans:
                tier = _confidence_tier(sp.confidence)
                cell[(ex.language, sp.label, tier)] += 1
                non_audit = [s for s in sp.signals if s != "audit"]
                n_signals_dist[len(non_audit)] += 1
                for s in sp.signals:
                    per_labeller_per_cat[s][sp.label] += 1

            if n_chunks % 100_000 == 0:
                print(f"  read {n_chunks:,} chunks", flush=True)

    print(f"\nTotal chunks: {n_chunks:,}")
    print(f"  chunks with no PII: {sum(n_negative_per_lang.values()):,} "
          f"({100*sum(n_negative_per_lang.values())/n_chunks:.1f}%)")
    print(f"  chunks with ≥2 spans: {n_chunks_multi_span:,} "
          f"({100*n_chunks_multi_span/n_chunks:.1f}%)")
    print(f"  chunks with ≥2 categories: {n_chunks_multi_cat:,} "
          f"({100*n_chunks_multi_cat/n_chunks:.1f}%) "
          f"[v1 was 14.4%]")
    print()

    print("Spans-per-chunk distribution:")
    for k in sorted(spans_per_chunk_dist):
        n = spans_per_chunk_dist[k]
        label = f"{k}+" if k == 10 else str(k)
        print(f"  {label:>3}: {n:>9,} ({100*n/n_chunks:5.1f}%)")
    print()

    print("Signal-strength histogram (non-audit signals per span):")
    total_spans = sum(n_signals_dist.values())
    for k in sorted(n_signals_dist):
        n = n_signals_dist[k]
        print(f"  {k} signals: {n:>9,} ({100*n/total_spans:5.1f}%)")
    print(f"  TOTAL spans: {total_spans:,}")
    print()

    print("Per-(language × category × confidence) counts:")
    print(f"  {'lang':<8} {'cat':<22} {'high':>10} {'mid':>10} {'single':>10}")
    for la in LANGS:
        for cat in CATS:
            h = cell[(la, cat, "high")]
            m = cell[(la, cat, "mid")]
            s = cell[(la, cat, "single")]
            if h + m + s == 0:
                continue
            print(f"  {la:<8} {cat:<22} {h:>10,} {m:>10,} {s:>10,}")
    print()

    print("Per-labeller coverage matrix (spans flagged by labeller × category):")
    print(f"  {'labeller':<12} " + " ".join(f"{c:<20}" for c in CATS))
    for lbl in V2_SIGNALS:
        if lbl == "audit":
            continue  # audit not present on the main corpus
        row = [f"{per_labeller_per_cat[lbl][c]:<20,}" for c in CATS]
        print(f"  {lbl:<12} " + " ".join(row))
    print()

    print("Per-language: chunks + negative rate:")
    for la in sorted(chunks_per_lang, key=lambda l: -chunks_per_lang[l]):
        n = chunks_per_lang[la]
        neg = n_negative_per_lang[la]
        print(f"  {la:<8} chunks={n:>8,}  negatives={neg:>7,} "
              f"({100*neg/n:5.1f}%)")
    print()

    report = {
        "n_chunks": n_chunks,
        "n_chunks_with_any_pii": n_chunks_with_any_pii,
        "n_chunks_multi_cat": n_chunks_multi_cat,
        "n_chunks_multi_span": n_chunks_multi_span,
        "spans_per_chunk_dist": {str(k): v for k, v in spans_per_chunk_dist.items()},
        "signal_strength_dist": {str(k): v for k, v in n_signals_dist.items()},
        "total_spans": total_spans,
        "per_cell": {
            f"{la}__{cat}__{tier}": cell[(la, cat, tier)]
            for la in LANGS for cat in CATS
            for tier in ("high", "mid", "single")
            if cell[(la, cat, tier)] > 0
        },
        "per_labeller_per_cat": {
            lbl: dict(per_labeller_per_cat[lbl])
            for lbl in V2_SIGNALS if lbl != "audit"
        },
        "per_language": {
            la: {"chunks": chunks_per_lang[la], "negatives": n_negative_per_lang[la]}
            for la in chunks_per_lang
        },
    }
    DEFAULT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
