"""P4b: Pre-balance instrumentation pass — span density, negative ratio,
placeholder spans, and per-(top-doc × cat) contribution.

Read-only scan over the same two inputs as p4_freq. Produces the four
measurements needed to set defensible caps in P5:

  1. Span density per chunk — histogram of len(spans) per chunk, by
     (lang × cat-set-size). Tells us whether dense chunks need
     down-weighting beyond the per-doc cap.
  2. Negative ratio — count of chunks with len(spans)==0 by language
     and subset. Tells us whether we have ~15% negatives natively or
     need to mine more.
  3. Placeholder span values — count spans whose value matches the
     literal patterns <*-pii>, <pii-*>, [REDACTED], etc. Quantifies
     the noise the placeholder filter will remove.
  4. Top-30 docs × cat matrix — for the heaviest docs (by chunk count),
     which categories do they contribute? Tells us whether per-doc
     capping disproportionately hurts certain cells.

Inputs:  same as p4_freq (data/layer5v4_lang_fix.jsonl + regex_new)
Output:  human-readable report on stdout.
"""
from __future__ import annotations

import json
import re
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

# Patterns that almost certainly indicate label-noise / placeholder leakage.
# Conservative: must look like a structured tag, not just contain the substring.
PLACEHOLDER_PATTERNS = [
    re.compile(r"^<[a-z_-]*pii[a-z_-]*>$", re.IGNORECASE),       # <email-pii>, <iban-pii>, <pii-name>
    re.compile(r"^\[REDACTED[A-Z0-9_]*\]$"),                      # [REDACTED], [REDACTED_NAME]
    re.compile(r"^_+[A-Za-z]+_+$"),                                # ____Name____
    re.compile(r"^[A-Z]_+@[a-z.]+$"),                              # CH______@gmail.com (AI4Privacy artifact)
    re.compile(r"^X+$", re.IGNORECASE),                            # XXXXXX
]

# Numeric-degenerate patterns (likely false-positive account/phone).
NUMERIC_DEGENERATE_PATTERNS = [
    re.compile(r"^[0\s]+$"),                                       # 0000 0000 0000 0000
    re.compile(r"^(\d)(\s+\1)+$"),                                 # 21 21 21 21
]


def _is_placeholder(value: str) -> bool:
    v = value.strip()
    return any(p.match(v) for p in PLACEHOLDER_PATTERNS)


def _is_numeric_degenerate(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    for p in NUMERIC_DEGENERATE_PATTERNS:
        if p.match(v):
            return True
    # Numeric run with very low distinct-digit count (e.g. "21 21 21 70 70 70 21 21")
    digits = [c for c in v if c.isdigit()]
    if len(digits) >= 8 and len(set(digits)) <= 3 and re.match(r"^[\d\s]+$", v):
        return True
    return False


def _density_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    return "21+"


DENSITY_BUCKETS_ORDER = ["0", "1", "2-3", "4-5", "6-10", "11-20", "21+"]


def main() -> None:
    # --- aggregations ---
    # 1) span density
    density_per_lang: dict[str, Counter[str]] = defaultdict(Counter)
    span_count_total = 0
    chunk_count_total = 0

    # 2) negatives by lang × subset
    neg_by_lang_subset: Counter[tuple[str, str]] = Counter()
    pos_by_lang_subset: Counter[tuple[str, str]] = Counter()

    # 3) placeholder + degenerate frequencies
    placeholder_by_cat: Counter[tuple[str, str, str]] = Counter()  # (lang, cat, value)
    degenerate_by_cat: Counter[tuple[str, str, str]] = Counter()
    placeholder_total = 0
    degenerate_total = 0

    # 4) doc → cat contribution. Track only docs that exceed a threshold.
    chunks_per_doc: Counter[str] = Counter()
    doc_cat_chunks: dict[str, Counter[str]] = defaultdict(Counter)  # doc_id → Counter(cat)
    doc_lang: dict[str, str] = {}

    for path in INPUTS:
        if not path.exists():
            print(f"WARNING: missing {path}, skipping")
            continue
        print(f"Reading: {path}", flush=True)
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                chunk_count_total += 1
                lang = rec.get("language", "")
                subset = rec.get("subset", "unknown") or "unknown"
                doc_id = rec.get("doc_id", "")
                spans = rec.get("spans", [])
                n_spans = len(spans)
                span_count_total += n_spans

                density_per_lang[lang][_density_bucket(n_spans)] += 1
                if n_spans == 0:
                    neg_by_lang_subset[(lang, subset)] += 1
                else:
                    pos_by_lang_subset[(lang, subset)] += 1

                if doc_id:
                    chunks_per_doc[doc_id] += 1
                    doc_lang[doc_id] = lang
                    cats_in_chunk = {sp["label"] for sp in spans}
                    for c in cats_in_chunk:
                        doc_cat_chunks[doc_id][c] += 1

                for sp in spans:
                    val = sp.get("value", "")
                    label = sp.get("label", "")
                    if _is_placeholder(val):
                        placeholder_by_cat[(lang, label, val.strip())] += 1
                        placeholder_total += 1
                    elif _is_numeric_degenerate(val):
                        degenerate_by_cat[(lang, label, val.strip())] += 1
                        degenerate_total += 1

                if chunk_count_total % 500_000 == 0:
                    print(f"  scanned {chunk_count_total:,}", flush=True)

    print()
    print(f"TOTAL chunks: {chunk_count_total:,}")
    print(f"TOTAL spans:  {span_count_total:,}")
    print()

    # ---------- Section 1: span density ----------
    print("=" * 80)
    print("SPAN DENSITY per chunk, by language")
    print("=" * 80)
    header = f"{'lang':<10} " + " ".join(f"{b:>9}" for b in DENSITY_BUCKETS_ORDER) + f"  {'total':>10}"
    print(header)
    for lang in LANGS_TARGET:
        row = density_per_lang.get(lang, Counter())
        total = sum(row.values())
        if total == 0:
            continue
        cells = " ".join(f"{row.get(b, 0):>9,}" for b in DENSITY_BUCKETS_ORDER)
        print(f"{lang:<10} {cells}  {total:>10,}")
    # Aggregate across non-target langs
    other = Counter()
    for lang, row in density_per_lang.items():
        if lang in LANGS_TARGET:
            continue
        other.update(row)
    if sum(other.values()) > 0:
        cells = " ".join(f"{other.get(b, 0):>9,}" for b in DENSITY_BUCKETS_ORDER)
        print(f"{'(other)':<10} {cells}  {sum(other.values()):>10,}")
    print()
    # Percent split for target langs
    print("Same as percentages (target langs only):")
    print(header)
    for lang in LANGS_TARGET:
        row = density_per_lang.get(lang, Counter())
        total = sum(row.values())
        if total == 0:
            continue
        cells = " ".join(f"{100*row.get(b, 0)/total:>8.2f}%" for b in DENSITY_BUCKETS_ORDER)
        print(f"{lang:<10} {cells}  {total:>10,}")

    # ---------- Section 2: negatives by lang × subset ----------
    print()
    print("=" * 80)
    print("NEGATIVE RATIO by (lang × subset)")
    print("=" * 80)
    subsets = sorted({s for _, s in list(neg_by_lang_subset.keys()) + list(pos_by_lang_subset.keys())})
    print(f"{'lang':<10} {'subset':<22} {'neg':>10} {'pos':>10} {'total':>10} {'neg_%':>8}")
    for lang in LANGS_TARGET:
        for s in subsets:
            n = neg_by_lang_subset.get((lang, s), 0)
            p = pos_by_lang_subset.get((lang, s), 0)
            t = n + p
            if t == 0:
                continue
            print(f"{lang:<10} {s:<22} {n:>10,} {p:>10,} {t:>10,} {100*n/t:>7.2f}%")
    # Per-lang totals
    print()
    print("Per-language totals:")
    print(f"{'lang':<10} {'neg':>10} {'pos':>10} {'total':>10} {'neg_%':>8}")
    for lang in LANGS_TARGET:
        n = sum(c for (la, _), c in neg_by_lang_subset.items() if la == lang)
        p = sum(c for (la, _), c in pos_by_lang_subset.items() if la == lang)
        t = n + p
        if t == 0:
            continue
        print(f"{lang:<10} {n:>10,} {p:>10,} {t:>10,} {100*n/t:>7.2f}%")
    # Overall (target-lang)
    n_all = sum(c for (la, _), c in neg_by_lang_subset.items() if la in LANGS_TARGET)
    p_all = sum(c for (la, _), c in pos_by_lang_subset.items() if la in LANGS_TARGET)
    t_all = n_all + p_all
    if t_all > 0:
        print(f"{'OVERALL':<10} {n_all:>10,} {p_all:>10,} {t_all:>10,} {100*n_all/t_all:>7.2f}%")

    # ---------- Section 3: placeholder + degenerate spans ----------
    print()
    print("=" * 80)
    print("PLACEHOLDER + DEGENERATE SPAN VALUES")
    print("=" * 80)
    print(f"Total placeholder spans:  {placeholder_total:,}  ({100*placeholder_total/max(span_count_total,1):.3f}% of all spans)")
    print(f"Total degenerate spans:   {degenerate_total:,}  ({100*degenerate_total/max(span_count_total,1):.3f}% of all spans)")
    print()
    if placeholder_total > 0:
        print("Top 30 placeholder values (lang × cat × value → count):")
        for (la, cat, val), c in Counter(placeholder_by_cat).most_common(30):
            short = val if len(val) < 50 else val[:47] + "..."
            print(f"  {c:>8,}  {la:<8} {cat:<22} {short}")
    print()
    if degenerate_total > 0:
        print("Top 30 numeric-degenerate values:")
        for (la, cat, val), c in Counter(degenerate_by_cat).most_common(30):
            short = val if len(val) < 50 else val[:47] + "..."
            print(f"  {c:>8,}  {la:<8} {cat:<22} {short}")

    # ---------- Section 4: top-30 docs × cat ----------
    print()
    print("=" * 80)
    print("TOP-30 DOCS × CAT CONTRIBUTION (chunks of that cat from this doc)")
    print("=" * 80)
    cap_30 = 30
    cap_50 = 50
    impact_30 = 0
    impact_50 = 0
    for doc_id, total in chunks_per_doc.items():
        if total > cap_30:
            impact_30 += (total - cap_30)
        if total > cap_50:
            impact_50 += (total - cap_50)
    print(f"Per-doc cap impact:")
    print(f"  cap=30 → would remove {impact_30:,} chunks across {sum(1 for c in chunks_per_doc.values() if c > 30):,} docs")
    print(f"  cap=50 → would remove {impact_50:,} chunks across {sum(1 for c in chunks_per_doc.values() if c > 50):,} docs")
    print()
    print(f"{'chunks':>8} {'lang':<8} {'doc_id':<60}  {'cats present (chunks of cat)'}")
    for doc_id, total in chunks_per_doc.most_common(30):
        la = doc_lang.get(doc_id, "?")
        cat_summary = ", ".join(f"{c}:{n}" for c, n in doc_cat_chunks[doc_id].most_common())
        short = doc_id if len(doc_id) < 60 else "..." + doc_id[-57:]
        print(f"{total:>8,} {la:<8} {short}  {cat_summary[:120]}")


if __name__ == "__main__":
    main()
