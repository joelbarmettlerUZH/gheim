"""P8: Profile the REMAPped AI4Privacy openpii-1m data per (lang × cat).

Goal: decide what to mix into the training set on top of gheim-pii.

Reads:
  data/layer2_v2.jsonl   — openpii-1m de/fr/it (already REMAPped to our 8 cats)
  data/layer4_v2.jsonl   — openpii-1m en       (already REMAPped to our 8 cats)

Reports:
  1. Per (lang × cat) chunk count + span count + n_unique values + top1_%
  2. Region distribution per language (CH/DE/AT/IT/FR/...)
  3. Side-by-side comparison vs gheim-pii train cell counts
  4. Recommended targeted top-up sizes per cell

Source files are read-only; no writes.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

INPUTS_AI4P = [
    Path("data/layer2_v2.jsonl"),
    Path("data/layer4_v2.jsonl"),
]
GHEIM_TRAIN = Path("data/gheim_pii_balanced_train.jsonl")

LANGS_ORDER = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]


def normalize_value(value: str, label: str) -> str:
    v = value.strip()
    if label in ("private_person", "private_address"):
        return v.casefold()
    return v


def _profile(paths: list[Path], label: str) -> tuple[
    Counter, Counter, dict, dict, Counter, int
]:
    """Return per-cell stats for the given dataset.

    Returns:
        cell_chunks: (lang, cat) → n_chunks_with_this_cat
        cell_spans:  (lang, cat) → n_spans
        cell_values: (lang, cat) → Counter(value → freq)
        region_per_lang: lang → Counter(region → n_chunks)
        lang_chunks: lang → n_chunks
        n_total
    """
    cell_chunks: Counter[tuple[str, str]] = Counter()
    cell_spans: Counter[tuple[str, str]] = Counter()
    cell_values: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    region_per_lang: dict[str, Counter[str]] = defaultdict(Counter)
    lang_chunks: Counter[str] = Counter()
    n_total = 0

    for path in paths:
        if not path.exists():
            print(f"WARNING: missing {path}, skipping")
            continue
        print(f"  Reading: {path}", flush=True)
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                n_total += 1
                lang = rec.get("language", "")
                lang_chunks[lang] += 1
                region = (rec.get("meta") or {}).get("region", "?") or "?"
                region_per_lang[lang][region] += 1

                spans = rec.get("spans", [])
                cats_in_chunk = set()
                text = rec.get("text", "")
                for sp in spans:
                    cat = sp.get("label", "")
                    if cat not in CATS:
                        continue
                    cell_spans[(lang, cat)] += 1
                    val_str = text[sp["start"]:sp["end"]]
                    cell_values[(lang, cat)][normalize_value(val_str, cat)] += 1
                    cats_in_chunk.add(cat)
                for cat in cats_in_chunk:
                    cell_chunks[(lang, cat)] += 1
                if n_total % 50_000 == 0:
                    print(f"    scanned {n_total:,}", flush=True)
    return cell_chunks, cell_spans, cell_values, region_per_lang, lang_chunks, n_total


def _profile_gheim(path: Path) -> tuple[Counter, Counter]:
    """Return (cell_chunks, lang_chunks) for the gheim-pii train file."""
    cell_chunks: Counter[tuple[str, str]] = Counter()
    lang_chunks: Counter[str] = Counter()
    print(f"  Reading: {path}", flush=True)
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            lang = rec.get("language", "")
            lang_chunks[lang] += 1
            cats = {sp["label"] for sp in rec.get("spans", []) if sp.get("label") in CATS}
            for cat in cats:
                cell_chunks[(lang, cat)] += 1
    return cell_chunks, lang_chunks


def main() -> None:
    print("=" * 80)
    print("Profiling AI4Privacy openpii-1m (REMAPped layers)")
    print("=" * 80)
    a4p_chunks, a4p_spans, a4p_values, a4p_regions, a4p_lang, a4p_total = _profile(
        INPUTS_AI4P, "AI4Privacy openpii-1m"
    )
    print(f"\nAI4Privacy total chunks: {a4p_total:,}")

    print()
    print("=" * 80)
    print("Profiling gheim-pii train")
    print("=" * 80)
    gheim_cells, gheim_lang = _profile_gheim(GHEIM_TRAIN)

    # ---------- Per-language summary ----------
    print()
    print("=" * 80)
    print("PER-LANGUAGE TOTALS")
    print("=" * 80)
    print(f"{'lang':<8} {'gheim_train':>12} {'ai4p':>10}")
    for lang in LANGS_ORDER:
        print(f"{lang:<8} {gheim_lang.get(lang, 0):>12,} {a4p_lang.get(lang, 0):>10,}")

    # ---------- Region distribution per AI4Privacy lang ----------
    print()
    print("=" * 80)
    print("AI4PRIVACY REGION DISTRIBUTION (top 8 regions per language)")
    print("=" * 80)
    for lang in LANGS_ORDER:
        regions = a4p_regions.get(lang, Counter())
        if not regions:
            continue
        total = sum(regions.values())
        print(f"\n{lang} — total {total:,} chunks")
        for region, n in regions.most_common(8):
            print(f"  {region:<6} {n:>8,}  ({100*n/total:>5.1f}%)")

    # ---------- Cell-by-cell comparison ----------
    print()
    print("=" * 80)
    print("CELL-BY-CELL: gheim-pii train vs AI4Privacy availability")
    print("=" * 80)
    print(f"{'lang':<8} {'cat':<22} {'gheim_train':>12} {'a4p_chunks':>12} "
          f"{'a4p_spans':>10} {'a4p_unique':>10} {'top1_%':>7}")
    for lang in LANGS_ORDER:
        for cat in CATS:
            cell = (lang, cat)
            g = gheim_cells.get(cell, 0)
            a_chunks = a4p_chunks.get(cell, 0)
            a_spans = a4p_spans.get(cell, 0)
            vals = a4p_values.get(cell, Counter())
            a_unique = len(vals)
            top1 = (vals.most_common(1)[0][1] / sum(vals.values()) * 100) if vals else 0.0
            if g == 0 and a_chunks == 0:
                continue
            print(f"{lang:<8} {cat:<22} {g:>12,} {a_chunks:>12,} "
                  f"{a_spans:>10,} {a_unique:>10,} {top1:>6.2f}%")

    # ---------- Where AI4Privacy adds value: cells where AI4Privacy >> gheim ----------
    print()
    print("=" * 80)
    print("CELLS WHERE AI4PRIVACY CAN USEFULLY SUPPLEMENT (a4p ≥ 2× gheim)")
    print("=" * 80)
    print("Sorted by gheim cell sparsity (smallest cells first).")
    print(f"\n{'lang':<8} {'cat':<22} {'gheim':>10} {'a4p':>10} {'ratio':>8}")
    candidates = []
    for lang in LANGS_ORDER:
        for cat in CATS:
            cell = (lang, cat)
            g = gheim_cells.get(cell, 0)
            a = a4p_chunks.get(cell, 0)
            if a == 0:
                continue
            if g == 0 or a >= 2 * g:
                ratio = a / g if g > 0 else float("inf")
                candidates.append((g, lang, cat, a, ratio))
    candidates.sort()
    for g, lang, cat, a, ratio in candidates:
        ratio_s = f"{ratio:.1f}×" if ratio != float("inf") else "∞"
        print(f"{lang:<8} {cat:<22} {g:>10,} {a:>10,} {ratio_s:>8}")

    # ---------- Top values per cell (AI4Privacy) — synthetic-Faker check ----------
    print()
    print("=" * 80)
    print("AI4PRIVACY: top-5 values per cell (Faker dominance check)")
    print("=" * 80)
    for lang in LANGS_ORDER:
        for cat in CATS:
            vals = a4p_values.get((lang, cat), Counter())
            if not vals:
                continue
            total = sum(vals.values())
            top5 = vals.most_common(5)
            top5_share = sum(c for _, c in top5) / total * 100
            print(f"\n--- {lang} × {cat} ({total:,} spans, {len(vals):,} unique, "
                  f"top-5 = {top5_share:.1f}%) ---")
            for v, c in top5:
                short = v if len(v) < 60 else v[:57] + "..."
                print(f"  {c:>6,} ({100*c/total:>5.2f}%)  {short}")


if __name__ == "__main__":
    main()
