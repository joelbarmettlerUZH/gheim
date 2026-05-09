"""P2 step 3: Compute Gemma↔majority-subagent F1 from the 4 independent labelings.

Inputs:
  data/p2_sample.jsonl                   — gold (Gemma+regex spans)
  data/p2_subagent_{1,2,3,4}.jsonl       — independent subagent labelings

Output: human-readable report on stdout.

Scoring:
- Build majority vote per (chunk_id, normalized_value, label): vote is
  IN if ≥2 of 4 subagents flagged the same value-label pair on the same
  chunk. Normalization: strip whitespace, casefold for matching only.
- Token-level overlap is NOT computed — value-string match is the unit.
  This is robust to minor offset disagreements (subagents return values,
  not offsets).
- For Gemma+regex gold, use ALL spans on each chunk (regex-augmented
  layer5v4 spans).
- F1 per (lang × cat) and overall.

Gate:
  PASS  → F1 ≥ 0.95 overall AND every cell with ≥10 gold spans has F1 ≥ 0.90
  FAIL  → run Qwen second-opinion (P3)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

GOLD = Path("data/p2_sample.jsonl")
SUBAGENT_PATHS = [Path(f"data/p2_subagent_{i}.jsonl") for i in (1, 2, 3, 4)]
LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]

# Gate thresholds (per CLAUDE.md / user's call)
GATE_OVERALL = 0.95
GATE_CELL = 0.90
GATE_CELL_MIN_GOLD = 10  # cells with fewer than this many gold spans don't gate


def _key(value: str, label: str) -> tuple[str, str]:
    """Normalize a span for matching: strip + casefold the value."""
    return (value.strip().casefold(), label)


def _load_jsonl(path: Path) -> dict[str, dict]:
    """Load a JSONL file keyed by 'id'."""
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def main() -> None:
    gold = _load_jsonl(GOLD)
    subs = [_load_jsonl(p) for p in SUBAGENT_PATHS]

    print(f"Loaded gold: {len(gold)} chunks")
    for i, s in enumerate(subs, 1):
        print(f"  subagent_{i}: {len(s)} chunks")
    print()

    # Per-chunk: for each subagent, collect their span set.
    # majority: a (key, label) is "in" if ≥2 of 4 flagged it for that chunk.
    # gold_set: from p2_sample.jsonl, all spans regardless of source.
    tp = defaultdict(int)  # (lang, cat) -> count
    fp = defaultdict(int)
    fn = defaultdict(int)
    gold_count = defaultdict(int)
    pred_count = defaultdict(int)

    for cid, gold_rec in gold.items():
        lang = gold_rec.get("language", "")
        gold_keys: set[tuple[str, str]] = set()
        for sp in gold_rec.get("spans", []):
            gold_keys.add(_key(sp["value"], sp["label"]))

        # Subagent votes per (key, label)
        votes: Counter[tuple[str, str]] = Counter()
        for s in subs:
            sub_rec = s.get(cid, {"spans": []})
            sub_keys = {_key(sp["value"], sp["label"]) for sp in sub_rec.get("spans", [])}
            for k in sub_keys:
                votes[k] += 1
        majority_keys = {k for k, v in votes.items() if v >= 2}

        for k in gold_keys | majority_keys:
            cat = k[1]
            cell = (lang, cat)
            in_gold = k in gold_keys
            in_pred = k in majority_keys
            if in_gold and in_pred:
                tp[cell] += 1
            elif in_gold:
                fn[cell] += 1
            else:
                fp[cell] += 1
            if in_gold:
                gold_count[cell] += 1
            if in_pred:
                pred_count[cell] += 1

    # Compute F1
    def f1(t: int, f: int, n: int) -> float | None:
        if t == 0 and f == 0 and n == 0:
            return None
        if (t + f) == 0 or (t + n) == 0:
            return 0.0
        p = t / (t + f)
        r = t / (t + n)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    print("Per (lang × cat) F1 [gold_n / pred_n / TP / FP / FN / F1]:")
    print(f"{'lang':<8} {'cat':<22} {'gold':>5} {'pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6}")
    cell_f1: dict[tuple[str, str], float | None] = {}
    for la in LANGS:
        for cat in CATS:
            cell = (la, cat)
            t, f, n = tp[cell], fp[cell], fn[cell]
            g = gold_count[cell]
            p = pred_count[cell]
            score = f1(t, f, n)
            cell_f1[cell] = score
            sd = f"{score:.3f}" if score is not None else "n/a"
            print(f"{la:<8} {cat:<22} {g:>5} {p:>5} {t:>4} {f:>4} {n:>4} {sd:>6}")

    # Overall
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    overall = f1(total_tp, total_fp, total_fn)

    print()
    print(f"OVERALL: TP={total_tp} FP={total_fp} FN={total_fn} → F1={overall:.3f}")
    print()

    # Gate decision
    print("=" * 70)
    print("GATE")
    print("=" * 70)
    pass_overall = overall is not None and overall >= GATE_OVERALL
    print(f"  Overall F1 ≥ {GATE_OVERALL}: {'PASS' if pass_overall else 'FAIL'}  ({overall:.3f})")

    cell_fails = []
    for cell, score in cell_f1.items():
        if gold_count[cell] >= GATE_CELL_MIN_GOLD:
            if score is None or score < GATE_CELL:
                cell_fails.append((cell, score, gold_count[cell]))
    if not cell_fails:
        print(f"  All cells with ≥{GATE_CELL_MIN_GOLD} gold spans pass F1 ≥ {GATE_CELL}: PASS")
    else:
        print(f"  Cells with ≥{GATE_CELL_MIN_GOLD} gold spans BELOW F1 ≥ {GATE_CELL}: FAIL")
        for (la, cat), score, gc in cell_fails:
            sd = f"{score:.3f}" if score is not None else "n/a"
            print(f"    {la:<8} {cat:<22} F1={sd}  (gold_n={gc})")
    print()

    overall_pass = pass_overall and not cell_fails
    if overall_pass:
        print("→ GATE PASS — skip Qwen, proceed to P4 (data science)")
    else:
        print("→ GATE FAIL — run P3 (Qwen second-opinion pass over full corpus)")


if __name__ == "__main__":
    main()
