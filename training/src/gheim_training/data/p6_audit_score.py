"""P6 audit step 3: Score the 4-way subagent labelings against the
LLM-labelled gold (val + test sample).

Inputs:
  data/p6_audit_sample.jsonl                  — gold (the dataset's labels for the 580 chunks)
  data/p6_audit_subagent_{1,2,3,4}.jsonl      — independent subagent labelings

Output: human-readable report on stdout (per-split, per-(lang × cat), overall).

Scoring identical to p2_score.py:
- Build majority vote per (chunk_id, normalized value, label): vote IN if
  ≥2 of 4 subagents flagged the same value-label pair on the same chunk.
- Normalization: strip whitespace + casefold.
- Token-level overlap is NOT computed — value-string match.
- F1 per (lang × cat) and overall, broken down by original_split (val vs test).

This is a quality estimate, NOT a relabeling. The subagents do not become
the new gold — we trust the dataset labels (recall-aggressive policy)
and use the F1 number to honestly characterise label noise on the
held-out portion.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

GOLD = Path("data/p6_audit_sample.jsonl")
SUBAGENT_PATHS = [Path(f"data/p6_audit_subagent_{i}.jsonl") for i in (1, 2, 3, 4)]
LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]
SPLITS = ["validation", "test"]


def _key(value: str, label: str) -> tuple[str, str]:
    return (value.strip().casefold(), label)


def _load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def f1(t: int, fp: int, fn: int) -> float | None:
    if t == 0 and fp == 0 and fn == 0:
        return None
    if (t + fp) == 0 or (t + fn) == 0:
        return 0.0
    p = t / (t + fp)
    r = t / (t + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def main() -> None:
    gold = _load_jsonl(GOLD)
    subs = [_load_jsonl(p) for p in SUBAGENT_PATHS]

    print(f"Gold (LLM-labelled): {len(gold)} chunks")
    for i, s in enumerate(subs, 1):
        print(f"  subagent_{i}: {len(s)} chunks")
    if any(len(s) != len(gold) for s in subs):
        print("WARNING: subagent line counts differ from gold")
    print()

    # cell counters keyed by (split, lang, cat)
    tp: dict[tuple[str, str, str], int] = Counter()
    fp: dict[tuple[str, str, str], int] = Counter()
    fn: dict[tuple[str, str, str], int] = Counter()
    gold_count: dict[tuple[str, str, str], int] = Counter()

    for cid, gold_rec in gold.items():
        lang = gold_rec.get("language", "")
        split = gold_rec.get("original_split", "?")
        gold_keys = {_key(sp["value"], sp["label"]) for sp in gold_rec.get("spans", [])}

        votes: Counter[tuple[str, str]] = Counter()
        for s in subs:
            sub_rec = s.get(cid, {"spans": []})
            sub_keys = {_key(sp["value"], sp["label"]) for sp in sub_rec.get("spans", [])}
            for k in sub_keys:
                votes[k] += 1
        majority_keys = {k for k, v in votes.items() if v >= 2}

        for k in gold_keys | majority_keys:
            cell = (split, lang, k[1])
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

    # ---------- Per (split × lang × cat) ----------
    for split in SPLITS:
        print("=" * 80)
        print(f"SPLIT = {split.upper()}")
        print("=" * 80)
        print(f"{'lang':<8} {'cat':<22} {'gold':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6}")
        for la in LANGS:
            for cat in CATS:
                cell = (split, la, cat)
                t, p, n = tp[cell], fp[cell], fn[cell]
                g = gold_count[cell]
                score = f1(t, p, n)
                if g == 0 and t == 0 and p == 0 and n == 0:
                    continue
                sd = f"{score:.3f}" if score is not None else "n/a"
                print(f"{la:<8} {cat:<22} {g:>5} {t:>4} {p:>4} {n:>4} {sd:>6}")

        # split totals
        s_tp = sum(v for k, v in tp.items() if k[0] == split)
        s_fp = sum(v for k, v in fp.items() if k[0] == split)
        s_fn = sum(v for k, v in fn.items() if k[0] == split)
        s_overall = f1(s_tp, s_fp, s_fn)
        print()
        print(f"  {split} overall: TP={s_tp} FP={s_fp} FN={s_fn} → F1={s_overall:.3f}")
        print()

    # ---------- Combined overall ----------
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    overall = f1(total_tp, total_fp, total_fn)
    print("=" * 80)
    print("COMBINED (val + test)")
    print("=" * 80)
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn} → F1={overall:.3f}")
    print()

    # Per-cat aggregate (across splits + langs) for comparison vs P2
    print(f"{'cat':<22} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6}")
    for cat in CATS:
        c_tp = sum(v for k, v in tp.items() if k[2] == cat)
        c_fp = sum(v for k, v in fp.items() if k[2] == cat)
        c_fn = sum(v for k, v in fn.items() if k[2] == cat)
        score = f1(c_tp, c_fp, c_fn)
        sd = f"{score:.3f}" if score is not None else "n/a"
        print(f"{cat:<22} {c_tp:>4} {c_fp:>4} {c_fn:>4} {sd:>6}")
    print()

    # P2 reference
    print("Reference: P2 audit (training-data sample, n=176) gave F1 = 0.664")


if __name__ == "__main__":
    main()
