"""P7 audit step 2: score the dataset's labels against the four-model
OpenRouter consensus produced by ``p7_audit_openrouter.py``.

Same scoring contract as ``p6_audit_score.py`` (per-(chunk_id, casefold value,
label) match, majority of the four models = consensus). Writes both a
human-readable report on stdout and a machine-readable JSON for the paper.

Inputs
------
- ``data/p6_audit_sample.jsonl``        — gold (the dataset's own labels)
- ``data/p7_audit_<safe_slug>.jsonl``   — one per model, written by p7 step 1

Outputs
-------
- stdout: per-(split × lang × cat) breakdown + overall numbers
- ``data/p7_audit_report.json``: the same numbers in a parseable form so
  the paper / DataCard can quote them
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from gheim_training.data.p7_audit_openrouter import MODELS, _safe_slug

GOLD_PATH = Path("data/p6_audit_sample.jsonl")
REPORT_PATH = Path("data/p7_audit_report.json")
LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
]
SPLITS = ["validation", "test"]


def _key(value: str, label: str) -> tuple[str, str]:
    return (value.strip().casefold(), label)


def _load_jsonl_by_id(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _f1(tp: int, fp: int, fn: int) -> float | None:
    if tp == 0 and fp == 0 and fn == 0:
        return None
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def main() -> None:
    gold = _load_jsonl_by_id(GOLD_PATH)
    model_outputs: dict[str, dict[str, dict]] = {}
    for m in MODELS:
        path = Path(f"data/p7_audit_{_safe_slug(m)}.jsonl")
        if not path.exists():
            raise SystemExit(f"missing {path}")
        model_outputs[m] = _load_jsonl_by_id(path)

    print(f"Gold (dataset labels): {len(gold)} chunks")
    for m, recs in model_outputs.items():
        print(f"  {m}: {len(recs)} records")
    print()

    # cells keyed by (split, lang, cat)
    tp: Counter[tuple[str, str, str]] = Counter()
    fp: Counter[tuple[str, str, str]] = Counter()
    fn: Counter[tuple[str, str, str]] = Counter()
    gold_count: Counter[tuple[str, str, str]] = Counter()

    # Per-model F1 vs gold (independent agreement; not consensus)
    per_model_tp: Counter[str] = Counter()
    per_model_fp: Counter[str] = Counter()
    per_model_fn: Counter[str] = Counter()

    for cid, gold_rec in gold.items():
        lang = gold_rec.get("language", "")
        split = gold_rec.get("original_split", "?")
        gold_keys = {_key(sp["value"], sp["label"]) for sp in gold_rec.get("spans", [])}

        votes: Counter[tuple[str, str]] = Counter()
        for m in MODELS:
            rec = model_outputs[m].get(cid, {"spans": []})
            keys = {_key(sp["value"], sp["label"]) for sp in rec.get("spans", [])}
            for k in keys:
                votes[k] += 1
            # Per-model F1
            for k in gold_keys | keys:
                in_gold = k in gold_keys
                in_pred = k in keys
                if in_gold and in_pred:
                    per_model_tp[m] += 1
                elif in_gold:
                    per_model_fn[m] += 1
                else:
                    per_model_fp[m] += 1

        consensus = {k for k, v in votes.items() if v >= 2}
        for k in gold_keys | consensus:
            cell = (split, lang, k[1])
            in_gold = k in gold_keys
            in_pred = k in consensus
            if in_gold and in_pred:
                tp[cell] += 1
            elif in_gold:
                fn[cell] += 1
            else:
                fp[cell] += 1
            if in_gold:
                gold_count[cell] += 1

    # Per-(split × lang × cat) breakdown
    for split in SPLITS:
        print("=" * 80)
        print(f"SPLIT = {split.upper()}")
        print("=" * 80)
        print(
            f"{'lang':<8} {'cat':<22} {'gold':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6}"
        )
        for la in LANGS:
            for cat in CATS:
                cell = (split, la, cat)
                t, p, n = tp[cell], fp[cell], fn[cell]
                g = gold_count[cell]
                score = _f1(t, p, n)
                if g == 0 and t == 0 and p == 0 and n == 0:
                    continue
                sd = f"{score:.3f}" if score is not None else "n/a"
                print(f"{la:<8} {cat:<22} {g:>5} {t:>4} {p:>4} {n:>4} {sd:>6}")
        s_tp = sum(v for k, v in tp.items() if k[0] == split)
        s_fp = sum(v for k, v in fp.items() if k[0] == split)
        s_fn = sum(v for k, v in fn.items() if k[0] == split)
        s_overall = _f1(s_tp, s_fp, s_fn)
        print()
        print(
            f"  {split} overall: TP={s_tp} FP={s_fp} FN={s_fn} → "
            f"F1={s_overall:.3f}"
        )
        print()

    # Combined
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    overall = _f1(total_tp, total_fp, total_fn)
    print("=" * 80)
    print("COMBINED (val + test) vs majority-of-4 consensus")
    print("=" * 80)
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn} → F1={overall:.3f}")
    print()

    print(f"{'cat':<22} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6}")
    cat_table: dict[str, dict[str, float | int | None]] = {}
    for cat in CATS:
        c_tp = sum(v for k, v in tp.items() if k[2] == cat)
        c_fp = sum(v for k, v in fp.items() if k[2] == cat)
        c_fn = sum(v for k, v in fn.items() if k[2] == cat)
        score = _f1(c_tp, c_fp, c_fn)
        sd = f"{score:.3f}" if score is not None else "n/a"
        print(f"{cat:<22} {c_tp:>4} {c_fp:>4} {c_fn:>4} {sd:>6}")
        cat_table[cat] = {"tp": c_tp, "fp": c_fp, "fn": c_fn, "f1": score}
    print()

    # Per-model F1 vs gold (independent, not consensus)
    print("Per-model F1 vs dataset labels (independent, no voting):")
    print(f"  {'model':<32} {'TP':>5} {'FP':>5} {'FN':>5} {'F1':>6}")
    per_model_summary: dict[str, dict[str, float | int | None]] = {}
    for m in MODELS:
        t, p, n = per_model_tp[m], per_model_fp[m], per_model_fn[m]
        s = _f1(t, p, n)
        sd = f"{s:.3f}" if s is not None else "n/a"
        print(f"  {m:<32} {t:>5} {p:>5} {n:>5} {sd:>6}")
        per_model_summary[m] = {"tp": t, "fp": p, "fn": n, "f1": s}
    print()

    # Per-(split × lang × cat) JSON cells
    cells_json: dict[str, dict[str, dict[str, dict[str, int | float | None]]]] = {
        s: {la: {} for la in LANGS} for s in SPLITS
    }
    for (s, la, cat), t in tp.items():
        cells_json[s][la][cat] = {
            "tp": t,
            "fp": fp[(s, la, cat)],
            "fn": fn[(s, la, cat)],
            "f1": _f1(t, fp[(s, la, cat)], fn[(s, la, cat)]),
        }

    report = {
        "_meta": {
            "models": MODELS,
            "n_chunks": len(gold),
            "consensus_rule": "≥2 of 4 models flagged the (value, label) pair",
            "scoring": "value-string equality (casefold) + label match",
        },
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "f1": overall,
        },
        "per_category": cat_table,
        "per_model_vs_dataset": per_model_summary,
        "cells": cells_json,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
