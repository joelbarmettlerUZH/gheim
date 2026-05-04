"""Day 2 baselines driver: zero-shot privacy-filter + swissner+regex.

Runs both baseline detectors against both trend-line eval sets
(``audit_gold.jsonl``, ``address_gold_v1.jsonl``) and writes one
combined report to ``eval/baselines.json``. These are the floor numbers
the bake-off must beat (+3 F1 vs swissner+regex on test_v1 is the leader
gate per CLAUDE.md Week 2).

Usage:
    uv run python -m gheim_training.eval.run_baselines \\
        --out eval/baselines.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .baselines.swissbert_ner import SwissBertNERDetector
from .baselines.zero_shot import HFDetector
from .harness import _fmt_f1, _load_eval_jsonl, evaluate

EVAL_SETS: dict[str, Path] = {
    "audit_gold": Path("data/audit_gold.jsonl"),
    "address_gold_v1": Path("data/address_gold_v1.jsonl"),
    "test_v1": Path("data/test_v1.jsonl"),
}


def _summary_row(rep: dict[str, Any]) -> dict[str, Any]:
    """Pull just the headline numbers from a full evaluate() report for the
    comparison table — overall F1 strict + overlap, and per-entity overlap."""
    overall_strict = rep["strict"]["overall"]
    overall_overlap = rep["overlap"]["overall"]
    by_ent = {
        cat: {
            "strict_f1": rep["strict"]["by_entity"].get(cat, {}).get("f1"),
            "overlap_f1": rep["overlap"]["by_entity"].get(cat, {}).get("f1"),
        }
        for cat in sorted(rep["strict"]["by_entity"].keys() | rep["overlap"]["by_entity"].keys())
    }
    return {
        "n_cases": rep.get("n_cases"),
        "strict_overall_f1": overall_strict["f1"],
        "overlap_overall_f1": overall_overlap["f1"],
        "by_entity": by_ent,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("eval/baselines.json"))
    ap.add_argument("--privacy-filter-id", default="openai/privacy-filter")
    args = ap.parse_args()

    print(f"=== Day 2 baselines run @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    for name, p in EVAL_SETS.items():
        if not p.exists():
            raise FileNotFoundError(f"eval set {name!r} missing: {p}")
        print(f"  {name:<18} → {p} ({sum(1 for _ in p.open()):,} chunks)")
    print()

    # Detectors are loaded lazily on first .detect() call. Sharing instances
    # across eval sets reuses the loaded model weights.
    detectors: dict[str, Any] = {
        "zero_shot_privacy_filter": HFDetector(args.privacy_filter_id),
        "swissner_plus_regex": SwissBertNERDetector(),
    }

    full: dict[str, dict[str, dict[str, Any]]] = {}
    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for det_name, det in detectors.items():
        print(f"\n=== Running {det_name} ===")
        full[det_name] = {}
        summary[det_name] = {}
        for set_name, set_path in EVAL_SETS.items():
            t0 = time.time()
            cases = _load_eval_jsonl(set_path)
            rep = evaluate(det, cases)
            elapsed = time.time() - t0
            full[det_name][set_name] = rep
            summary[det_name][set_name] = _summary_row(rep)
            print(
                f"  {set_name:<18} "
                f"strict={_fmt_f1(rep['strict']['overall']['f1'])} "
                f"overlap={_fmt_f1(rep['overlap']['overall']['f1'])} "
                f"({elapsed:.1f}s)"
            )

    # Pretty comparison table
    print("\n=== Comparison: overall F1 ===")
    print(f"{'detector':<28}  {'eval set':<18}  {'strict':>7}  {'overlap':>7}")
    for det_name in detectors:
        for set_name in EVAL_SETS:
            s = summary[det_name][set_name]
            print(
                f"{det_name:<28}  {set_name:<18}  "
                f"{_fmt_f1(s['strict_overall_f1']):>7}  "
                f"{_fmt_f1(s['overlap_overall_f1']):>7}"
            )

    # Per-entity (overlap) on audit_gold — the eval set with the broadest
    # category coverage — so we can see where each detector is strong/weak.
    print("\n=== Per-entity overlap F1 on audit_gold ===")
    cats = sorted({
        cat
        for det_name in detectors
        for cat in summary[det_name]["audit_gold"]["by_entity"]
    })
    print(f"{'entity':<22} " + "  ".join(f"{n[:24]:>26}" for n in detectors))
    for cat in cats:
        row = []
        for det_name in detectors:
            v = summary[det_name]["audit_gold"]["by_entity"].get(cat, {}).get("overlap_f1")
            row.append(f"{_fmt_f1(v):>26}")
        print(f"{cat:<22} " + "  ".join(row))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_sets": {n: str(p) for n, p in EVAL_SETS.items()},
        "detectors": {
            "zero_shot_privacy_filter": {"model_id": args.privacy_filter_id},
            "swissner_plus_regex": {"model_id": "ZurichNLP/swissbert-ner"},
        },
        "summary": summary,
        "full": full,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote → {args.out}")


if __name__ == "__main__":
    main()
