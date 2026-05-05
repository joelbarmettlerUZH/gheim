"""Day 9 bake-off matrix: 3 trained models + 2 baselines × 3 eval sets.

Scores each detector on each eval set and writes one combined report to
``eval/bakeoff_matrix.json``. The leader is the trained model with the
highest overall overlap F1 on test_v1 — that's the bake-off gate per
CLAUDE.md:

  Gate: leader must beat best baseline (zero-shot priv-filter at 0.415
  overlap on test_v1) by >=3 F1 → target >= 0.445 overlap.

Detector × eval-set matrix is 5 × 3 = 15 evaluations. Each takes a few
minutes on one 4090; total ~30-45 minutes.

Run:
    uv run python -m gheim_training.eval.run_bakeoff_matrix \\
        --out eval/bakeoff_matrix.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .baselines.zero_shot import HFDetector
from .harness import _fmt_f1, _load_eval_jsonl, evaluate

EVAL_SETS: dict[str, Path] = {
    "test_v1": Path("data/test_v1.jsonl"),
    "audit_gold": Path("data/audit_gold.jsonl"),
    "address_gold_v1": Path("data/address_gold_v1.jsonl"),
}

# Detector factories. Lazy so we don't load every model at startup.
DETECTORS: dict[str, Callable[[], Any]] = {
    "bakeoff_xlmr": lambda: HFDetector("checkpoints/bakeoff-xlmr"),
    # Day 13 ablations: same xlmr config, dataset minus one layer.
    "ablation_no_l7": lambda: HFDetector("checkpoints/ablation_xlmr_no_l7"),
    "ablation_no_l8": lambda: HFDetector("checkpoints/ablation_xlmr_no_l8"),
    # Day 14 composite: regex front-end + bakeoff_xlmr on remainder.
    "composite_xlmr": lambda: _make_composite("checkpoints/bakeoff-xlmr"),
}


def _make_composite(model_path: str) -> Any:
    """Wrap composite around HFDetector (NOT LocalDetector — LocalDetector
    returns fragmented subword spans whereas HFDetector applies _merge_adjacent.
    Composite delegates to whatever detector is passed in; the bake-off
    matrix evaluates the leader through HFDetector, so the composite must
    use the same wrapper for the comparison to be apples-to-apples)."""
    from gheim.detectors.composite import CompositeDetector
    return CompositeDetector(model=HFDetector(model_path))


def _summary(rep: dict[str, Any]) -> dict[str, Any]:
    """Pull the headline numbers from a full report for the comparison table."""
    return {
        "n_cases": rep.get("n_cases"),
        "strict_overall_f1": rep["strict"]["overall"]["f1"],
        "overlap_overall_f1": rep["overlap"]["overall"]["f1"],
        "by_entity_overlap": {
            cat: rep["overlap"]["by_entity"][cat]["f1"]
            for cat in rep["overlap"]["by_entity"]
        },
        "by_language_overlap": {
            lang: rep["overlap"]["by_language"][lang]["f1"]
            for lang in rep["overlap"]["by_language"]
        },
        "by_presence": rep.get("by_presence"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("eval/bakeoff_matrix.json"))
    args = ap.parse_args()

    full: dict[str, dict[str, dict[str, Any]]] = {}
    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for det_name, factory in DETECTORS.items():
        print(f"\n=== Loading {det_name} ===")
        det = factory()
        full[det_name] = {}
        summary[det_name] = {}
        for set_name, set_path in EVAL_SETS.items():
            t0 = time.time()
            cases = _load_eval_jsonl(set_path)
            rep = evaluate(det, cases)
            elapsed = time.time() - t0
            full[det_name][set_name] = rep
            summary[det_name][set_name] = _summary(rep)
            print(
                f"  {set_name:<18} "
                f"strict={_fmt_f1(rep['strict']['overall']['f1'])} "
                f"overlap={_fmt_f1(rep['overlap']['overall']['f1'])} "
                f"({elapsed:.0f}s)"
            )

    # Headline comparison
    print("\n=== Bake-off matrix: overall overlap F1 ===")
    print(f"{'detector':<28}  " + "  ".join(f"{s:<16}" for s in EVAL_SETS))
    for det_name in DETECTORS:
        cells = [_fmt_f1(summary[det_name][s]["overlap_overall_f1"])
                 for s in EVAL_SETS]
        print(f"{det_name:<28}  " + "  ".join(f"{c:<16}" for c in cells))

    # FP rate on test_v1 controls
    print("\n=== test_v1 control-set FP/chunk (lower is better) ===")
    for det_name in DETECTORS:
        pres = summary[det_name]["test_v1"].get("by_presence")
        if pres and "control" in pres:
            ctrl = pres["control"]
            print(f"  {det_name:<28} {ctrl['fp_per_chunk']:.3f} "
                  f"({ctrl['fp']} FP / {ctrl['n_chunks']} chunks)")

    # Leader gate per CLAUDE.md — only run if both baselines are present
    print("\n=== Leader gate (CLAUDE.md Day 9): overlap on test_v1 vs best baseline ===")
    if {"zero_shot_privacy_filter", "swissner_plus_regex"} <= summary.keys():
        baseline = max(
            summary["zero_shot_privacy_filter"]["test_v1"]["overlap_overall_f1"],
            summary["swissner_plus_regex"]["test_v1"]["overlap_overall_f1"],
        )
        target = baseline + 0.03
        print(f"  Best baseline overlap: {baseline:.3f}; gate threshold: {target:.3f}")
        candidates = [d for d in summary
                      if d not in {"zero_shot_privacy_filter", "swissner_plus_regex"}]
        leader = max(candidates,
                     key=lambda d: summary[d]["test_v1"]["overlap_overall_f1"] or 0)
        leader_f1 = summary[leader]["test_v1"]["overlap_overall_f1"]
        margin = (leader_f1 or 0) - baseline
        if leader_f1 is not None and leader_f1 >= target:
            print(f"  ✓ Leader: {leader} at {leader_f1:.3f} (margin +{margin:.3f}). PASS.")
        else:
            print(f"  ✗ Leader: {leader} at {leader_f1}. Below threshold by "
                  f"{(target - (leader_f1 or 0)):.3f}. FAIL.")
    else:
        print("  baselines not in this run's DETECTORS; skip leader gate "
              "(it was already evaluated in eval/baselines.json + bakeoff_matrix.json).")
        leader = max(summary, key=lambda d: summary[d]["test_v1"]["overlap_overall_f1"] or 0)
        leader_f1 = summary[leader]["test_v1"]["overlap_overall_f1"]
        baseline = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_sets": {n: str(p) for n, p in EVAL_SETS.items()},
        "detectors": list(DETECTORS),
        "summary": summary,
        "full": full,
        "leader": leader,
        "leader_overlap_f1_test_v1": leader_f1,
        "best_baseline_overlap_f1_test_v1": baseline,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote → {args.out}")


if __name__ == "__main__":
    main()
