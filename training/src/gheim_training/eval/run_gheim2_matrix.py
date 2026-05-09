"""gheim-2 bake-off evaluation matrix.

Scores the 3 gold-core checkpoints (priv / xlmr / swissbert) against:
  - test_v1            — 800 chunks of held-out Swiss text (same as gheim-1)
  - test_en            — held-out English from openpii-1m validation, with
                          hash-based exclusion vs Layer 4.v2 training data
  - audit_gold         — gheim-1 trend line
  - address_gold_v1    — address-targeted trend line

Plus baselines for context:
  - zero_shot_privacy_filter (the gheim-1 base)
  - swissner_plus_regex      (best non-trained baseline)

Tie-break for leader (per user direction):
  1. F1 within 0.02 → smaller params win
  2. → Swiss-specific origin wins

So if SwissBERT (270M, ZurichNLP) comes within 0.02 of XLM-R (550M),
SwissBERT wins. If only XLM-R is within 0.02 of priv (1.4B MoE), XLM-R
wins. The script prints the winner explicitly.

Run:
    uv run python -m gheim_training.eval.run_gheim2_matrix \\
        --out eval/gheim2_matrix.json
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
    "test_en": Path("data/test_en.jsonl"),
    "audit_gold": Path("data/audit_gold.jsonl"),
    "address_gold_v1": Path("data/address_gold_v1.jsonl"),
}

def _make_swissner() -> Any:
    from .baselines.swissbert_ner import SwissBertNERDetector
    return SwissBertNERDetector()


# (name, factory, params_M, swiss_specific). Used for tie-break.
DetSpec = tuple[Callable[[], Any], int, bool]
DETECTORS: dict[str, DetSpec] = {
    "gheim2_priv":      (lambda: HFDetector("checkpoints/gheim2-priv"),       1400, False),
    "gheim2_xlmr":      (lambda: HFDetector("checkpoints/gheim2-xlmr"),        559, False),
    "gheim2_swissbert": (lambda: HFDetector("checkpoints/gheim2-swissbert"),   270, True),
    # Baselines for context — not eligible for the leader tie-break.
    "base_priv_filter": (lambda: HFDetector("openai/privacy-filter"),         1400, False),
    "swissner_regex":   (_make_swissner, 270, True),
}


def _summary(rep: dict[str, Any]) -> dict[str, Any]:
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


def _pick_leader(summary: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, str]:
    """Apply the tie-break: F1 within 0.02 → smaller params win → Swiss wins.

    Considers only the gheim2_* detectors; baselines never become leaders.
    Decision is made on test_v1 overlap F1 (the user-facing Swiss benchmark).
    """
    candidates = {n: DETECTORS[n] for n in DETECTORS if n.startswith("gheim2_")}
    f1s = {n: summary[n]["test_v1"]["overlap_overall_f1"] for n in candidates
           if n in summary and "test_v1" in summary[n]}
    if not f1s:
        return ("none", "no candidate detectors evaluated")
    best_f1 = max(f1s.values())
    contenders = [n for n, f in f1s.items() if best_f1 - f <= 0.02]
    # tie-break 1: smaller params
    contenders.sort(key=lambda n: (DETECTORS[n][1], not DETECTORS[n][2]))
    leader = contenders[0]
    rationale = f"test_v1 overlap F1 = {f1s[leader]:.3f}"
    if len(contenders) > 1:
        size = DETECTORS[leader][1]
        swiss = "Swiss-specific" if DETECTORS[leader][2] else "general-purpose"
        rationale += (f"; tied within 0.02 with {len(contenders)-1} other(s), "
                      f"won on {size}M params + {swiss}")
    return (leader, rationale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("eval/gheim2_matrix.json"))
    ap.add_argument("--detectors", type=str, default=None,
                    help="Comma-separated subset to run (default: all).")
    args = ap.parse_args()

    detectors_to_run = (args.detectors.split(",") if args.detectors
                        else list(DETECTORS))

    full: dict[str, dict[str, dict[str, Any]]] = {}
    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for det_name in detectors_to_run:
        if det_name not in DETECTORS:
            print(f"unknown detector {det_name!r}; skipping")
            continue
        factory, _params_m, _swiss = DETECTORS[det_name]
        print(f"\n=== Loading {det_name} ===")
        try:
            det = factory()
        except Exception as exc:
            print(f"  ERROR loading: {exc!r}")
            continue
        full[det_name] = {}
        summary[det_name] = {}
        for set_name, set_path in EVAL_SETS.items():
            if not set_path.exists():
                print(f"  {set_name:<18} SKIP (missing {set_path})")
                continue
            t0 = time.time()
            cases = _load_eval_jsonl(set_path)
            try:
                rep = evaluate(det, cases)
            except Exception as exc:
                print(f"  {set_name:<18} ERROR: {exc!r}")
                continue
            elapsed = time.time() - t0
            full[det_name][set_name] = rep
            summary[det_name][set_name] = _summary(rep)
            print(f"  {set_name:<18} "
                  f"strict={_fmt_f1(rep['strict']['overall']['f1'])} "
                  f"overlap={_fmt_f1(rep['overlap']['overall']['f1'])} "
                  f"({elapsed:.0f}s)")

    print("\n=== Headline overlap F1 ===")
    print(f"{'detector':<22}  " + "  ".join(f"{s:<14}" for s in EVAL_SETS))
    for det_name in detectors_to_run:
        if det_name not in summary:
            continue
        cells = [_fmt_f1(summary[det_name].get(s, {}).get("overlap_overall_f1"))
                 for s in EVAL_SETS]
        print(f"{det_name:<22}  " + "  ".join(f"{c:<14}" for c in cells))

    leader, rationale = _pick_leader(summary)
    print(f"\n=== LEADER: {leader} ===")
    print(f"  {rationale}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_sets": {n: str(p) for n, p in EVAL_SETS.items()},
        "summary": summary,
        "full": full,
        "leader": leader,
        "leader_rationale": rationale,
        "tie_break": "F1 within 0.02 → smaller params → Swiss-specific origin",
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote → {args.out}")


if __name__ == "__main__":
    main()
