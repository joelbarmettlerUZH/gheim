"""Day 15 production-readiness report.

Three questions, one report:

1. Did we improve over the BASE openai/privacy-filter? (and over swissner+regex?)
2. How do we compare to other open-source PII tools (Microsoft Presidio
   with Swiss-aware recognizers)?
3. Are our metrics actually good enough for prod, per language and per
   category, with Wilson 95% CIs and explicit thresholds?

Produces:
- eval/day15_report.json — full structured report
- prints a human-readable per-(language, category) prod-readiness matrix
  with thresholds:
    overlap F1 ≥ 0.90  → ✅  ship-ready (high confidence)
    overlap F1 ≥ 0.80  → ✓  acceptable (deploy with monitoring)
    overlap F1 ≥ 0.70  → ⚠️  marginal (not recommended without humans-in-loop)
    overlap F1 <  0.70  → ✗  not ready
    n_gold = 0          → —  unmeasured

Run:
    uv run python -m gheim_training.eval.day15_report \\
        --out eval/day15_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .baselines.zero_shot import HFDetector
from .harness import _load_eval_jsonl, evaluate

EVAL_SETS: dict[str, Path] = {
    "test_v1": Path("data/test_v1.jsonl"),
}


def _wilson_95(p: float | None, n: int) -> tuple[float, float] | None:
    """Wilson 95% CI for a Bernoulli proportion (used for F1 with TP+FN gold)."""
    if p is None or n <= 0:
        return None
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return centre - half, centre + half


def _readiness_marker(f1: float | None, n: int) -> str:
    if n == 0 or f1 is None:
        return "—"
    if f1 >= 0.90:
        return "✅"
    if f1 >= 0.80:
        return "✓"
    if f1 >= 0.70:
        return "⚠️"
    return "✗"


def _detectors() -> dict[str, Callable[[], Any]]:
    return {
        "base_openai_privacy_filter": lambda: HFDetector("openai/privacy-filter"),
        "swissner_plus_regex":        _make_swissner,
        "presidio_ch":                _make_presidio,
        "gheim_xlmr_bare":            lambda: HFDetector("checkpoints/bakeoff-xlmr"),
        "gheim_xlmr_composite":       _make_composite,
    }


def _make_composite() -> Any:
    from gheim.detectors.composite import CompositeDetector
    return CompositeDetector(model=HFDetector("checkpoints/bakeoff-xlmr"))


def _make_swissner() -> Any:
    from .baselines.swissbert_ner import SwissBertNERDetector
    return SwissBertNERDetector()


def _make_presidio() -> Any:
    from .baselines.presidio_ch import PresidioCHDetector
    return PresidioCHDetector()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("eval/day15_report.json"))
    args = ap.parse_args()

    cases = _load_eval_jsonl(EVAL_SETS["test_v1"])
    print(f"Loaded {len(cases)} chunks from test_v1")
    print("  Per-language presence: see eval/test_v1 readme.")

    full: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for det_name, factory in _detectors().items():
        print(f"\n=== {det_name} ===")
        det = factory()
        t0 = time.time()
        try:
            rep = evaluate(det, cases)
        except Exception as exc:
            print(f"  ERROR: {exc!r}")
            full[det_name] = {"error": repr(exc)}
            summary[det_name] = {"error": repr(exc)}
            continue
        elapsed = time.time() - t0
        full[det_name] = rep
        summary[det_name] = _summary(rep)
        print(f"  overlap F1: {_fmt(rep['overlap']['overall']['f1'])}  "
              f"strict F1: {_fmt(rep['strict']['overall']['f1'])}  "
              f"({elapsed:.0f}s)")

    _print_q1(summary)
    _print_q2(summary)
    _print_q3(summary, full)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_cases": len(cases),
        "summary": summary,
        "full": full,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote → {args.out}")


def _summary(rep: dict[str, Any]) -> dict[str, Any]:
    return {
        "overlap_overall_f1": rep["overlap"]["overall"]["f1"],
        "strict_overall_f1": rep["strict"]["overall"]["f1"],
        "overall_tp": rep["overlap"]["overall"]["tp"],
        "overall_fp": rep["overlap"]["overall"]["fp"],
        "overall_fn": rep["overlap"]["overall"]["fn"],
        "by_language_overlap": {
            lang: {"f1": v["f1"], "tp": v["tp"], "fp": v["fp"], "fn": v["fn"]}
            for lang, v in rep["overlap"]["by_language"].items()
        },
        "by_entity_overlap": {
            ent: {"f1": v["f1"], "tp": v["tp"], "fp": v["fp"], "fn": v["fn"]}
            for ent, v in rep["overlap"]["by_entity"].items()
        },
        "by_lang_entity_overlap": {
            le: {"f1": v["f1"], "tp": v["tp"], "fp": v["fp"], "fn": v["fn"]}
            for le, v in rep["overlap"]["by_lang_entity"].items()
        },
        "by_presence": rep.get("by_presence"),
    }


def _fmt(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


def _print_q1(summary: dict[str, Any]) -> None:
    """Q1: Did we improve over the base openai/privacy-filter?"""
    print("\n" + "=" * 78)
    print("Q1: DID WE IMPROVE OVER BASE openai/privacy-filter?")
    print("=" * 78)
    base = summary.get("base_openai_privacy_filter", {}).get("overlap_overall_f1")
    if base is None:
        print("  base detector failed; can't compare.")
        return
    print(f"  base openai/privacy-filter overlap F1 on test_v1: {base:.3f}")
    print()
    for det in ("gheim_xlmr_bare", "gheim_xlmr_composite",
                "swissner_plus_regex", "presidio_ch"):
        f1 = summary.get(det, {}).get("overlap_overall_f1")
        if f1 is None:
            print(f"  {det:<28} n/a")
            continue
        delta = f1 - base
        s = "✅" if delta >= 0.30 else "✓" if delta >= 0.10 else "⚠️" if delta >= 0 else "✗"
        print(f"  {det:<28} {f1:.3f}  Δ={delta:+.3f}  {s}")


def _print_q2(summary: dict[str, Any]) -> None:
    """Q2: How do we compare to other open-source tools?"""
    print("\n" + "=" * 78)
    print("Q2: VS OTHER OPEN-SOURCE PII TOOLS (overlap F1 on test_v1)")
    print("=" * 78)
    for det in ("base_openai_privacy_filter", "swissner_plus_regex",
                "presidio_ch", "gheim_xlmr_bare", "gheim_xlmr_composite"):
        f1 = summary.get(det, {}).get("overlap_overall_f1")
        # FP/control
        pres = summary.get(det, {}).get("by_presence") or {}
        ctrl = pres.get("control") or {}
        fp_ctrl = ctrl.get("fp_per_chunk")
        f1s = "  n/a" if f1 is None else f"{f1:.3f}"
        fps = "  n/a" if fp_ctrl is None else f"{fp_ctrl:.3f}"
        print(f"  {det:<28} F1={f1s}    FP/control={fps}")


def _print_q3(summary: dict[str, Any], full: dict[str, Any]) -> None:
    """Q3: Per-(language, category) prod readiness."""
    print("\n" + "=" * 78)
    print("Q3: PROD READINESS — gheim_xlmr_composite per (language × category)")
    print("=" * 78)
    print("  ✅ ≥0.90  ✓ ≥0.80  ⚠️ ≥0.70  ✗ <0.70  — unmeasured (no gold)")
    print()
    rep = full.get("gheim_xlmr_composite") or full.get("gheim_xlmr_bare")
    if not rep or "error" in rep:
        print("  leader detector failed; can't build matrix.")
        return
    by_le = rep["overlap"]["by_lang_entity"]
    langs = sorted({le.split("/")[0] for le in by_le})
    cats = sorted({le.split("/")[1] for le in by_le})
    head = "  cat / lang        " + "  ".join(f"{lang:<8}" for lang in langs)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for cat in cats:
        cells = []
        for lang in langs:
            cell = by_le.get(f"{lang}/{cat}", {})
            n_gold = cell.get("tp", 0) + cell.get("fn", 0)
            f1 = cell.get("f1")
            marker = _readiness_marker(f1, n_gold)
            f1s = "  n/a" if f1 is None else f"{f1:.2f}"
            cells.append(f"{marker} {f1s} (n={n_gold:>3})")
        print(f"  {cat:<22}  " + " ".join(cells))

    # Per-language overall + Wilson 95% CI
    print()
    print("  Per-language overall (overlap F1, ±half-width 95% CI):")
    for lang, cell in sorted(rep["overlap"]["by_language"].items()):
        n_gold = cell["tp"] + cell["fn"]
        f1 = cell["f1"]
        ci = _wilson_95(f1, max(n_gold, 1))
        if f1 is None or ci is None:
            print(f"    {lang:<8} n/a (n={n_gold})")
        else:
            half = (ci[1] - ci[0]) / 2
            marker = _readiness_marker(f1, n_gold)
            print(f"    {lang:<8} {marker} {f1:.3f} ± {half:.3f}  (n={n_gold})")


if __name__ == "__main__":
    main()
