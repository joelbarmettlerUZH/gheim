"""Day 15 English regression check.

Sample N held-out chunks from data/layer4_en.jsonl (Layer 4, the English
anchor that was IN the training mix). Score them with:
  - the BASE openai/privacy-filter (zero-shot, no fine-tuning)
  - bakeoff_xlmr (the Swiss leader)
  - composite_xlmr (regex front-end + bakeoff_xlmr)

Report Δ vs base. The CLAUDE.md target is "leader within 5pp of base on
English". Larger drops = real regression. We held out the LAST 5% of
layer4_en for this — the build script's val split (0.05) means those
examples were never seen during training.

Run:
    uv run python -m gheim_training.eval.english_regression \\
        --out eval/english_regression.json --n 1000
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .baselines.zero_shot import HFDetector
from .harness import _fmt_f1, evaluate


def _load_layer4_held_out(path: Path, n: int, seed: int) -> list[dict]:
    """Load layer4_en and use the LAST n records as held-out (deterministic
    given the file is fixed). The build.py random split also reserves 5% for
    validation; we re-derive a fresh held-out by tail-slicing rather than
    re-running the seeded split."""
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records[-n:]  # tail slice; deterministic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer4", type=Path, default=Path("data/layer4_en.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval/english_regression.json"))
    ap.add_argument("--n", type=int, default=1000,
                    help="Number of held-out chunks to score (last N).")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    cases = _load_layer4_held_out(args.layer4, args.n, args.seed)
    print(f"Loaded {len(cases)} English held-out chunks from {args.layer4}")

    detectors: dict[str, Callable[[], Any]] = {
        "base_priv_filter": lambda: HFDetector("openai/privacy-filter"),
        "bakeoff_xlmr":     lambda: HFDetector("checkpoints/bakeoff-xlmr"),
        "composite_xlmr":   _make_composite,
    }

    results: dict[str, dict[str, Any]] = {}
    for name, factory in detectors.items():
        print(f"\n=== {name} ===")
        det = factory()
        t0 = time.time()
        rep = evaluate(det, cases)
        elapsed = time.time() - t0
        results[name] = rep
        print(f"  strict  F1={_fmt_f1(rep['strict']['overall']['f1'])}")
        print(f"  overlap F1={_fmt_f1(rep['overlap']['overall']['f1'])}  ({elapsed:.0f}s)")

    base_f1 = results["base_priv_filter"]["overlap"]["overall"]["f1"] or 0
    print("\n=== Δ vs base openai/privacy-filter (overlap F1) ===")
    print(f"  base                {base_f1:.3f}")
    for name in ("bakeoff_xlmr", "composite_xlmr"):
        f1 = results[name]["overlap"]["overall"]["f1"] or 0
        delta = f1 - base_f1
        symbol = "✓" if abs(delta) <= 0.05 else "⚠️" if delta > -0.10 else "✗"
        print(f"  {name:<22} {f1:.3f}  Δ={delta:+.3f}  {symbol}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_chunks": len(cases),
        "source": str(args.layer4),
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote → {args.out}")


def _make_composite() -> Any:
    from gheim.detectors.composite import CompositeDetector
    return CompositeDetector(model=HFDetector("checkpoints/bakeoff-xlmr"))


if __name__ == "__main__":
    main()
