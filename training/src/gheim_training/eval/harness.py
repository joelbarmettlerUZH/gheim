"""Span-level F1 evaluation across language × entity cuts.

Given any object exposing ``detect(text) -> list[Span]`` (compatible with
``gheim.detectors.base.Span`` or our ``data.schema.Span``), runs it against
the handcrafted eval JSONLs (and optionally other eval splits) and prints a
per-cut F1 matrix.

Usage:
    uv run python -m gheim_training.eval.harness \\
        --detector finetuned --model-path ./checkpoints/gheim-1-v1
    uv run python -m gheim_training.eval.harness \\
        --detector zero_shot --model-id openai/privacy-filter
    uv run python -m gheim_training.eval.harness \\
        --detector swissbert_ner
    uv run python -m gheim_training.eval.harness \\
        --detector presidio_ch
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from ..data.label_space import CATEGORIES
from ..data.schema import Span

EVAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "handcrafted"


class Detector(Protocol):
    def detect(self, text: str) -> list: ...


def _load_handcrafted() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in sorted(EVAL_DIR.glob("*.jsonl")):
        lang = p.stem
        cases: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
        out[lang] = cases
    return out


def _to_set(spans) -> set[tuple[int, int, str]]:
    s: set[tuple[int, int, str]] = set()
    for sp in spans:
        # Accept either dataclass-shaped or dict-shaped spans, and either our
        # internal categories or the gheim detector labels (same names).
        start = sp.start if hasattr(sp, "start") else sp["start"]
        end = sp.end if hasattr(sp, "end") else sp["end"]
        label = sp.label if hasattr(sp, "label") else sp["label"]
        s.add((int(start), int(end), str(label)))
    return s


def _f1(tp: int, fp: int, fn: int) -> float | None:
    """Span-level F1. Returns None when the cell has no data (no gold AND
    no predictions) so it doesn't pollute aggregate averages — sparse
    per-(language, category) cells are common and "0.0" would be wrong."""
    if tp + fp + fn == 0:
        return None
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def evaluate(detector: Detector) -> dict[str, dict[str, float]]:
    cases = _load_handcrafted()

    # Per (lang, entity) → tp/fp/fn
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for lang, items in cases.items():
        for item in items:
            gold = _to_set(item["spans"])
            pred = _to_set(detector.detect(item["text"]))
            for cat in CATEGORIES:
                gold_c = {s for s in gold if s[2] == cat}
                pred_c = {s for s in pred if s[2] == cat}
                tp = len(gold_c & pred_c)
                fp = len(pred_c - gold_c)
                fn = len(gold_c - pred_c)
                counts[(lang, cat)][0] += tp
                counts[(lang, cat)][1] += fp
                counts[(lang, cat)][2] += fn

    # Roll up: per language, per entity, overall.
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_ent: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    overall = [0, 0, 0]
    for (lang, cat), (tp, fp, fn) in counts.items():
        for i, v in enumerate((tp, fp, fn)):
            by_lang[lang][i] += v
            by_ent[cat][i] += v
            overall[i] += v

    report: dict[str, dict[str, float]] = {
        "by_language": {k: {"f1": _f1(*v), "tp": v[0], "fp": v[1], "fn": v[2]} for k, v in by_lang.items()},
        "by_entity": {k: {"f1": _f1(*v), "tp": v[0], "fp": v[1], "fn": v[2]} for k, v in by_ent.items()},
        "by_lang_entity": {
            f"{lang}/{cat}": {"f1": _f1(*v), "tp": v[0], "fp": v[1], "fn": v[2]}
            for (lang, cat), v in counts.items()
        },
        "overall": {"f1": _f1(*overall), "tp": overall[0], "fp": overall[1], "fn": overall[2]},
    }
    return report


def _fmt_f1(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


def _print(report: dict) -> None:
    print("\n=== Overall ===")
    o = report["overall"]
    print(f"  F1={_fmt_f1(o['f1'])}  TP={o['tp']}  FP={o['fp']}  FN={o['fn']}")

    print("\n=== By language ===")
    for lang, s in sorted(report["by_language"].items()):
        print(f"  {lang:<8} F1={_fmt_f1(s['f1'])}  (TP={s['tp']} FP={s['fp']} FN={s['fn']})")

    print("\n=== By entity ===")
    for ent, s in sorted(report["by_entity"].items()):
        print(f"  {ent:<18} F1={_fmt_f1(s['f1'])}  (TP={s['tp']} FP={s['fp']} FN={s['fn']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True,
                    choices=("finetuned", "zero_shot", "presidio_ch", "swissbert_ner"))
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-id", default="openai/privacy-filter")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.detector == "finetuned":
        from .baselines.zero_shot import HFDetector
        det = HFDetector(args.model_path or args.model_id)
    elif args.detector == "zero_shot":
        from .baselines.zero_shot import HFDetector
        det = HFDetector(args.model_id)
    elif args.detector == "presidio_ch":
        from .baselines.presidio_ch import PresidioCHDetector
        det = PresidioCHDetector()
    elif args.detector == "swissbert_ner":
        from .baselines.swissbert_ner import SwissBertNERDetector
        det = SwissBertNERDetector()
    else:
        raise ValueError(args.detector)

    report = evaluate(det)
    _print(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport → {args.out}")


# Re-export Span for baseline modules that don't want to import gheim.
__all__ = ["Span", "evaluate", "Detector"]


if __name__ == "__main__":
    main()
