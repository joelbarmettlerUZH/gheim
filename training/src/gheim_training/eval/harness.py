"""Span-level F1 evaluation across language × entity cuts.

Given any object exposing ``detect(text) -> list[Span]`` (compatible with
``gheim.detectors.base.Span`` or our ``data.schema.Span``), runs it against
a JSONL eval file and prints/returns a per-cut F1 matrix.

Each JSONL line: ``{"text": str, "language": str, "spans": [{"start", "end",
"label"}, ...]}``. ``language`` is optional — falls back to ``"unknown"``.

Both strict (exact-tuple) and overlap (any-overlap, same-label) F1 are
reported. Overlap F1 is the production-meaningful number for redaction;
strict F1 measures boundary precision.

Usage:
    uv run python -m gheim_training.eval.harness \\
        --detector zero_shot \\
        --model-id openai/privacy-filter \\
        --eval-jsonl data/test_v1.jsonl
    uv run python -m gheim_training.eval.harness \\
        --detector finetuned \\
        --model-path ./checkpoints/bakeoff-priv \\
        --eval-jsonl data/test_v1.jsonl \\
        --out eval/bakeoff_priv_test_v1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from ..data.label_space import CATEGORIES
from ..data.schema import Span


class Detector(Protocol):
    """Anything with a ``detect(text) -> list[Span]`` method. Both ``Span``
    dataclasses and dict-shaped spans are tolerated downstream by ``_to_set``,
    but the protocol is stricter so type-checkers catch detectors that
    accidentally return raw HF pipeline output."""
    def detect(self, text: str) -> list[Span]: ...


def _load_eval_jsonl(path: Path) -> list[dict]:
    """Load ``{"text", "language"?, "spans": [...]}`` records from a JSONL file."""
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def _to_set(spans: list[Any]) -> set[tuple[int, int, str]]:
    """Coerce a list of Span dataclasses or dicts to {(start, end, label)} tuples."""
    out: set[tuple[int, int, str]] = set()
    for sp in spans:
        # Accept either dataclass-shaped or dict-shaped spans.
        start = sp.start if hasattr(sp, "start") else sp["start"]
        end = sp.end if hasattr(sp, "end") else sp["end"]
        label = sp.label if hasattr(sp, "label") else sp["label"]
        out.add((int(start), int(end), str(label)))
    return out


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


def _cell(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Build one cell of the report dict (F1 + raw counts)."""
    return {"f1": _f1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def _overlap_match(g: tuple[int, int, str], p: tuple[int, int, str]) -> bool:
    """True iff g and p share a label and any character span overlap."""
    return g[2] == p[2] and g[0] < p[1] and p[0] < g[1]


def _count_overlap(
    gold: set[tuple[int, int, str]],
    pred: set[tuple[int, int, str]],
) -> tuple[int, int, int]:
    """Lenient (overlap) matching: a gold span counts as TP if any predicted
    span with the same label overlaps it. Each predicted span can match at
    most one gold span (and vice versa) to avoid double-counting."""
    matched_g: set[tuple[int, int, str]] = set()
    matched_p: set[tuple[int, int, str]] = set()
    for g in gold:
        for p in pred:
            if p in matched_p:
                continue
            if _overlap_match(g, p):
                matched_g.add(g)
                matched_p.add(p)
                break
    tp = len(matched_g)
    fp = len(pred - matched_p)
    fn = len(gold - matched_g)
    return tp, fp, fn


def evaluate(detector: Detector, cases: list[dict]) -> dict[str, Any]:
    """Run detector against eval cases and return a structured report.

    Per-(language, category) buckets, rolled up to per-language, per-entity,
    and overall. Both strict and overlap F1 are reported.

    If any case carries a ``presence`` field (``"pii_dense"`` or
    ``"control"``, set by the test_v1 surfacer), an additional
    ``by_presence`` section reports false-positive counts on the control
    subset specifically — every prediction on a control chunk is by
    definition a false positive (controls have no gold spans), so the
    raw FP count is the cleanest measure of the detector's noise floor.
    """
    strict_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    overlap_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    # Per-presence FP and chunk count: lets us report "FPs per control chunk"
    # which is the bake-off's noise-floor metric.
    presence_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [fp, n_chunks]

    for item in cases:
        gold = _to_set(item["spans"])
        pred = _to_set(detector.detect(item["text"]))
        lang = item.get("language") or "unknown"
        presence = item.get("presence")
        if isinstance(presence, str):
            presence_counts[presence][0] += len(pred - gold)
            presence_counts[presence][1] += 1
        for cat in CATEGORIES:
            gold_c = {s for s in gold if s[2] == cat}
            pred_c = {s for s in pred if s[2] == cat}

            # Strict matching: exact (start, end, label) tuple equality.
            s_tp = len(gold_c & pred_c)
            s_fp = len(pred_c - gold_c)
            s_fn = len(gold_c - pred_c)
            strict_counts[(lang, cat)][0] += s_tp
            strict_counts[(lang, cat)][1] += s_fp
            strict_counts[(lang, cat)][2] += s_fn

            # Overlap matching: any character overlap, same label.
            o_tp, o_fp, o_fn = _count_overlap(gold_c, pred_c)
            overlap_counts[(lang, cat)][0] += o_tp
            overlap_counts[(lang, cat)][1] += o_fp
            overlap_counts[(lang, cat)][2] += o_fn

    report: dict[str, Any] = {
        "strict": _build_report(strict_counts),
        "overlap": _build_report(overlap_counts),
        "n_cases": len(cases),
    }
    if presence_counts:
        report["by_presence"] = {
            presence: {
                "fp": fp,
                "n_chunks": n,
                "fp_per_chunk": fp / n if n else None,
            }
            for presence, (fp, n) in presence_counts.items()
        }
    return report


def _build_report(counts: dict[tuple[str, str], list[int]]) -> dict[str, Any]:
    """Roll up per-(lang, cat) counts to per-lang, per-entity, and overall."""
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_ent: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    overall = [0, 0, 0]
    for (lang, cat), (tp, fp, fn) in counts.items():
        for i, v in enumerate((tp, fp, fn)):
            by_lang[lang][i] += v
            by_ent[cat][i] += v
            overall[i] += v
    return {
        "by_language": {k: _cell(*v) for k, v in by_lang.items()},
        "by_entity": {k: _cell(*v) for k, v in by_ent.items()},
        "by_lang_entity": {
            f"{lang}/{cat}": _cell(*v) for (lang, cat), v in counts.items()
        },
        "overall": _cell(*overall),
    }


def _fmt_f1(v: float | None) -> str:
    return "  n/a" if v is None else f"{v:.3f}"


def _print(report: dict[str, Any]) -> None:
    n = report.get("n_cases", "?")
    for mode in ("strict", "overlap"):
        sub = report[mode]
        print(f"\n=== {mode.upper()} (n={n}) ===")
        o = sub["overall"]
        print(f"  Overall F1={_fmt_f1(o['f1'])}  TP={o['tp']}  FP={o['fp']}  FN={o['fn']}")
        print("  By language:")
        for lang, s in sorted(sub["by_language"].items()):
            print(f"    {lang:<8} F1={_fmt_f1(s['f1'])}  (TP={s['tp']} FP={s['fp']} FN={s['fn']})")
        print("  By entity:")
        for ent, s in sorted(sub["by_entity"].items()):
            print(f"    {ent:<18} F1={_fmt_f1(s['f1'])}  (TP={s['tp']} FP={s['fp']} FN={s['fn']})")
    if "by_presence" in report:
        print("\n=== By presence (false-positive rate on PII-free controls) ===")
        for presence, s in sorted(report["by_presence"].items()):
            fpc = s.get("fp_per_chunk")
            fpc_s = "n/a" if fpc is None else f"{fpc:.3f}"
            print(f"  {presence:<10} n={s['n_chunks']:>4}  FP={s['fp']:>4}  FP/chunk={fpc_s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", required=True,
                    choices=("finetuned", "zero_shot", "swissbert_ner"),
                    help="Which detector to evaluate.")
    ap.add_argument("--eval-jsonl", type=Path, required=True,
                    help="JSONL with {text, language?, spans: [...]}.")
    ap.add_argument("--model-path", default=None,
                    help="Local checkpoint path (used by --detector finetuned).")
    ap.add_argument("--model-id", default="openai/privacy-filter",
                    help="Hub model id (used by --detector zero_shot).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON report output path.")
    args = ap.parse_args()

    if args.detector == "finetuned":
        from .baselines.zero_shot import HFDetector
        det: Detector = HFDetector(args.model_path or args.model_id)
    elif args.detector == "zero_shot":
        from .baselines.zero_shot import HFDetector
        det = HFDetector(args.model_id)
    elif args.detector == "swissbert_ner":
        from .baselines.swissbert_ner import SwissBertNERDetector
        det = SwissBertNERDetector()
    else:
        raise ValueError(args.detector)

    cases = _load_eval_jsonl(args.eval_jsonl)
    report = evaluate(det, cases)
    _print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport → {args.out}")


# Re-export Span for baseline modules that don't want to import gheim.
__all__ = ["Span", "evaluate", "Detector"]


if __name__ == "__main__":
    main()
