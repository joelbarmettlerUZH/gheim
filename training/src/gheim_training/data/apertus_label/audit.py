"""Compute Apertus's labeling F1 against a hand-labeled audit set.

Workflow:
  1. Run ``generate.py`` on a small batch (say 200 chunks) → JSONL.
  2. Author hand labels for those same 200 chunks under
     ``training/eval/apertus_audit/<lang>.jsonl`` (same schema as eval/handcrafted).
  3. Run ``audit.py`` to compare Apertus output vs. hand labels per category.

Phase 1 gate (per CLAUDE.md):
  - private_person: P ≥ 0.85, R ≥ 0.80
  - private_address: P ≥ 0.80, R ≥ 0.75
  - private_date: P ≥ 0.90, R ≥ 0.85

If any of these fail, switch to Apertus-70B 4-bit via llama.cpp.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _spans_set(record: dict, label_filter: set[str] | None = None) -> set[tuple[int, int, str]]:
    out: set[tuple[int, int, str]] = set()
    for s in record["spans"]:
        if label_filter is None or s["label"] in label_filter:
            out.add((int(s["start"]), int(s["end"]), str(s["label"])))
    return out


def audit(apertus_path: Path, gold_path: Path) -> dict:
    apertus = {r["text"]: r for r in _load_jsonl(apertus_path)}
    gold = _load_jsonl(gold_path)

    # Audit only the three categories Apertus is responsible for.
    AUDIT_LABELS = {"private_person", "private_address", "private_date"}

    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    n_seen = 0
    n_missing_in_apertus = 0

    for g in gold:
        text = g["text"]
        a = apertus.get(text)
        if a is None:
            n_missing_in_apertus += 1
            continue
        n_seen += 1
        gold_spans = _spans_set(g, AUDIT_LABELS)
        pred_spans = _spans_set(a, AUDIT_LABELS)
        for label in AUDIT_LABELS:
            g_l = {s for s in gold_spans if s[2] == label}
            p_l = {s for s in pred_spans if s[2] == label}
            counts[label][0] += len(g_l & p_l)  # tp
            counts[label][1] += len(p_l - g_l)  # fp
            counts[label][2] += len(g_l - p_l)  # fn

    report: dict[str, dict] = {}
    for label, (tp, fp, fn) in counts.items():
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        report[label] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
        }
    report["_meta"] = {
        "audited_chunks": n_seen,
        "missing_in_apertus_output": n_missing_in_apertus,
        "gold_chunks": len(gold),
    }
    return report


# Gate criteria from CLAUDE.md.
GATE = {
    "private_person":  {"precision": 0.85, "recall": 0.80},
    "private_address": {"precision": 0.80, "recall": 0.75},
    "private_date":    {"precision": 0.90, "recall": 0.85},
}


def check_gate(report: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for label, thresholds in GATE.items():
        scores = report.get(label, {})
        p = scores.get("precision", 0.0)
        r = scores.get("recall", 0.0)
        if p < thresholds["precision"]:
            failures.append(f"{label}: P={p:.3f} < {thresholds['precision']}")
        if r < thresholds["recall"]:
            failures.append(f"{label}: R={r:.3f} < {thresholds['recall']}")
    return (not failures), failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apertus", type=Path, required=True,
                    help="JSONL produced by gheim_training.data.apertus_label.generate")
    ap.add_argument("--gold", type=Path, required=True,
                    help="Hand-labeled JSONL (same schema as eval/handcrafted)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = audit(args.apertus, args.gold)
    print(json.dumps(report, indent=2))
    passed, failures = check_gate(report)
    print()
    if passed:
        print("GATE: PASS — Apertus-8B meets Phase 1 thresholds.")
    else:
        print("GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        print("Action: switch to Apertus-70B 4-bit via llama.cpp; re-run audit.")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport → {args.out}")


if __name__ == "__main__":
    main()
