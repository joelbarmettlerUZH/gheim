"""Shared scoring helpers used across the eval scripts.

Kept tiny on purpose: only the helpers that were duplicated verbatim in
char_metric.py / spacy_test_eval.py / presidio_test_eval.py live here, so
the JSON outputs of all three remain byte-comparable to their prior runs.
"""
from __future__ import annotations


def f1_pr(c: dict[str, int]) -> dict[str, float]:
    """Compute F1 / precision / recall from a {tp, fp, fn} counter.

    Returns zeros for empty cells rather than raising — the eval reports
    aggregate over many categories and a sparse cell would otherwise
    interrupt scoring.
    """
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"f1": f, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}
