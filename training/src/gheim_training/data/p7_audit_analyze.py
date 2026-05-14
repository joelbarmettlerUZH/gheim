"""P7 audit step 3: classify the disagreements between dataset and consensus.

Given the four per-model labellings produced by ``p7_audit_openrouter.py`` and
the dataset's own labels (the ``p6_audit_sample.jsonl`` chunks), this script
splits every false-positive (consensus-only) and false-negative (dataset-only)
into one of three buckets so we can tell *what* kind of disagreement is
driving the headline F1 of 0.71:

1. **BOUNDARY** — same entity, different boundary. The other side has a span
   on the same chunk with the same label whose value is a substring of this
   one (e.g. ``"Müller"`` vs ``"Dr. Müller"``) or shares a token with it
   (e.g. fragmented vs joined surname pieces). Counted as boundary noise.

2. **CAT-SWAP** — same string, different label. Both sides flagged the same
   value but disagreed on which of the eight categories it belongs to.

3. **REAL** — neither boundary overlap nor category swap. The two labellers
   genuinely disagree about whether the entity is PII at all (or whether
   that surface form is in scope).

Output
------
``data/p7_audit_analysis.json`` containing:
- the strict and "fuzzy" (substring-overlap-on-same-label) F1 numbers, so the
  paper can quote how much of the gap is boundary noise
- per-(category, direction) counts of REAL disagreements
- a small bank of example chunks per (category, direction) for paper prose

Run
---
    uv run python -m gheim_training.data.p7_audit_analyze
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from gheim_training.data.p7_audit_openrouter import MODELS, _safe_slug

GOLD_PATH = Path("data/p6_audit_sample.jsonl")
OUT_PATH = Path("data/p7_audit_analysis.json")
EXAMPLES_PER_BUCKET = 8


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


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _consensus(model_outputs: dict[str, dict[str, dict]], cid: str) -> set[tuple[str, str]]:
    votes: Counter[tuple[str, str]] = Counter()
    for m in MODELS:
        rec = model_outputs[m].get(cid, {"spans": []})
        for sp in rec.get("spans", []):
            votes[_key(sp["value"], sp["label"])] += 1
    return {k for k, v in votes.items() if v >= 2}


def _has_overlap(target_value: str, others: list[str]) -> bool:
    """Substring or word-overlap match against any of ``others``."""
    for o in others:
        if target_value in o or o in target_value:
            return True
        if set(target_value.split()) & set(o.split()):
            return True
    return False


def _context(text: str, value: str, window: int = 50) -> str:
    """Return a short text excerpt around the first occurrence of ``value``."""
    i = text.lower().find(value.lower())
    if i < 0:
        return value
    start = max(0, i - window)
    end = min(len(text), i + len(value) + window)
    snippet = text[start:end].replace("\n", " ")
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(text) else "")


def main() -> None:
    gold = _load_jsonl_by_id(GOLD_PATH)
    model_outputs = {
        m: _load_jsonl_by_id(Path(f"data/p7_audit_{_safe_slug(m)}.jsonl"))
        for m in MODELS
    }

    # Strict and fuzzy counters (one global tally each).
    strict_tp = strict_fp = strict_fn = 0
    fuzzy_tp = fuzzy_fp = fuzzy_fn = 0

    # REAL-disagreement classification, per (label, direction).
    # direction is 'dataset_over' (dataset has it, consensus doesn't)
    # or 'consensus_over' (consensus has it, dataset doesn't).
    real_counts: dict[tuple[str, str], int] = Counter()
    boundary_counts: dict[tuple[str, str], int] = Counter()
    catswap_counts: dict[tuple[str, str], int] = Counter()
    real_examples: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for cid, gold_rec in gold.items():
        text = gold_rec["text"]
        gold_keys = {_key(sp["value"], sp["label"]) for sp in gold_rec.get("spans", [])}
        cons_keys = _consensus(model_outputs, cid)

        # Strict
        strict_tp += len(gold_keys & cons_keys)
        strict_fp += len(cons_keys - gold_keys)
        strict_fn += len(gold_keys - cons_keys)

        # Fuzzy: a gold key matches if ANY consensus key has same label and value-overlap.
        gold_matched: set[tuple[str, str]] = set()
        cons_matched: set[tuple[str, str]] = set()
        for kg in gold_keys:
            v_g, lbl_g = kg
            for kc in cons_keys:
                v_c, lbl_c = kc
                if lbl_g == lbl_c and (v_g in v_c or v_c in v_g):
                    gold_matched.add(kg)
                    cons_matched.add(kc)
                    break
        fuzzy_tp += len(gold_matched)
        fuzzy_fp += len(cons_keys - cons_matched)
        fuzzy_fn += len(gold_keys - gold_matched)

        # Classify each FN
        for k in gold_keys - cons_keys:
            v_g, lbl_g = k
            same_label_in_c = [v for v, lbl in cons_keys if lbl == lbl_g]
            cat_swap = any(v_g == v_c for v_c, lbl_c in cons_keys if lbl_c != lbl_g)
            if _has_overlap(v_g, same_label_in_c):
                boundary_counts[(lbl_g, "dataset_over")] += 1
            elif cat_swap:
                catswap_counts[(lbl_g, "dataset_over")] += 1
            else:
                real_counts[(lbl_g, "dataset_over")] += 1
                if len(real_examples[(lbl_g, "dataset_over")]) < EXAMPLES_PER_BUCKET:
                    real_examples[(lbl_g, "dataset_over")].append(
                        {"value": v_g, "context": _context(text, v_g),
                         "language": gold_rec.get("language")}
                    )

        # Classify each FP
        for k in cons_keys - gold_keys:
            v_c, lbl_c = k
            same_label_in_g = [v for v, lbl in gold_keys if lbl == lbl_c]
            cat_swap = any(v_c == v_g for v_g, lbl_g in gold_keys if lbl_g != lbl_c)
            if _has_overlap(v_c, same_label_in_g):
                boundary_counts[(lbl_c, "consensus_over")] += 1
            elif cat_swap:
                catswap_counts[(lbl_c, "consensus_over")] += 1
            else:
                real_counts[(lbl_c, "consensus_over")] += 1
                if len(real_examples[(lbl_c, "consensus_over")]) < EXAMPLES_PER_BUCKET:
                    real_examples[(lbl_c, "consensus_over")].append(
                        {"value": v_c, "context": _context(text, v_c),
                         "language": gold_rec.get("language")}
                    )

    # ---------- Print + structure for JSON ----------
    print("Strict F1:", round(_f1(strict_tp, strict_fp, strict_fn), 4),
          f"(TP={strict_tp} FP={strict_fp} FN={strict_fn})")
    print("Fuzzy  F1:", round(_f1(fuzzy_tp, fuzzy_fp, fuzzy_fn), 4),
          f"(TP={fuzzy_tp} FP={fuzzy_fp} FN={fuzzy_fn})")
    boundary_pp = (_f1(fuzzy_tp, fuzzy_fp, fuzzy_fn) - _f1(strict_tp, strict_fp, strict_fn)) * 100
    print(f"Boundary noise contributes only {boundary_pp:.1f} pp.\n")

    print("Total disagreements (FN + FP):")
    total_real = sum(real_counts.values())
    total_boundary = sum(boundary_counts.values())
    total_catswap = sum(catswap_counts.values())
    total = total_real + total_boundary + total_catswap
    print(f"  REAL (policy / scope drift): {total_real} ({100*total_real/total:.1f}%)")
    print(f"  BOUNDARY (same entity, different span): {total_boundary} ({100*total_boundary/total:.1f}%)")
    print(f"  CAT-SWAP (same value, different label): {total_catswap} ({100*total_catswap/total:.1f}%)")
    print()

    print("REAL disagreements per (category, direction):")
    cats = sorted({c for c, _ in real_counts})
    for cat in cats:
        d_over = real_counts.get((cat, "dataset_over"), 0)
        c_over = real_counts.get((cat, "consensus_over"), 0)
        print(f"  {cat:20s}  dataset-only={d_over:4d}  consensus-only={c_over:4d}")
    print()

    out = {
        "_meta": {
            "models": MODELS,
            "n_chunks": len(gold),
            "scoring": "value-string equality (casefold) + label match",
            "fuzzy_match_rule": "same label + substring overlap either direction",
        },
        "f1": {
            "strict": {"tp": strict_tp, "fp": strict_fp, "fn": strict_fn,
                       "f1": _f1(strict_tp, strict_fp, strict_fn)},
            "fuzzy": {"tp": fuzzy_tp, "fp": fuzzy_fp, "fn": fuzzy_fn,
                      "f1": _f1(fuzzy_tp, fuzzy_fp, fuzzy_fn)},
            "boundary_noise_pp": round(boundary_pp, 2),
        },
        "totals": {
            "real": total_real,
            "boundary": total_boundary,
            "cat_swap": total_catswap,
        },
        "per_category_real": {
            cat: {
                "dataset_over": real_counts.get((cat, "dataset_over"), 0),
                "consensus_over": real_counts.get((cat, "consensus_over"), 0),
            }
            for cat in cats
        },
        "per_category_boundary": {
            cat: {
                "dataset_over": boundary_counts.get((cat, "dataset_over"), 0),
                "consensus_over": boundary_counts.get((cat, "consensus_over"), 0),
            }
            for cat in sorted({c for c, _ in boundary_counts})
        },
        "examples": {
            f"{cat}__{direction}": real_examples[(cat, direction)]
            for (cat, direction) in sorted(real_examples.keys())
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
