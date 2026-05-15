"""Score the two extra v2 candidate labellers against:

1. The dataset's own labels (the released gheim-ch-pii v1 labels for the
   580-chunk P7 audit sample).
2. The existing four-LLM majority consensus from the P7 audit
   (Kimi + DeepSeek-V4-Pro + MiniMax + GLM, ≥2-of-4).

Reports per-model F1 numbers and a per-category breakdown so we can
judge whether either candidate is worth scaling up to the full
2.3M-chunk corpus.

Decision rules (the user-set thresholds from the v2 plan):

  - F1 vs consensus ≥0.65 → scale up (the model agrees with the
    majority more than it disagrees)
  - F1 vs consensus <0.50 → skip (model is mostly an outlier)
  - 0.50–0.65 → judgement call: scaling buys some signal but at cost

Run
---
    uv run python -m gheim_training.data.v2.p7_extra_score
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..p7_audit_openrouter import MODELS as P7_ORIGINAL_MODELS
from ..p7_audit_openrouter import _safe_slug

EXTRA_MODELS = [
    "deepseek/deepseek-v4-flash",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]
GOLD_PATH = Path("data/p6_audit_sample.jsonl")
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]
LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]


def _key(value: str, label: str) -> tuple[str, str]:
    return (value.strip().casefold(), label)


def _load_jsonl(path: Path) -> dict[str, dict]:
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


def _consensus_4llm(p7_models: dict[str, dict[str, dict]],
                    cid: str) -> set[tuple[str, str]]:
    """≥2-of-4 majority across the original P7 audit models."""
    votes: Counter[tuple[str, str]] = Counter()
    for m in P7_ORIGINAL_MODELS:
        rec = p7_models[m].get(cid, {"spans": []})
        for sp in rec.get("spans", []):
            votes[_key(sp["value"], sp["label"])] += 1
    return {k for k, v in votes.items() if v >= 2}


def _score_against(reference_per_chunk: dict[str, set[tuple[str, str]]],
                   candidate_per_chunk: dict[str, set[tuple[str, str]]],
                   gold_lang: dict[str, str]) -> dict:
    """Compute per-cat / per-lang / overall F1 of candidate vs reference."""
    tp = fp = fn = 0
    per_cat_tp: Counter[str] = Counter()
    per_cat_fp: Counter[str] = Counter()
    per_cat_fn: Counter[str] = Counter()
    per_lang_tp: Counter[str] = Counter()
    per_lang_fp: Counter[str] = Counter()
    per_lang_fn: Counter[str] = Counter()
    for cid, ref in reference_per_chunk.items():
        cand = candidate_per_chunk.get(cid, set())
        lang = gold_lang.get(cid, "?")
        tp += len(ref & cand)
        fp += len(cand - ref)
        fn += len(ref - cand)
        for k in ref & cand:
            per_cat_tp[k[1]] += 1
            per_lang_tp[lang] += 1
        for k in cand - ref:
            per_cat_fp[k[1]] += 1
            per_lang_fp[lang] += 1
        for k in ref - cand:
            per_cat_fn[k[1]] += 1
            per_lang_fn[lang] += 1
    return {
        "tp": tp, "fp": fp, "fn": fn, "f1": _f1(tp, fp, fn),
        "per_cat": {c: {
            "tp": per_cat_tp[c], "fp": per_cat_fp[c], "fn": per_cat_fn[c],
            "f1": _f1(per_cat_tp[c], per_cat_fp[c], per_cat_fn[c]),
        } for c in CATS},
        "per_lang": {l: {
            "tp": per_lang_tp[l], "fp": per_lang_fp[l], "fn": per_lang_fn[l],
            "f1": _f1(per_lang_tp[l], per_lang_fp[l], per_lang_fn[l]),
        } for l in LANGS},
    }


def main() -> None:
    gold = _load_jsonl(GOLD_PATH)
    gold_lang = {cid: rec.get("language", "?") for cid, rec in gold.items()}

    # 1. Reference set A: dataset's own labels
    dataset_per_chunk = {
        cid: {_key(sp["value"], sp["label"]) for sp in rec.get("spans", [])}
        for cid, rec in gold.items()
    }

    # 2. Reference set B: existing 4-LLM majority consensus
    p7_models = {
        m: _load_jsonl(Path(f"data/p7_audit_{_safe_slug(m)}.jsonl"))
        for m in P7_ORIGINAL_MODELS
    }
    consensus_per_chunk = {
        cid: _consensus_4llm(p7_models, cid) for cid in gold
    }

    # 3. Load each candidate and score
    print("Candidate F1 numbers on the 580-chunk P7 audit sample\n")
    print(f"{'Model':<60} {'F1 vs dataset':>14} {'F1 vs consensus':>17}")
    print("-" * 95)
    full_report = {}
    for model in EXTRA_MODELS:
        path = Path(f"data/p7_audit_{_safe_slug(model)}.jsonl")
        if not path.exists():
            print(f"  {model}: MISSING ({path})")
            continue
        cand = _load_jsonl(path)
        cand_per_chunk = {
            cid: {_key(sp["value"], sp["label"]) for sp in rec.get("spans", [])}
            for cid, rec in cand.items()
        }
        vs_dataset = _score_against(dataset_per_chunk, cand_per_chunk, gold_lang)
        vs_cons = _score_against(consensus_per_chunk, cand_per_chunk, gold_lang)
        print(f"{model:<60} {vs_dataset['f1']:>14.3f} {vs_cons['f1']:>17.3f}")
        full_report[model] = {
            "vs_dataset": vs_dataset,
            "vs_consensus": vs_cons,
        }

    # Reference: also score each original P7 model the same way, for context.
    print("\nReference: same numbers for the four original P7 audit models\n")
    print(f"{'Model':<60} {'F1 vs dataset':>14} {'F1 vs consensus':>17}")
    print("-" * 95)
    for m in P7_ORIGINAL_MODELS:
        recs = p7_models[m]
        cand_per_chunk = {
            cid: {_key(sp["value"], sp["label"]) for sp in rec.get("spans", [])}
            for cid, rec in recs.items()
        }
        vs_dataset = _score_against(dataset_per_chunk, cand_per_chunk, gold_lang)
        vs_cons = _score_against(consensus_per_chunk, cand_per_chunk, gold_lang)
        print(f"{m:<60} {vs_dataset['f1']:>14.3f} {vs_cons['f1']:>17.3f}")

    # Per-candidate detailed breakdown
    for model, rep in full_report.items():
        print(f"\n--- {model} ---")
        print(f"  Overall vs consensus: F1={rep['vs_consensus']['f1']:.3f} "
              f"(P={rep['vs_consensus']['tp']/(rep['vs_consensus']['tp']+rep['vs_consensus']['fp']):.3f}, "
              f"R={rep['vs_consensus']['tp']/(rep['vs_consensus']['tp']+rep['vs_consensus']['fn']):.3f})")
        print("  Per-category F1 vs consensus:")
        for c in CATS:
            v = rep["vs_consensus"]["per_cat"][c]
            print(f"    {c:<22} TP={v['tp']:>4} FP={v['fp']:>4} FN={v['fn']:>4}  F1={v['f1']:.3f}")
        print("  Per-language F1 vs consensus:")
        for la in LANGS:
            v = rep["vs_consensus"]["per_lang"][la]
            print(f"    {la:<8} TP={v['tp']:>5} FP={v['fp']:>5} FN={v['fn']:>5}  F1={v['f1']:.3f}")

    # Save machine-readable
    out_path = Path("data/p7_audit_extra_report.json")
    out_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
