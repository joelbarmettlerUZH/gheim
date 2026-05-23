"""V2-7: re-audit the v2 dataset's labels against the existing 4-LLM
OpenRouter consensus.

The v1 P7 audit (paper §3.5) measured F1=0.71 between the dataset's
v1 labels and a majority-of-4 consensus across Kimi-K2.6,
DeepSeek-V4-Pro, MiniMax-M2.7, and GLM-5.1 on the same 580-chunk
sample. This script re-runs that comparison for the v2 dataset,
producing a directly-comparable number.

Re-uses the cached OpenRouter outputs (``data/p7_audit_<model>.jsonl``)
so there's zero new API spend.

For each chunk in the 580-chunk audit set:

1. Load the v2 labels for that chunk. We avoid running the full
   ``assemble.py`` over 2.3M chunks just for 580 audit chunks; instead
   we merge inline from gemma/qwen/nemotron/regex per chunk, same
   logic as ``assemble.merge_signals``.

2. Build the 4-LLM ≥2-of-4 majority consensus from the cached
   ``data/p7_audit_*.jsonl`` outputs.

3. Score v2 labels vs consensus using the same value-string + label
   key match as v1's p7_audit_score (so the F1 number is directly
   comparable).

Outputs
-------
- stdout: per-(lang × cat) F1 + headline number + delta vs v1
- ``data/audit_report.json``: machine-readable

Run
---
    uv run python -m gheim_training.data.analysis.audit_score
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from gheim.detectors.composite import _find_regex_spans_with_subtype

from ..analysis.audit_openrouter import MODELS as P7_AUDIT_MODELS
from ..analysis.audit_openrouter import _safe_slug
from .assemble import _to_raw_spans
from .merge import _RawSpan, merge_signals

AUDIT_GOLD = Path("data/p6_audit_sample.jsonl")
GEMMA_PATH = Path("data/layer5v4.jsonl")
QWEN_PATH = Path("data/layer5v4_qwen.jsonl")
NEMOTRON_PATH = Path("data/layer5v4_nemotron.jsonl")
REPORT_PATH = Path("data/audit_report.json")
LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
CATS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]
V1_F1 = 0.71  # paper §3.5 headline


def _key(value: str, label: str) -> tuple[str, str]:
    return (value.strip().casefold(), label)


def _load_jsonl_by_id(path: Path, *, ids: set[str] | None = None) -> dict[str, dict]:
    """Load a JSONL file into ``{id: record}``. If ``ids`` is given,
    only keeps records with matching ids (avoids loading 2.3M records
    when we only need 580)."""
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("id")
            if cid is None:
                continue
            if ids is not None and cid not in ids:
                continue
            out[cid] = rec
            if ids is not None and len(out) == len(ids):
                break
    return out


def _consensus_4llm(p7_models: dict[str, dict[str, dict]],
                    cid: str) -> set[tuple[str, str]]:
    """≥2-of-4 majority across the original P7 audit models."""
    votes: Counter[tuple[str, str]] = Counter()
    for m in P7_AUDIT_MODELS:
        rec = p7_models[m].get(cid, {"spans": []})
        for sp in rec.get("spans", []):
            votes[_key(sp["value"], sp["label"])] += 1
    return {k for k, v in votes.items() if v >= 2}


def _v2_keys(text: str,
             gemma_claims: list[dict],
             qwen_claims: list[dict],
             nemo_claims: list[dict]) -> set[tuple[str, str]]:
    """Build the v2-merged span key set for one chunk by running
    merge_signals with whichever labellers covered it."""
    gemma_spans = _to_raw_spans(text, gemma_claims)
    qwen_spans = _to_raw_spans(text, qwen_claims)
    nemo_spans = _to_raw_spans(text, nemo_claims)
    regex_spans = [
        _RawSpan(start=sp.start, end=sp.end, label=sp.label,
                 value=text[sp.start:sp.end], regex_subtype=sub)
        for sp, sub in _find_regex_spans_with_subtype(text)
    ]
    # n_candidate_signals affects only confidence; for set-match
    # scoring we just need the merged span keys, so 4 is fine.
    merged = merge_signals(text, gemma=gemma_spans, qwen=qwen_spans,
                           nemotron=nemo_spans, regex=regex_spans,
                           n_candidate_signals=4)
    return {_key(sp.value, sp.label) for sp in merged}


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> None:
    print(f"Loading audit gold ({AUDIT_GOLD})…", flush=True)
    gold = _load_jsonl_by_id(AUDIT_GOLD)
    audit_ids = set(gold.keys())
    print(f"  {len(gold)} audit chunks")

    print("Loading per-labeller spans for the audit subset…", flush=True)
    gemma = _load_jsonl_by_id(GEMMA_PATH, ids=audit_ids)
    qwen = _load_jsonl_by_id(QWEN_PATH, ids=audit_ids)
    nemo = _load_jsonl_by_id(NEMOTRON_PATH, ids=audit_ids)
    print(f"  gemma={len(gemma)} qwen={len(qwen)} nemotron={len(nemo)}")

    print("Loading P7 audit outputs (4 LLMs)…", flush=True)
    p7_models = {
        m: _load_jsonl_by_id(Path(f"data/p7_audit_{_safe_slug(m)}.jsonl"),
                             ids=audit_ids)
        for m in P7_AUDIT_MODELS
    }

    print("Scoring v2 labels vs 4-LLM consensus…", flush=True)
    cell_tp: Counter[tuple[str, str]] = Counter()
    cell_fp: Counter[tuple[str, str]] = Counter()
    cell_fn: Counter[tuple[str, str]] = Counter()
    overall_tp = overall_fp = overall_fn = 0

    for cid, gold_rec in gold.items():
        text = gold_rec["text"]
        lang = gold_rec.get("language", "?")
        v2_keys = _v2_keys(
            text,
            gemma.get(cid, {}).get("spans", []) or [],
            qwen.get(cid, {}).get("qwen_spans", []) or [],
            nemo.get(cid, {}).get("nemotron_spans", []) or [],
        )
        cons = _consensus_4llm(p7_models, cid)

        tp = v2_keys & cons
        fp = v2_keys - cons
        fn = cons - v2_keys
        overall_tp += len(tp)
        overall_fp += len(fp)
        overall_fn += len(fn)
        for k in tp:
            cell_tp[(lang, k[1])] += 1
        for k in fp:
            cell_fp[(lang, k[1])] += 1
        for k in fn:
            cell_fn[(lang, k[1])] += 1

    overall = _f1(overall_tp, overall_fp, overall_fn)
    print()
    print("=" * 70)
    print(f"V2 vs 4-LLM consensus on {len(gold)}-chunk audit set")
    print("=" * 70)
    print(f"  TP={overall_tp}  FP={overall_fp}  FN={overall_fn}")
    print(f"  F1={overall:.3f}  (v1 was {V1_F1:.2f} — delta {(overall - V1_F1)*100:+.1f} pp)")
    print()

    print("Per-(lang × cat) breakdown:")
    print(f"  {'lang':<8} {'cat':<22} {'TP':>5} {'FP':>5} {'FN':>5}  F1")
    for la in LANGS:
        for cat in CATS:
            t = cell_tp[(la, cat)]
            p = cell_fp[(la, cat)]
            n = cell_fn[(la, cat)]
            if t + p + n == 0:
                continue
            print(f"  {la:<8} {cat:<22} {t:>5} {p:>5} {n:>5}  {_f1(t, p, n):.3f}")
    print()

    report = {
        "_meta": {
            "v1_baseline_f1": V1_F1,
            "n_chunks": len(gold),
            "audit_models": list(P7_AUDIT_MODELS),
            "consensus_rule": "≥2-of-4 LLM majority vote (value, label)",
        },
        "overall": {
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
            "f1": overall,
            "delta_pp_vs_v1": (overall - V1_F1) * 100,
        },
        "per_cell": {
            f"{la}__{cat}": {
                "tp": cell_tp[(la, cat)],
                "fp": cell_fp[(la, cat)],
                "fn": cell_fn[(la, cat)],
                "f1": _f1(cell_tp[(la, cat)], cell_fp[(la, cat)],
                          cell_fn[(la, cat)]),
            }
            for la in LANGS for cat in CATS
            if cell_tp[(la, cat)] + cell_fp[(la, cat)] + cell_fn[(la, cat)] > 0
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
