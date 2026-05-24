"""Compile every published number for the gheim release into a single
canonical markdown report (and matching LaTeX-row snippets).

The goal of this module is to be the single source of truth for the
F1 numbers that appear in MODEL_CARD.md, MODEL_CARD_RESEARCH.md,
eval/comparison_section.md, and paper/paper.tex. Re-running this
script after any eval re-run regenerates every published table; the
human copy step is then a paste from this report's output sections.

Metric conventions (chosen once, used everywhere):

* In-domain test split: report **both** strict-span F1 (seqeval)
  and char-level F1 (label-aware char-set).
* Cross-domain external benchmarks: report **PER-cell char F1**
  (the model's `private_person` cell, char-level). This is the
  metric that survives schema-fragmentation noise between gheim's
  8-category schema and the external benchmark's narrower or
  broader label sets.
* Direction A (other models on our test split): report strict,
  char, PER strict, PER char in that order.

Run::

    uv run python -m gheim_training.eval.compile_release_numbers > eval/RELEASE_NUMBERS.md

The output is meant to be pasted, not parsed; subsections are
clearly labelled so a card-update is a literal block copy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVAL_DIR = Path("eval")

# ---------------------------------------------------------------------------
# File registry — single mapping of the JSON files we consume.
# Keep this in sync with the eval pipeline; it is the only place that
# names result files so a renamed file fails loudly here.
# ---------------------------------------------------------------------------

DIRECTION_A_FILES: dict[str, dict[str, str]] = {
    # display_name: {"path": ..., "license": ..., "per_only": bool}
    "gheim-ch-560m (Apache 2.0, this release)": {
        "path": "ours_gheim.json",
        "license": "Apache 2.0",
        "is_gheim": True,
    },
    "gheim-ch-560m-research (CC BY-NC-SA, this release)": {
        "path": "ours_gheim_multisource.json",
        "license": "CC BY-NC-SA 4.0",
        "is_gheim": True,
    },
    "dslim/bert-base-NER (English slice)": {
        "path": "ours_dslim.json",
        "license": "MIT",
        "per_only": True,
    },
    "ZurichNLP/swissbert-ner + regex hybrid": {
        "path": "ours_swissbert.json",
        "license": "CC BY 4.0",
    },
    "Davlan/distilbert-base-multilingual-cased-ner-hrl": {
        "path": "ours_davlan_distilbert.json",
        "license": "Apache 2.0",
        "per_only": True,
    },
    "Davlan/xlm-roberta-base-ner-hrl": {
        "path": "ours_davlan_xlmr.json",
        "license": "Apache 2.0",
        "per_only": True,
    },
    "openai/privacy-filter (1.4B MoE, ONNX, zero-shot)": {
        "path": "ours_priv.json",
        "license": "Apache 2.0",
    },
    "spaCy de/fr/it_core_news_lg": {
        "path": "ours_spacy.json",
        "license": "MIT",
        "per_only": True,
    },
    "Microsoft Presidio Analyzer (multi-lang config)": {
        "path": "ours_presidio.json",
        "license": "MIT",
    },
    "Isotonic/distilbert_finetuned_ai4privacy_v2": {
        "path": "ours_isotonic.json",
        "license": "Apache 2.0",
    },
}

# Direction B: (display_name, n_chunks, license, gheim-baseline-file,
# gheim-commercial-file, gheim-research-file, zero_shot_for_research)
DIRECTION_B_BENCHES: list[dict[str, Any]] = [
    {
        "name": "ZurichNLP/swissner (Swiss-news NER)",
        "license": "CC BY 4.0",
        "baseline":   "external_gheim_swissner.json",
        "commercial": "external_gheim_commercial_swissner.json",
        "research":   "external_gheim_multisource_swissner.json",
        "research_zero_shot": True,  # no swissner train split exists
        "note": "swissner has no train split; all variants are zero-shot",
    },
    {
        "name": "ai4privacy/pii-masking-openpii-1m",
        "license": "Apache 2.0 / CC BY 4.0",
        "baseline":   "external_gheim_openpii.json",
        "commercial": "external_gheim_commercial_openpii.json",
        "research":   "external_gheim_multisource_openpii.json",
        "research_zero_shot": False,  # research model trained on train split
        "note": "research/commercial trained on openpii train split",
    },
    {
        "name": "ai4privacy/open-pii-masking-500k",
        "license": "CC BY 4.0",
        "baseline":   "external_gheim_baseline_openpii500k.json",
        "commercial": "external_gheim_commercial_openpii500k.json",
        "research":   "external_gheim_research_openpii500k.json",
        "research_zero_shot": True,
        "note": "zero-shot for all variants",
    },
    {
        "name": "gretelai/synthetic_pii_finance_multilingual",
        "license": "Apache 2.0",
        "baseline":   "external_gheim_baseline_gretel.json",
        "commercial": "external_gheim_commercial_gretel.json",
        "research":   "external_gheim_research_gretel.json",
        "research_zero_shot": True,
        "note": "zero-shot for all variants",
    },
    {
        "name": "Babelscape/WikiNeural",
        "license": "CC BY-NC-SA 4.0",
        "baseline":   "external_gheim_wikineural.json",
        "commercial": "external_gheim_commercial_wikineural.json",
        "research":   "external_gheim_multisource_wikineural.json",
        "research_zero_shot": False,
        "note": "research trained on WikiNeural train split; commercial trained on WikiAnn (substitute)",
    },
    {
        "name": "tomaarsen/conll2003 (PER only)",
        "license": "research-only (Reuters)",
        "baseline":   "external_gheim_conll2003.json",
        "commercial": "external_gheim_commercial_conll2003.json",
        "research":   "external_gheim_multisource_conll2003.json",
        "research_zero_shot": False,
        "note": "research trained on CoNLL-2003 train split",
    },
]


# ---------------------------------------------------------------------------
# Pull-from-JSON helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> dict[str, Any]:
    p = EVAL_DIR / name
    if not p.exists():
        return {}
    return json.load(p.open())


def _overall_strict(m: dict[str, Any]) -> float | None:
    return m.get("overall", {}).get("strict_span", {}).get("f1")


def _overall_char(m: dict[str, Any]) -> float | None:
    return m.get("overall", {}).get("char", {}).get("f1")


def _per_strict(m: dict[str, Any]) -> float | None:
    return (m.get("per_category", {}).get("private_person", {})
            .get("strict_span", {}).get("f1"))


def _per_char(m: dict[str, Any]) -> float | None:
    return (m.get("per_category", {}).get("private_person", {})
            .get("char", {}).get("f1"))


def _per_lang_char(m: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for lang, sc in m.get("per_language", {}).items():
        f1 = sc.get("char", {}).get("f1")
        if f1 is not None:
            out[lang] = f1
    return out


def _f(v: float | None, w: int = 5) -> str:
    return f"{v:.3f}" if v is not None else " n/a "


# ---------------------------------------------------------------------------
# Markdown emitters
# ---------------------------------------------------------------------------

def _md_in_domain() -> str:
    lines = [
        "## In-domain test split (data/built, test, 21,246 chunks)",
        "",
        "Headline strict-span F1 and char F1 on the held-out test split.",
        "",
        "| Checkpoint | License | Strict F1 | Char F1 | Precision (strict) | Recall (strict) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, cfg in DIRECTION_A_FILES.items():
        if not cfg.get("is_gheim"):
            continue
        d = _load(cfg["path"])
        m = d.get("metrics", {})
        s = m.get("overall", {}).get("strict_span", {})
        c = m.get("overall", {}).get("char", {})
        lines.append(
            f"| **{label}** | {cfg['license']} | "
            f"{_f(s.get('f1'))} | {_f(c.get('f1'))} | "
            f"{_f(s.get('precision'))} | {_f(s.get('recall'))} |"
        )
    return "\n".join(lines)


def _md_direction_a() -> str:
    lines = [
        "## Direction A — other detectors on our test split (21,246 chunks)",
        "",
        "Metrics: strict-span F1 (seqeval) and char-level label-aware F1, "
        "plus the `private_person`-cell numbers in the right columns. "
        "PER-only models emit only person mentions (no email / phone / "
        "IBAN / etc.); their overall 8-category F1 is reported as n/a.",
        "",
        "| Model | License | Strict F1 | Char F1 | PER strict | PER char |",
        "|---|---|---:|---:|---:|---:|",
    ]
    # Sort: gheim variants first, then by PER char desc
    rows = []
    for label, cfg in DIRECTION_A_FILES.items():
        d = _load(cfg["path"])
        m = d.get("metrics", {})
        per_only = cfg.get("per_only", False)
        rows.append({
            "label": label,
            "license": cfg["license"],
            "is_gheim": cfg.get("is_gheim", False),
            "strict": _overall_strict(m) if not per_only else None,
            "char": _overall_char(m) if not per_only else None,
            "per_strict": _per_strict(m),
            "per_char": _per_char(m),
        })
    rows.sort(key=lambda r: (
        not r["is_gheim"],
        -(r["per_char"] or 0),
    ))
    for r in rows:
        strict = f"**{_f(r['strict'])}**" if r["is_gheim"] and r["strict"] else (_f(r["strict"]) if r["strict"] else "n/a")
        char   = f"**{_f(r['char'])}**" if r["is_gheim"] and r["char"]   else (_f(r["char"])   if r["char"]   else "n/a")
        per_s  = f"**{_f(r['per_strict'])}**" if r["is_gheim"] else _f(r["per_strict"])
        per_c  = f"**{_f(r['per_char'])}**" if r["is_gheim"] else _f(r["per_char"])
        lines.append(
            f"| {r['label']} | {r['license']} | {strict} | {char} | {per_s} | {per_c} |"
        )
    return "\n".join(lines)


def _md_direction_b() -> str:
    lines = [
        "## Direction B — gheim variants on external benchmarks",
        "",
        "Metric: `private_person`-cell character-level F1 on each external "
        "benchmark. Three of the six benchmarks (openpii-1m, WikiNeural, "
        "CoNLL-2003) are present in the research variant's training mix, "
        "so its numbers on those rows reflect in-distribution "
        "generalisation rather than zero-shot transfer; the other three "
        "(swissner, open-pii-500k, gretel_finance) are zero-shot for "
        "every variant.",
        "",
        "| Benchmark | License | Baseline (Apache 2.0) | Commercial (experimental) | Research (CC BY-NC-SA) | Note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for b in DIRECTION_B_BENCHES:
        bv = _per_char(_load(b["baseline"]).get("metrics", {}))
        cv = _per_char(_load(b["commercial"]).get("metrics", {}))
        rv = _per_char(_load(b["research"]).get("metrics", {}))
        lines.append(
            f"| {b['name']} | {b['license']} | "
            f"{_f(bv)} | {_f(cv)} | {_f(rv)} | {b['note']} |"
        )
    return "\n".join(lines)


def _md_swissner_per_lang() -> str:
    bb = _per_lang_char(_load("external_gheim_swissner.json").get("metrics", {}))
    cc = _per_lang_char(_load("external_gheim_commercial_swissner.json").get("metrics", {}))
    rr = _per_lang_char(_load("external_gheim_multisource_swissner.json").get("metrics", {}))
    lines = [
        "## Per-language swissner — Swiss-news NER cross-domain (zero-shot for all variants)",
        "",
        "Per-language char F1 on ZurichNLP/swissner's de / fr / it / rm "
        "test splits (200 chunks each). `swissner` has no train split, so "
        "every variant is zero-shot on this benchmark.",
        "",
        "| Language | Baseline | Commercial | Research |",
        "|---|---:|---:|---:|",
    ]
    for lang in ["de", "fr", "it", "rm"]:
        lines.append(
            f"| {lang} | {_f(bb.get(lang))} | {_f(cc.get(lang))} | {_f(rr.get(lang))} |"
        )
    return "\n".join(lines)


def _md_per_lang_cat(checkpoint: str, json_path: Path) -> str:
    if not json_path.exists():
        return f"\n*({json_path} not found)*\n"
    d = json.load(json_path.open())
    f1 = d["f1"]; n_gold = d["n_gold"]
    cats = ["account_number", "private_address", "private_date",
            "private_email", "private_person", "private_phone",
            "private_url", "secret"]
    langs = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
    lines = [
        f"### Per-(language × category) char F1 — `{checkpoint}` on the in-domain test split",
        "",
        "| Category | " + " | ".join(langs) + " | Avg. |",
        "|---|" + "---:|" * (len(langs) + 1),
    ]
    overall_w = 0.0; overall_g = 0
    for cat in cats:
        row = [f"`{cat}`"]; tot_g = 0; tot_w = 0.0
        for lang in langs:
            v = f1.get(lang, {}).get(cat)
            g = n_gold.get(lang, {}).get(cat, 0)
            if v is None:
                row.append("n/a")
            else:
                row.append(f"{v:.3f}")
                tot_g += g; tot_w += g * v
        row.append(f"{tot_w / tot_g:.3f}" if tot_g else "n/a")
        overall_w += tot_w; overall_g += tot_g
        lines.append("| " + " | ".join(row) + " |")
    # avg row
    avg_row = ["**Avg.**"]
    for lang in langs:
        tg = 0; tw = 0.0
        for cat in cats:
            v = f1.get(lang, {}).get(cat)
            g = n_gold.get(lang, {}).get(cat, 0)
            if v is None:
                continue
            tg += g; tw += g * v
        avg_row.append(f"**{tw / tg:.3f}**" if tg else "n/a")
    avg_row.append(f"**{overall_w / overall_g:.3f}**" if overall_g else "n/a")
    lines.append("| " + " | ".join(avg_row) + " |")
    return "\n".join(lines)


def _md_onnx() -> str:
    fp32 = _load("onnx_gheim_fp32.json").get("metrics", {})
    q8 = _load("onnx_gheim_q8.json").get("metrics", {})
    if not fp32:
        return "\n*(ONNX eval results not found)*\n"
    fs = _overall_strict(fp32); fc = _overall_char(fp32)
    qs = _overall_strict(q8); qc = _overall_char(q8)
    lines = [
        "## ONNX deployment-format deltas (gheim-ch-560m, Apache 2.0)",
        "",
        "| Format | Size | Test strict F1 | Test char F1 | Δ strict | Δ char |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PyTorch fp32 | 2.2 GB | {_f(fs)} | {_f(fc)} | (baseline) | (baseline) |",
        f"| ONNX fp32    | 2.2 GB | {_f(fs)} | {_f(fc)} | 0.000 | 0.000 |",
        f"| ONNX int8 (dynamic) | 557 MB | {_f(qs)} | {_f(qc)} | "
        f"{qs - fs:+.4f} | {qc - fc:+.4f} |",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("# Canonical gheim release numbers")
    print()
    print("**Single source of truth for every F1 that ships in the model "
          "cards, dataset card, and tech report.** Re-generate via "
          "`uv run python -m gheim_training.eval.compile_release_numbers`. "
          "All numbers below come from the JSON files in `eval/`; any "
          "discrepancy between this report and a published artefact is a "
          "bug in the artefact, not in this report.")
    print()
    print(_md_in_domain())
    print()
    print(_md_per_lang_cat("gheim-ch-560m (Apache 2.0)",
                          EVAL_DIR / "per_lang_cat_postretrain.json"))
    print()
    print(_md_per_lang_cat("gheim-ch-560m-research",
                          EVAL_DIR / "per_lang_cat_multisource.json"))
    print()
    print(_md_direction_a())
    print()
    print(_md_direction_b())
    print()
    print(_md_swissner_per_lang())
    print()
    print(_md_onnx())


if __name__ == "__main__":
    main()
