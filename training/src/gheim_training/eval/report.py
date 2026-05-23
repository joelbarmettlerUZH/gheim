"""Aggregate ``eval/ours_*.json`` and ``eval/external_*.json`` into one
matrix JSON and (optionally) the markdown block that ships in
``MODEL_CARD.md``.

Replaces ``compile_positioning_matrix.py`` and the previously
hand-edited markdown table. Every number in the rendered markdown comes
from a JSON field; there are no hard-coded F1 values here.

Usage::

    report.py --out-json eval/positioning_matrix.json
    report.py --out-md   eval/comparison_section.md
    report.py --out-json eval/positioning_matrix.json \\
              --out-md   eval/comparison_section.md
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EVAL_DIR = Path("eval")

# Canonical model id used to flag the headline row.
HEADLINE_MODEL = "joelbarmettler/gheim-ch-560m"

# Friendly labels for the markdown tables. The eval JSONs store the
# raw `model_id` (a HF repo / local checkpoint path / a spaCy spec
# string), which is correct for archival but ugly in the model card.
# Keyed by the substring we want to match in `model_id`.
DISPLAY_NAMES: dict[str, str] = {
    "stage2_xlmr": "joelbarmettler/gheim-ch-560m",
    "stage2_xlmr_onnx_q8": "joelbarmettler/gheim-ch-560m (ONNX int8)",
    "openai/privacy-filter": "openai/privacy-filter (1.4B MoE, zero-shot)",
    "Isotonic": "Isotonic/distilbert_finetuned_ai4privacy_v2",
    "Davlan/xlm-roberta-base-ner-hrl": "Davlan/xlm-roberta-base-ner-hrl",
    "Davlan/distilbert-base-multilingual-cased-ner-hrl":
        "Davlan/distilbert-base-multilingual-cased-ner-hrl",
    "dslim/bert-base-NER": "dslim/bert-base-NER (English slice)",
    "spacy:": "spaCy de/fr/it core_news_lg",
    "presidio": "Microsoft Presidio Analyzer (multi-lang config)",
}

# Models that only emit a person/PER class (no other PII categories).
# Their `Overall F1` across our 8-category schema is artificially low
# because they can never score on email / phone / address / etc., so we
# render it as `n/a` and leave the meaningful comparison in the PER cell.
PER_ONLY_PATTERNS: tuple[str, ...] = (
    "Davlan/",
    "dslim/",
    "spacy:",
)


def _display_name(model_id: str) -> str:
    """Return a friendly display label, falling back to the raw model_id."""
    for key, label in DISPLAY_NAMES.items():
        if key in model_id:
            return label
    return model_id


def _is_per_only(model_id: str) -> bool:
    return any(pat in model_id for pat in PER_ONLY_PATTERNS)


# ---------------------------------------------------------------------------
# JSON ingestion
# ---------------------------------------------------------------------------

def _list_jsons(prefix: str) -> list[Path]:
    if not EVAL_DIR.exists():
        return []
    return sorted(p for p in EVAL_DIR.iterdir()
                  if p.is_file() and p.name.startswith(prefix) and p.suffix == ".json")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _is_headline(blob: dict[str, Any]) -> bool:
    """Identify the gheim-ch-560m row.

    Matches by canonical model id OR by checkpoint paths that historically
    pointed at the same artifact (``stage2_xlmr`` etc.) so reports can
    pick up local-checkpoint runs as the headline too.
    """
    mid = (blob.get("model_id") or "").lower()
    if mid == HEADLINE_MODEL.lower():
        return True
    return "gheim-ch-560m" in mid or "stage2_xlmr" in mid


def _f1(cell: dict[str, Any] | None) -> float | None:
    if not cell:
        return None
    v = cell.get("f1")
    return float(v) if v is not None else None


def _fmt(v: float | None, *, bold: bool = False) -> str:
    """Format an F1 number for the markdown tables; ``None`` renders as
    ``n/a`` and bolds via ``**...**`` when requested."""
    if v is None:
        return "n/a"
    s = f"{v:.3f}"
    return f"**{s}**" if bold else s


def _per_category(blob: dict[str, Any], category: str) -> dict[str, Any]:
    return blob.get("metrics", {}).get("per_category", {}).get(category, {})


def _overall(blob: dict[str, Any]) -> dict[str, Any]:
    return blob.get("metrics", {}).get("overall", {})


def _per_language(blob: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return blob.get("metrics", {}).get("per_language", {})


def _gold_n(blob: dict[str, Any], category: str) -> int:
    cell = _per_category(blob, category).get("strict_span") or {}
    return int(cell.get("n_gold", 0))


# ---------------------------------------------------------------------------
# Direction A / B materialisation
# ---------------------------------------------------------------------------

def _direction_a() -> list[dict[str, Any]]:
    """Other models on our test set (one row per ``ours_*.json``)."""
    rows: list[dict[str, Any]] = []
    for path in _list_jsons("ours_"):
        blob = _load(path)
        ov = _overall(blob)
        pp = _per_category(blob, "private_person")
        rows.append({
            "file": path.name,
            "model_id": blob.get("model_id", path.stem),
            "backend": blob.get("backend"),
            "n_chunks": blob.get("n_chunks"),
            "is_headline": _is_headline(blob),
            "overall_strict_f1": _f1(ov.get("strict_span")),
            "overall_char_f1":   _f1(ov.get("char")),
            "per_strict_f1":     _f1(pp.get("strict_span")),
            "per_char_f1":       _f1(pp.get("char")),
            "per_language":      _per_language(blob),
        })
    rows.sort(key=lambda r: (
        not r["is_headline"],
        -(r["overall_strict_f1"] or r["per_strict_f1"] or 0.0),
    ))
    return rows


def _direction_b() -> list[dict[str, Any]]:
    """gheim on each external benchmark (one row per ``external_*.json``)."""
    rows: list[dict[str, Any]] = []
    for path in _list_jsons("external_"):
        blob = _load(path)
        ov = _overall(blob)
        pp = _per_category(blob, "private_person")
        rows.append({
            "file": path.name,
            "model_id": blob.get("model_id"),
            "dataset_id": blob.get("dataset_id"),
            "n_chunks": blob.get("n_chunks"),
            "overall_strict_f1": _f1(ov.get("strict_span")),
            "overall_char_f1":   _f1(ov.get("char")),
            "per_strict_f1":     _f1(pp.get("strict_span")),
            "per_char_f1":       _f1(pp.get("char")),
            "per_language":      _per_language(blob),
        })
    rows.sort(key=lambda r: -(r["overall_strict_f1"] or 0.0))
    return rows


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _build_matrix() -> dict[str, Any]:
    a = _direction_a()
    b = _direction_b()
    headline_overall = next(
        (r["overall_strict_f1"] for r in a if r["is_headline"] and r["overall_strict_f1"]),
        None,
    )
    return {
        "headline": {
            "model": HEADLINE_MODEL,
            "test_overall_strict_f1": headline_overall,
        },
        "direction_a_other_models_on_our_test": a,
        "direction_b_gheim_on_external_datasets": b,
        "notes": [
            "Direction A: scored on data/built/test (the held-out test split).",
            "Direction B: gheim run on external benchmarks; per-dataset converters "
            "remap labels into our 8-cat schema before scoring.",
            "strict_span uses seqeval-token-level for HF/ONNX backends and "
            "char-set-strict for spaCy/Presidio (which lack a token grid).",
            "char F1 is char-label-aware and is comparable across all backends.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _md_two_metric_table(rows: Iterable[dict[str, Any]]) -> str:
    """Direction A: rows = other-models-on-our-test, columns = strict /
    char overall + per_person. PER-only contestants (Davlan/dslim/spaCy)
    show ``n/a`` in the overall columns since their model can never emit
    non-PER categories; the overall would be unfair to them otherwise."""
    out: list[str] = []
    out.append("| Model | Strict F1 | Char F1 | PER strict | PER char |")
    out.append("|---|---:|---:|---:|---:|")
    for r in rows:
        bold = r["is_headline"]
        per_only = _is_per_only(r["model_id"])
        name = _display_name(r["model_id"])
        if bold:
            name = f"**{name}**"
        overall_strict = "n/a" if per_only else _fmt(r["overall_strict_f1"], bold=bold)
        overall_char   = "n/a" if per_only else _fmt(r["overall_char_f1"],   bold=bold)
        out.append("| " + " | ".join([
            name,
            overall_strict,
            overall_char,
            _fmt(r["per_strict_f1"], bold=bold),
            _fmt(r["per_char_f1"],   bold=bold),
        ]) + " |")
    return "\n".join(out)


def _md_external_table(rows: Iterable[dict[str, Any]]) -> str:
    out: list[str] = []
    out.append("| External benchmark | n_chunks | Overall strict | Overall char | PER strict | PER char |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append("| " + " | ".join([
            f"`{r['dataset_id']}`",
            f"{r['n_chunks'] or 0:,}",
            _fmt(r["overall_strict_f1"]),
            _fmt(r["overall_char_f1"]),
            _fmt(r["per_strict_f1"]),
            _fmt(r["per_char_f1"]),
        ]) + " |")
    return "\n".join(out)


def _md_per_language_top3(rows: list[dict[str, Any]]) -> str:
    """Per-language overall char F1 for the top-three Direction A
    contestants (ranked by overall strict). Headline stays first; the
    next two are the strongest non-headline rows."""
    if not rows:
        return ""
    headline = next((r for r in rows if r["is_headline"]), None)
    others = [r for r in rows if not r["is_headline"]]
    others.sort(key=lambda r: -(r["overall_strict_f1"] or 0.0))
    cols = ([headline] if headline else []) + others[:2]
    cols = [c for c in cols if c is not None]
    if not cols:
        return ""

    langs: set[str] = set()
    for r in cols:
        langs.update(r["per_language"].keys())
    if not langs:
        return ""

    out: list[str] = []
    out.append("| Language | " + " | ".join(_display_name(r["model_id"]) for r in cols) + " |")
    out.append("|---" + "|---:" * len(cols) + "|")
    for la in sorted(langs):
        cells = [la]
        best = max(
            (_f1(r["per_language"].get(la, {}).get("char")) for r in cols),
            default=None,
            key=lambda v: (v is not None, v or 0),
        )
        for r in cols:
            v = _f1(r["per_language"].get(la, {}).get("char"))
            cells.append(_fmt(v, bold=(v is not None and best is not None and v == best)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _md_swissner_per_lang(b_rows: list[dict[str, Any]]) -> str:
    """Per-language PER F1 on the swissner row of Direction B (the
    Romansh-included Swiss-news cell)."""
    swissner = next((r for r in b_rows if "swissner" in (r["dataset_id"] or "").lower()), None)
    if not swissner:
        return ""
    pl = swissner["per_language"]
    if not pl:
        return ""
    out: list[str] = []
    out.append("| Language | PER char F1 |")
    out.append("|---|---:|")
    for la in sorted(pl):
        v = _f1(pl[la].get("char"))
        out.append(f"| {la} | {_fmt(v)} |")
    return "\n".join(out)


def _render_markdown() -> str:
    a = _direction_a()
    b = _direction_b()
    blocks: list[str] = []
    blocks.append("## Comparison to other models and benchmarks")
    blocks.append(
        "To position the model in the broader PII / NER landscape, two "
        "evaluations were run. Direction A scores other open PII detectors "
        "and multilingual NER models on the same held-out test split that "
        "produced the headline F1. Direction B scores `joelbarmettler/"
        "gheim-ch-560m` on widely-used external benchmarks."
    )

    blocks.append("### Two metrics: strict-span and char F1")
    blocks.append(
        "Cross-model PII evaluation has a structural problem: different "
        "models use different entity-segmentation policies. AI4Privacy "
        "splits a person name into `GIVENNAME` + `SURNAME`; `gheim` and "
        "CoNLL emit one combined `private_person` span; address detectors "
        "fragment \"Werdstrasse 36, 8004 Zürich\" at the comma or run it "
        "through. Strict-span F1 (the NER literature standard, computed "
        "by `seqeval`) penalises every boundary mismatch, even when the "
        "model has correctly masked the right characters.\n\n"
        "Both metrics are reported. **Strict-span F1**: exact (start, "
        "end, label) match per span via `seqeval` (token-level) or "
        "char-set match (for span-emitting backends like spaCy / Presidio "
        "that lack a token grid). **Char F1**: per-character precision "
        "and recall on the (character, category) set; fragmentation-"
        "invariant; reflects the redaction utility the model is built for."
    )

    blocks.append("### Direction A: other models on our held-out test set")
    blocks.append(_md_two_metric_table(a))

    p3 = _md_per_language_top3(a)
    if p3:
        blocks.append("Per-language overall char F1 (top three Direction A contestants):")
        blocks.append(p3)

    blocks.append("### Direction B: `gheim-ch-560m` on external benchmarks")
    blocks.append(
        "Each external dataset uses its own label schema; PER / SURNAME / "
        "GIVENNAME etc. are mapped to `private_person` and remaining "
        "categories to the closest `gheim` cell or to `O`. For pure-NER "
        "datasets (swissner, CoNLL-2003, WikiNeural) only PER maps to a "
        "`gheim` category, so the meaningful number is the PER cell. The "
        "non-PER `Overall F1` is dragged down by `gheim` predicting "
        "categories the NER datasets don't label."
    )
    blocks.append(_md_external_table(b))

    sn = _md_swissner_per_lang(b)
    if sn:
        blocks.append("Per-language PER char F1 on Swiss news (`swissner`):")
        blocks.append(sn)

    blocks.append(
        "<!-- Methodology footnote: strict_span is computed via "
        "seqeval-token-level for HF/ONNX backends and via char-set-strict "
        "for spaCy/Presidio (which emit char spans directly). char F1 is "
        "char-label-aware and uses an identical formulation across all "
        "backends. The full numerical breakdown is in "
        "[`eval/positioning_matrix.json`](eval/positioning_matrix.json). -->"
    )

    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-json", type=Path, default=None,
                   help="Where to write the matrix JSON. Skipped if omitted.")
    p.add_argument("--out-md", type=Path, default=None,
                   help="Where to write the comparison-section markdown. Skipped if omitted.")
    args = p.parse_args()

    if not args.out_json and not args.out_md:
        raise SystemExit("pass --out-json and/or --out-md")

    if args.out_json:
        matrix = _build_matrix()
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(matrix, indent=2))
        print(f"Wrote {args.out_json}", flush=True)
        a = matrix["direction_a_other_models_on_our_test"]
        b = matrix["direction_b_gheim_on_external_datasets"]
        print(f"  direction A: {len(a)} rows", flush=True)
        print(f"  direction B: {len(b)} rows", flush=True)

    if args.out_md:
        md = _render_markdown()
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
        print(f"Wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
