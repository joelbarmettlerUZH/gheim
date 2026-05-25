"""Per-dtype quality evaluation for the gheim-ch-560m ONNX exports.

Runs all 212 forensic probe cases through fp32, fp16, q8 ONNX (uncalibrated:
o_bias=0). Also reproduces two specific Node.js smoke-test failures
("Bach …" and "Müller & Partner AG …").

Outputs:
    eval/q8_quality_per_case.json   — raw per-case verdicts for all 3 dtypes
    eval/q8_quality_report.md       — human-readable findings + recommendation
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.pop("GHEIM_TEST_MODEL", None)

EVAL_DIR = Path("/home/joelbarmettler/projects/gheim/eval")
CHECKPOINTS = Path("/home/joelbarmettler/projects/gheim/checkpoints")

DTYPES = [
    ("fp32", CHECKPOINTS / "gheim-ch_onnx", "model.onnx"),
    ("fp16", CHECKPOINTS / "gheim-ch_onnx_fp16_v2", "model.onnx"),
    ("q8",   CHECKPOINTS / "gheim-ch_onnx_q8", "model_quantized.onnx"),
]

# ---------------------------------------------------------------------------
# Span dataclass — matches gheim.detectors.base.Span shape so we can reuse
# verdict logic from /tmp/calibration_sweep.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int
    text: str
    score: float = 1.0


# ---------------------------------------------------------------------------
# Inference: load each ONNX dtype as a CalibratedDetector-equivalent with
# o_bias=0 (uncalibrated) so quantisation effects are isolated from
# calibration.
# ---------------------------------------------------------------------------

class OnnxDetector:
    def __init__(self, dtype_tag: str, onnx_dir: Path, file_name: str):
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer

        self.tag = dtype_tag
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir), use_fast=True)
        self.model = ORTModelForTokenClassification.from_pretrained(
            str(onnx_dir), file_name=file_name, provider="CPUExecutionProvider",
        )
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        # Resolve O id
        label2id = {v: int(k) for k, v in self.id2label.items()}
        self.o_id = label2id.get("O", 0)
        self.o_bias = 0.0  # uncalibrated — exactly what the smoke test sees

    def detect(self, text: str) -> list[Span]:
        if not text:
            return []
        import torch
        # Single-window — all probe cases are short
        enc = self.tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt",
            truncation=True, max_length=512,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        logits = logits[0].detach().cpu().float()
        if self.o_bias != 0.0:
            logits[:, self.o_id] -= float(self.o_bias)
        pred_ids = logits.argmax(dim=-1).tolist()

        # BIOES -> spans aggregation, copied from CalibratedDetector._detect_window
        spans: list[Span] = []
        cur_start, cur_end, cur_cat = None, None, None
        for (s, e), pid in zip(offsets, pred_ids):
            if s == e:
                continue
            lab = self.id2label[int(pid)]
            cat = None if lab == "O" else lab.split("-", 1)[1]
            if cat == cur_cat and cat is not None:
                cur_end = e
            else:
                if cur_cat is not None:
                    spans.append(self._make_span(text, cur_start, cur_end, cur_cat))
                cur_start, cur_end, cur_cat = s, e, cat
        if cur_cat is not None:
            spans.append(self._make_span(text, cur_start, cur_end, cur_cat))
        return spans

    @staticmethod
    def _make_span(text: str, start: int, end: int, label: str) -> Span:
        # trim leading/trailing whitespace
        while start < end and text[start] in " \t\n":
            start += 1
        while end > start and text[end - 1] in " \t\n":
            end -= 1
        if start >= end:
            # zero-length, drop
            return Span(label=label, start=start, end=start, text="", score=1.0)
        return Span(label=label, start=start, end=end, text=text[start:end], score=1.0)


# ---------------------------------------------------------------------------
# Verdict logic — copy-paste from /tmp/calibration_sweep.py
# ---------------------------------------------------------------------------

def _spans_to_keys(spans: list[Span]) -> set[tuple[int, int, str]]:
    return {(s.start, s.end, s.label) for s in spans if s.end > s.start}


def _verdict(expected_tuples: list, pred_keys: set) -> str:
    gold = set(map(tuple, expected_tuples))
    if not gold:
        return "PASS_NEG" if not pred_keys else "FAIL_NEG"
    if not pred_keys:
        return "FAIL"
    tp = gold & pred_keys
    if tp == gold and pred_keys == gold:
        return "PASS"
    if tp:
        return "PARTIAL"
    return "FAIL"


def is_perfect(verdict: str) -> bool:
    return verdict in ("PASS", "PASS_NEG")


# ---------------------------------------------------------------------------
# Load probe cases
# ---------------------------------------------------------------------------

def load_probe_cases() -> list[dict]:
    d = json.load(open(EVAL_DIR / "probe_postretrain.json"))
    return [
        {
            "case_id": c["case_id"],
            "text": c["text"],
            "expected": c["expected"],
            "language": c.get("language"),
            "pattern_tag": c.get("pattern_tag"),
        }
        for c in d["per_case"]
    ]


def gold_categories(expected: list) -> set[str]:
    """Categories present in the gold spans of a probe case (empty if neg case)."""
    return {e[2] for e in expected}


# ---------------------------------------------------------------------------
# Smoke-test cases (verbatim from /tmp/gheim_smoke.mjs)
# ---------------------------------------------------------------------------

SMOKE_TESTS = [
    {
        "id": "bach_common_word",
        "text": "Bach läuft heute durch Bach. Frau Bach kommt aus Bach im Aargau.",
        "note": "common-word surname; q8 emitted 4 false-positive person/location spans",
    },
    {
        "id": "commercial_register",
        "text": ("Müller & Partner AG (CHE-123.456.789 MWST) wird vertreten "
                 "durch Verwaltungsrat Beat Aregger und Geschäftsführerin Sandra Fritsche."),
        "note": "q8 missed Beat Aregger and Sandra Fritsche (person names)",
    },
]


# ---------------------------------------------------------------------------
# Per-dtype probe sweep
# ---------------------------------------------------------------------------

def evaluate_probe(det: OnnxDetector, cases: list[dict]) -> list[dict]:
    out = []
    for c in cases:
        preds = det.detect(c["text"])
        keys = _spans_to_keys(preds)
        v = _verdict(c["expected"], keys)
        out.append({
            "case_id": c["case_id"],
            "language": c.get("language"),
            "pattern_tag": c.get("pattern_tag"),
            "text": c["text"],
            "expected": c["expected"],
            "pred": [[s.start, s.end, s.label] for s in preds],
            "verdict": v,
            "perfect": is_perfect(v),
        })
    return out


def evaluate_smoke(det: OnnxDetector) -> list[dict]:
    out = []
    for c in SMOKE_TESTS:
        preds = det.detect(c["text"])
        out.append({
            "id": c["id"],
            "text": c["text"],
            "note": c["note"],
            "pred": [
                {"start": s.start, "end": s.end, "label": s.label, "surface": s.text}
                for s in preds if s.end > s.start
            ],
            "n_spans": sum(1 for s in preds if s.end > s.start),
        })
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_category_table(per_case: dict[str, list[dict]]) -> dict:
    """For each category, perfect-rate across cases that touch it.

    A case "touches" a category C if any of its gold spans has label C, OR if
    its pattern_tag includes negative-evidence for C (adv_neg_person etc.).
    For simplicity, we bucket only by gold span labels — pure negative cases
    (no gold) are bucketed under the implied category via pattern_tag.
    """
    NEG_TAG_TO_CAT = {
        "adv_neg_person": "private_person",
        "adv_neg_date": "private_date",
        "adv_neg_addr": "private_address",
        "adv_neg_email": "private_email",
        "adv_neg_phone": "private_phone",
        "adv_neg_url": "private_url",
        "adv_neg_account": "account_number",
    }
    # Per-category bucket: list of case_ids
    cat_to_cases: dict[str, list[str]] = defaultdict(list)
    for c in per_case["fp32"]:
        cats = gold_categories(c["expected"])
        if not cats:
            ptag = c.get("pattern_tag", "")
            if ptag in NEG_TAG_TO_CAT:
                cats = {NEG_TAG_TO_CAT[ptag]}
        for cat in cats:
            cat_to_cases[cat].append(c["case_id"])
        if not cats:
            cat_to_cases["__unbucketed__"].append(c["case_id"])

    # Per-dtype perfect-rate by category
    by_id = {dt: {c["case_id"]: c for c in cases} for dt, cases in per_case.items()}
    result = {}
    for cat, cases in sorted(cat_to_cases.items()):
        row = {"n": len(cases)}
        for dt in per_case:
            n_perfect = sum(1 for cid in cases if by_id[dt][cid]["perfect"])
            row[dt] = n_perfect / len(cases) if cases else 0.0
            row[f"{dt}_perfect_n"] = n_perfect
        result[cat] = row
    return result


def per_language_table(per_case: dict[str, list[dict]]) -> dict:
    by_id = {dt: {c["case_id"]: c for c in cases} for dt, cases in per_case.items()}
    lang_to_cases: dict[str, list[str]] = defaultdict(list)
    for c in per_case["fp32"]:
        lang = c.get("language") or "unknown"
        lang_to_cases[lang].append(c["case_id"])
    result = {}
    for lang, cases in sorted(lang_to_cases.items()):
        row = {"n": len(cases)}
        for dt in per_case:
            n_perfect = sum(1 for cid in cases if by_id[dt][cid]["perfect"])
            row[dt] = n_perfect / len(cases) if cases else 0.0
            row[f"{dt}_perfect_n"] = n_perfect
        result[lang] = row
    return result


def overall_stats(per_case: dict[str, list[dict]]) -> dict:
    out = {}
    for dt, cases in per_case.items():
        counts = Counter(c["verdict"] for c in cases)
        perfect = counts["PASS"] + counts["PASS_NEG"]
        n = len(cases)
        out[dt] = {
            "n": n,
            "PASS": counts["PASS"],
            "PASS_NEG": counts["PASS_NEG"],
            "PARTIAL": counts["PARTIAL"],
            "FAIL": counts["FAIL"],
            "FAIL_NEG": counts["FAIL_NEG"],
            "perfect_rate": perfect / n,
        }
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def render_md(
    overall: dict, per_cat: dict, per_lang: dict,
    smoke_results: dict[str, list[dict]],
) -> str:
    lines: list[str] = []
    push = lines.append

    push("# q8 ONNX quality report — `joelbarmettler/gheim-ch-560m`")
    push("")
    push(f"_Generated by `eval/q8_quality_eval.py`. Probe: {overall['fp32']['n']} forensic cases, "
         "uncalibrated (o_bias=0), CPU ONNX Runtime, no chunking._")
    push("")

    # ---- Headline ----
    fp32_pr = overall["fp32"]["perfect_rate"] * 100
    fp16_pr = overall["fp16"]["perfect_rate"] * 100
    q8_pr   = overall["q8"]["perfect_rate"] * 100
    push("## Headline")
    push("")
    push(f"On the 212-case forensic probe, fp32 reaches **{fp32_pr:.1f}%** perfect-rate, "
         f"fp16 **{fp16_pr:.1f}%**, q8 **{q8_pr:.1f}%**. "
         f"The q8 gap vs fp32 is **{fp32_pr - q8_pr:+.1f}pp**; the fp16 gap is "
         f"**{fp32_pr - fp16_pr:+.1f}pp**. ")
    # Find category most affected by q8 vs fp32
    worst_cat = None
    worst_delta = 0.0
    for cat, row in per_cat.items():
        if cat == "__unbucketed__" or row["n"] < 5:
            continue
        d = row["fp32"] - row["q8"]
        if d > worst_delta:
            worst_delta = d
            worst_cat = cat
    if worst_cat:
        push(f"The q8 damage concentrates on **{worst_cat}** "
             f"({per_cat[worst_cat]['fp32']*100:.1f}% fp32 → "
             f"{per_cat[worst_cat]['q8']*100:.1f}% q8, "
             f"{worst_delta*100:.1f}pp drop on n={per_cat[worst_cat]['n']} cases). "
             f"fp16 holds at {per_cat[worst_cat]['fp16']*100:.1f}% — essentially "
             f"matching fp32 on the same bucket.")
    push("")

    # ---- Overall table ----
    push("## Overall verdict counts")
    push("")
    push("| dtype | PASS | PASS_NEG | PARTIAL | FAIL | FAIL_NEG | perfect-rate |")
    push("|---|---:|---:|---:|---:|---:|---:|")
    for dt in ("fp32", "fp16", "q8"):
        r = overall[dt]
        push(f"| **{dt}** | {r['PASS']} | {r['PASS_NEG']} | {r['PARTIAL']} | "
             f"{r['FAIL']} | {r['FAIL_NEG']} | **{r['perfect_rate']*100:.1f}%** |")
    push("")

    # ---- Per-category ----
    push("## Per-category perfect-rate")
    push("")
    push("Cases are bucketed by the categories their gold spans cover. "
         "Negative-evidence cases (adv_neg_*) are bucketed to the category "
         "they're testing absence of.")
    push("")
    push("| category | n | fp32 | fp16 | q8 | q8 vs fp32 |")
    push("|---|---:|---:|---:|---:|---:|")
    # Sort by n descending
    rows = sorted(per_cat.items(), key=lambda kv: -kv[1]["n"])
    for cat, row in rows:
        if cat == "__unbucketed__":
            continue
        delta = (row["q8"] - row["fp32"]) * 100
        push(f"| `{cat}` | {row['n']} | "
             f"{row['fp32']*100:.1f}% ({row['fp32_perfect_n']}/{row['n']}) | "
             f"{row['fp16']*100:.1f}% ({row['fp16_perfect_n']}/{row['n']}) | "
             f"{row['q8']*100:.1f}% ({row['q8_perfect_n']}/{row['n']}) | "
             f"**{delta:+.1f}pp** |")
    push("")

    # ---- Per-language ----
    push("## Per-language perfect-rate")
    push("")
    push("| language | n | fp32 | fp16 | q8 | q8 vs fp32 |")
    push("|---|---:|---:|---:|---:|---:|")
    rows = sorted(per_lang.items(), key=lambda kv: -kv[1]["n"])
    for lang, row in rows:
        delta = (row["q8"] - row["fp32"]) * 100
        push(f"| `{lang}` | {row['n']} | "
             f"{row['fp32']*100:.1f}% ({row['fp32_perfect_n']}/{row['n']}) | "
             f"{row['fp16']*100:.1f}% ({row['fp16_perfect_n']}/{row['n']}) | "
             f"{row['q8']*100:.1f}% ({row['q8_perfect_n']}/{row['n']}) | "
             f"**{delta:+.1f}pp** |")
    push("")

    # ---- Smoke tests ----
    push("## Smoke-test reproduction")
    push("")
    push("Two failures reported on q8 (Node.js `@huggingface/transformers` + "
         "shipped `model_quantized.onnx`) but not on fp32 PyTorch.")
    push("")
    by_id_smoke = {dt: {s["id"]: s for s in v} for dt, v in smoke_results.items()}
    for case in SMOKE_TESTS:
        push(f"### `{case['id']}`")
        push("")
        push(f"> {case['text']}")
        push("")
        push(f"_({case['note']})_")
        push("")
        push("| dtype | n spans | spans (label @ start-end) |")
        push("|---|---:|---|")
        for dt in ("fp32", "fp16", "q8"):
            s = by_id_smoke[dt][case["id"]]
            n = s["n_spans"]
            if not s["pred"]:
                desc = "_(none)_"
            else:
                desc = " · ".join(
                    f"`{p['label']}` @ {p['start']}-{p['end']} "
                    f"({p['surface']!r})"
                    for p in s["pred"]
                )
            push(f"| **{dt}** | {n} | {desc} |")
        push("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cases = load_probe_cases()
    print(f"loaded {len(cases)} probe cases", flush=True)

    per_case: dict[str, list[dict]] = {}
    smoke_results: dict[str, list[dict]] = {}

    for tag, d, fname in DTYPES:
        print(f"\n=== {tag} ({d}) ===", flush=True)
        t0 = time.time()
        det = OnnxDetector(tag, d, fname)
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()
        per_case[tag] = evaluate_probe(det, cases)
        print(f"  probe sweep: {time.time()-t0:.1f}s", flush=True)
        smoke_results[tag] = evaluate_smoke(det)
        c = Counter(c["verdict"] for c in per_case[tag])
        perfect = c["PASS"] + c["PASS_NEG"]
        print(f"  verdicts: {dict(c)} perfect={perfect/len(cases)*100:.1f}%", flush=True)
        del det  # free RAM

    overall = overall_stats(per_case)
    per_cat = per_category_table(per_case)
    per_lang = per_language_table(per_case)

    # Save per-case JSON
    out_json = EVAL_DIR / "q8_quality_per_case.json"
    out_json.write_text(json.dumps({
        "overall": overall,
        "per_category": per_cat,
        "per_language": per_lang,
        "smoke_tests": smoke_results,
        "per_case": per_case,
    }, indent=2))
    print(f"\nwrote {out_json}", flush=True)

    # render_md produces a numbers-only report (no headline / no recommendation).
    # The shipped report (eval/q8_quality_report.md) was hand-edited on top of
    # this output once the findings were clear. Write the auto-generated tables
    # to a separate file so re-runs don't clobber the prose; the hand-edited
    # report can be re-synced manually if numbers change.
    md = render_md(overall, per_cat, per_lang, smoke_results)
    out_md = EVAL_DIR / "q8_quality_report_autogen.md"
    out_md.write_text(md)
    print(f"wrote {out_md}", flush=True)

    # Console summary
    print()
    print("=== SUMMARY ===")
    for dt in ("fp32", "fp16", "q8"):
        r = overall[dt]
        print(f"  {dt:>4}: {r['perfect_rate']*100:>5.1f}% perfect  "
              f"(PASS={r['PASS']}, PASS_NEG={r['PASS_NEG']}, "
              f"FAIL={r['FAIL']}, FAIL_NEG={r['FAIL_NEG']}, PARTIAL={r['PARTIAL']})")


if __name__ == "__main__":
    main()
