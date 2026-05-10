"""Score any of {HF transformers, ONNX, spaCy, Presidio} on our test split.

Replaces the four legacy entry points (``test_eval``, ``onnx_test_eval``,
``spacy_test_eval``, ``presidio_test_eval``) with one CLI. The unified
JSON shape it emits is consumed by :mod:`report` together with the
output of :mod:`eval_on_external`.

JSON shape::

    {
      "model_id": "joelbarmettler/gheim-ch-560m",
      "backend": "hf",
      "dataset_id": "data/built_v2",
      "dataset_split": "test",
      "n_chunks": 15861,
      "metrics": {
        "overall": {
          "strict_span": {"f1": ..., "precision": ..., "recall": ...,
                            "method": "seqeval-token-level"},
          "char":         {"f1": ..., "precision": ..., "recall": ...,
                            "method": "char-label-aware"}
        },
        "per_category": {
          "<cat>": {
            "strict_span": {"f1": ..., "precision": ..., "recall": ..., "n_gold": <int>},
            "char":         {"f1": ..., "precision": ..., "recall": ...}
          }
        },
        "per_language": {
          "<lang>": {
            "strict_span": {"f1": ..., "precision": ..., "recall": ..., "method": ...},
            "char":         {"f1": ..., "precision": ..., "recall": ..., "method": ...},
            "n_chunks": <int>
          }
        }
      }
    }

The dataset is hard-coded to ``data/built_v2`` split ``test`` because
this script's purpose is "score X on our test set", not a general HF
dataset evaluator. Use ``--out`` to control the JSON filename.

Backends:

* ``hf``       — ``AutoModelForTokenClassification`` (the default; the
  same code path that produced the headline F1 = 0.9161).
* ``onnx``     — ``optimum.onnxruntime.ORTModelForTokenClassification``;
  used to score quantised exports.
* ``spacy``    — per-language spaCy NER pipelines (``de_core_news_lg``,
  ...). Span-emitting backend → ``strict_span`` uses ``char-set-strict``.
* ``presidio`` — Microsoft Presidio Analyzer with per-language NlpEngine.
  Same span-emitting treatment as spaCy.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._scoring import (
    CharSpan,
    ScoreAccumulator,
    build_pred_id_translation,
    gold_spans_from_row,
    load_test_dataset,
    merge_collapsed_spans,
    spans_from_label_sequence,
    write_result,
)

if TYPE_CHECKING:
    from datasets import Dataset

DEFAULT_DATASET = "data/built_v2"
DEFAULT_SPLIT = "test"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["hf", "onnx", "spacy", "presidio"], required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path (consumed by report.py).")

    # Common (HF / ONNX)
    p.add_argument("--model", type=str, default=None,
                   help="HF model id or local checkpoint dir (--backend hf).")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Pass trust_remote_code=True to AutoModel (needed for some HF models).")
    p.add_argument("--remap-json", type=Path, default=None,
                   help="JSON {their_label: our_label_or_O} for scoring an external "
                        "model whose label schema differs from ours.")
    p.add_argument("--languages", type=str, default=None,
                   help="Comma-separated language codes to keep (e.g. 'en' or 'en,de_ch').")

    # ONNX
    p.add_argument("--onnx-dir", type=Path, default=None,
                   help="Dir containing the ONNX file + tokenizer (--backend onnx).")
    p.add_argument("--onnx-file", type=str, default="model.onnx",
                   help="Filename of the ONNX model inside --onnx-dir.")
    p.add_argument("--provider", type=str, default="CPUExecutionProvider",
                   help="ORT provider: CPUExecutionProvider or CUDAExecutionProvider.")

    # spaCy
    p.add_argument("--spacy-models", nargs="+", default=None,
                   help="lang_code:spacy_model_id pairs, e.g. "
                        "'de:de_core_news_lg fr:fr_core_news_lg'.")
    p.add_argument("--remap", nargs="+", default=None,
                   help="spacy_label:our_label pairs for --backend spacy, e.g. "
                        "'PER:private_person LOC:O ORG:O MISC:O'.")

    # Presidio
    p.add_argument("--score-threshold", type=float, default=0.5,
                   help="Minimum confidence for a Presidio detection (--backend presidio).")
    return p


def _parse_languages(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    return [x.strip() for x in arg.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# HF / ONNX shared scoring loop
# ---------------------------------------------------------------------------

def _example_from_hf_record(rec: dict) -> Any:
    from ..data.schema import Example, Span
    return Example(
        text=rec["text"],
        spans=[Span(**s) for s in rec["spans"]],
        language=rec["language"],
        source=rec["source"],
        template_id=rec.get("template_id") or None,
    )


def _score_hf_or_onnx(
    *,
    raw: Dataset,
    tokenizer: Any,
    predict_logits: Any,  # callable: (input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray
    cfg_id2label: dict[int, str],
    args: argparse.Namespace,
) -> tuple[ScoreAccumulator, int]:
    """Run a token-classification model row-by-row and accumulate counters.

    The two backends (HF Trainer.predict and ONNX Runtime) plug in via
    ``predict_logits``; everything downstream — char-offsets, label
    translation, BIOES → span extraction, seqeval / char counters — is
    shared.
    """
    import numpy as np

    from ..data.label_space import ID2LABEL, LABEL2ID

    pred_translate = build_pred_id_translation(
        cfg_id2label, LABEL2ID, ID2LABEL, args.remap_json,
    )
    apply_merge = args.remap_json is not None

    acc = ScoreAccumulator(strict_method="seqeval-token-level")
    n = len(raw)
    bsz = args.batch_size
    t0 = time.time()
    for batch_start in range(0, n, bsz):
        rows = [raw[batch_start + i] for i in range(min(bsz, n - batch_start))]
        texts = [r["text"] for r in rows]
        enc = tokenizer(
            texts,
            return_offsets_mapping=True,
            padding=True, truncation=True,
            max_length=args.max_seq_length,
            return_tensors="np",
        )
        offsets_batch = enc.pop("offset_mapping")
        # Drop labels-related keys from the input — external models with
        # smaller num_labels CUDA-assert when computing internal CE loss.
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        logits = predict_logits(input_ids, attention_mask)
        preds = np.argmax(logits, axis=-1)

        for bi, row in enumerate(rows):
            offs = offsets_batch[bi].tolist()
            mask = attention_mask[bi].tolist()
            true_seq: list[str] = []
            pred_seq: list[str] = []
            char_offs: list[tuple[int, int]] = []
            # Build the gold token-level label sequence by re-running our
            # own char→token alignment on the gold spans, so seqeval sees
            # exactly the same token grid as the prediction.
            from ..data.bioes import IGNORE_INDEX, encode_example
            ex = _example_from_hf_record(row)
            enc_gold = encode_example(ex, tokenizer, max_length=args.max_seq_length)
            gold_label_ids = enc_gold["labels"]
            for tok_i in range(len(offs)):
                if mask[tok_i] == 0:
                    continue
                a, b = offs[tok_i]
                if a == 0 and b == 0:
                    # CLS / SEP / padding all carry (0, 0) offsets in fast tokenizers.
                    continue
                # Gold token label (gold_label_ids may be shorter than the
                # padded width if the row got truncated by the data
                # collator; treat missing positions as O).
                g_id = gold_label_ids[tok_i] if tok_i < len(gold_label_ids) else IGNORE_INDEX
                g_lab = "O" if g_id == IGNORE_INDEX else ID2LABEL[int(g_id)]
                p_lab = ID2LABEL[pred_translate.get(int(preds[bi, tok_i]), 0)]
                true_seq.append(g_lab)
                pred_seq.append(p_lab)
                char_offs.append((a, b))
            if apply_merge:
                pred_seq = merge_collapsed_spans(pred_seq)

            pred_spans = spans_from_label_sequence(pred_seq, char_offs)
            gold_spans: list[CharSpan] = gold_spans_from_row(row)

            acc.add_token_row(true_seq, pred_seq, gold_spans, pred_spans, ex.language)

        done = batch_start + len(rows)
        if done % (bsz * 25) == 0 or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            print(f"  {done:>6}/{n} · {rate:5.1f} chunks/s · elapsed {elapsed:5.0f}s",
                  flush=True)

    return acc, n


# ---------------------------------------------------------------------------
# Backend-specific drivers
# ---------------------------------------------------------------------------

def _run_hf(args: argparse.Namespace, raw: Dataset) -> tuple[ScoreAccumulator, int, str]:
    if not args.model:
        raise SystemExit("--backend hf requires --model")

    import numpy as np
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    print(f"Loading HF model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    cfg_id2label = {int(k): v for k, v in model.config.id2label.items()}

    def predict(input_ids: Any, attention_mask: Any) -> Any:
        with torch.no_grad():
            ids = torch.from_numpy(np.asarray(input_ids))
            am = torch.from_numpy(np.asarray(attention_mask))
            if torch.cuda.is_available():
                ids = ids.cuda()
                am = am.cuda()
            # .float() upcasts bf16/fp16 logits so np.argmax is well-defined.
            return model(input_ids=ids, attention_mask=am).logits.float().cpu().numpy()

    acc, n = _score_hf_or_onnx(
        raw=raw, tokenizer=tokenizer, predict_logits=predict,
        cfg_id2label=cfg_id2label, args=args,
    )
    return acc, n, args.model


def _run_onnx(args: argparse.Namespace, raw: Dataset) -> tuple[ScoreAccumulator, int, str]:
    if not args.onnx_dir:
        raise SystemExit("--backend onnx requires --onnx-dir")

    import numpy as np
    import torch
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer

    print(f"Loading ONNX: {args.onnx_dir}/{args.onnx_file} ({args.provider})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.onnx_dir), use_fast=True)
    model = ORTModelForTokenClassification.from_pretrained(
        str(args.onnx_dir),
        file_name=args.onnx_file,
        provider=args.provider,
    )
    cfg_id2label = {int(k): v for k, v in model.config.id2label.items()}

    def predict(input_ids: Any, attention_mask: Any) -> Any:
        out = model(
            input_ids=torch.from_numpy(np.asarray(input_ids)),
            attention_mask=torch.from_numpy(np.asarray(attention_mask)),
        )
        return out.logits.detach().cpu().numpy()

    acc, n = _score_hf_or_onnx(
        raw=raw, tokenizer=tokenizer, predict_logits=predict,
        cfg_id2label=cfg_id2label, args=args,
    )
    return acc, n, str(args.onnx_dir)


def _run_spacy(args: argparse.Namespace, raw: Dataset) -> tuple[ScoreAccumulator, int, str]:
    if not args.spacy_models or not args.remap:
        raise SystemExit("--backend spacy requires --spacy-models and --remap")

    import spacy

    lang_to_model: dict[str, str] = {}
    for spec in args.spacy_models:
        if ":" not in spec:
            raise SystemExit(f"--spacy-models entry must be lang:model, got {spec!r}")
        lang, model_id = spec.split(":", 1)
        lang_to_model[lang.strip()] = model_id.strip()
    remap: dict[str, str] = {}
    for spec in args.remap:
        if ":" not in spec:
            raise SystemExit(f"--remap entry must be SPACY_LABEL:our_label, got {spec!r}")
        k, v = spec.split(":", 1)
        remap[k.strip()] = v.strip()

    rows_by_lang: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(raw):
        lang_key = row["language"].split("_")[0]
        rows_by_lang[lang_key].append(i)
    print("  language histogram: " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(rows_by_lang.items())),
          flush=True)

    acc = ScoreAccumulator(strict_method="char-set-strict")
    total = 0
    for lang_key, idxs in sorted(rows_by_lang.items()):
        if lang_key not in lang_to_model:
            print(f"  skip language {lang_key}: no spaCy model configured", flush=True)
            # Still emit empty rows so n_chunks reflects everything we touched.
            continue
        model_id = lang_to_model[lang_key]
        print(f"\n=== {lang_key} ({len(idxs)} chunks) → {model_id} ===", flush=True)
        nlp = spacy.load(model_id, disable=["lemmatizer", "attribute_ruler"])
        nlp.max_length = 4_000_000
        t0 = time.time()
        for n, idx in enumerate(idxs, start=1):
            row = raw[idx]
            text = row["text"]
            gold_spans = gold_spans_from_row(row)
            doc = nlp(text)
            pred_spans: list[CharSpan] = []
            for ent in doc.ents:
                tgt = remap.get(ent.label_)
                if tgt is None or tgt == "O":
                    continue
                pred_spans.append((ent.start_char, ent.end_char, tgt))
            acc.add_span_row(gold_spans, pred_spans, row["language"])
            total += 1
            if n % 500 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n:>5}/{len(idxs)} · {rate:5.1f} chunks/s", flush=True)
        print(f"  done {lang_key} in {time.time() - t0:.0f}s", flush=True)
    return acc, total, "spacy:" + ",".join(f"{k}={v}" for k, v in sorted(lang_to_model.items()))


# Presidio entity → our 8-cat schema. Carried over verbatim from
# presidio_test_eval.py so re-runs are byte-comparable to the published
# numbers.
PRESIDIO_REMAP: dict[str, str] = {
    "PERSON":               "private_person",
    "EMAIL_ADDRESS":        "private_email",
    "PHONE_NUMBER":         "private_phone",
    "DATE_TIME":            "private_date",
    "URL":                  "private_url",
    "IP_ADDRESS":           "private_url",
    "DOMAIN_NAME":          "private_url",
    "IBAN_CODE":            "account_number",
    "CREDIT_CARD":          "account_number",
    "US_SSN":               "account_number",
    "US_BANK_NUMBER":       "account_number",
    "US_PASSPORT":          "account_number",
    "US_ITIN":               "account_number",
    "AU_ABN":               "account_number",
    "AU_ACN":               "account_number",
    "AU_TFN":               "account_number",
    "AU_MEDICARE":          "account_number",
    "UK_NHS":               "account_number",
    "UK_NINO":              "account_number",
    "ES_NIF":               "account_number",
    "IT_FISCAL_CODE":       "account_number",
    "IT_DRIVER_LICENSE":    "account_number",
    "IT_VAT_CODE":          "account_number",
    "IT_PASSPORT":          "account_number",
    "IT_IDENTITY_CARD":     "account_number",
    "PL_PESEL":             "account_number",
    "FI_PERSONAL_IDENTITY_CODE": "account_number",
    "IN_PAN":               "account_number",
    "IN_AADHAAR":           "account_number",
    "IN_VEHICLE_REGISTRATION": "account_number",
    "IN_VOTER":             "account_number",
    "IN_PASSPORT":          "account_number",
    "SG_NRIC_FIN":          "account_number",
    "SG_UEN":               "account_number",
    "MEDICAL_LICENSE":      "secret",
    "CRYPTO":               "account_number",
    "US_DRIVER_LICENSE":    "secret",
    "NRP":                  "O",
    "LOCATION":             "O",
    "ORGANIZATION":         "O",
}

PRESIDIO_SPACY_MODELS = {
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "it": "it_core_news_lg",
    "en": "en_core_web_lg",
}


def _run_presidio(args: argparse.Namespace, raw: Dataset) -> tuple[ScoreAccumulator, int, str]:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    print("Loading spaCy models for de/fr/it/en ...", flush=True)
    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": k, "model_name": v} for k, v in PRESIDIO_SPACY_MODELS.items()],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=list(PRESIDIO_SPACY_MODELS),
    )
    print("Presidio ready", flush=True)

    def analyzer_lang(row_lang: str) -> str:
        # rm and en-variant rows fall back to the English analyzer
        # (Presidio has no rm model).
        if row_lang.startswith("de"):
            return "de"
        if row_lang.startswith("fr"):
            return "fr"
        if row_lang.startswith("it"):
            return "it"
        return "en"

    acc = ScoreAccumulator(strict_method="char-set-strict")
    unmapped: set[str] = set()
    n_total = len(raw)
    t0 = time.time()
    for n, row in enumerate(raw, start=1):
        text = row["text"]
        gold_spans = gold_spans_from_row(row)
        analyzer_results = analyzer.analyze(text=text, language=analyzer_lang(row["language"]))
        pred_spans: list[CharSpan] = []
        for r in analyzer_results:
            if r.score < args.score_threshold:
                continue
            tgt = PRESIDIO_REMAP.get(r.entity_type)
            if tgt is None:
                unmapped.add(r.entity_type)
                continue
            if tgt == "O":
                continue
            pred_spans.append((r.start, r.end, tgt))
        acc.add_span_row(gold_spans, pred_spans, row["language"])
        if n % 250 == 0 or n == n_total:
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed else 0
            print(f"  {n:>6}/{n_total} · {rate:5.1f} chunks/s · elapsed {elapsed:5.0f}s",
                  flush=True)
    if unmapped:
        print(f"\n  unmapped Presidio entity types (treated as O): {sorted(unmapped)}",
              flush=True)
    return acc, n_total, "microsoft/presidio-analyzer"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    languages = _parse_languages(args.languages)

    print(f"Loading dataset: {DEFAULT_DATASET} (split={DEFAULT_SPLIT})", flush=True)
    raw = load_test_dataset(DEFAULT_DATASET, DEFAULT_SPLIT, languages)
    print(f"  {len(raw):,} chunks", flush=True)

    extra: dict[str, Any] = {}
    if args.backend == "hf":
        acc, n_chunks, model_id = _run_hf(args, raw)
    elif args.backend == "onnx":
        acc, n_chunks, model_id = _run_onnx(args, raw)
        extra["onnx_file"] = args.onnx_file
        extra["provider"] = args.provider
    elif args.backend == "spacy":
        acc, n_chunks, model_id = _run_spacy(args, raw)
    elif args.backend == "presidio":
        acc, n_chunks, model_id = _run_presidio(args, raw)
        extra["score_threshold"] = args.score_threshold
    else:  # pragma: no cover — argparse already constrains the choices.
        raise SystemExit(f"unknown backend {args.backend!r}")

    metrics = acc.finalize()
    write_result(
        args.out,
        model_id=model_id,
        backend=args.backend,
        dataset_id=DEFAULT_DATASET,
        dataset_split=DEFAULT_SPLIT,
        n_chunks=n_chunks,
        metrics=metrics,
        extra=extra,
    )

    ov = metrics["overall"]
    print()
    print(f"OVERALL  strict_span F1={ov['strict_span']['f1']:.4f}  "
          f"char F1={ov['char']['f1']:.4f}", flush=True)
    print()
    print("PER-LANGUAGE:")
    for lang, cell in sorted(metrics["per_language"].items()):
        print(f"  {lang:<8} strict={cell['strict_span']['f1']:.4f}  "
              f"char={cell['char']['f1']:.4f}  n={cell['n_chunks']}")
    print()
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
