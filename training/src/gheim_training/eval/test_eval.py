"""Test-set evaluation for a saved gheim_training checkpoint.

Loads a saved HF checkpoint, evaluates on a chosen split of an HF
DatasetDict, reports overall + per-category + per-language F1, and
optionally writes the metrics to JSON for the model card.

Run:
    uv run --project training python -m gheim_training.eval.test_eval \\
        --checkpoint ./checkpoints/stage2_xlmr \\
        --dataset-dir ./data/built_v2 \\
        --split test \\
        --out eval/stage2_xlmr_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.bioes import IGNORE_INDEX, encode_example
from ..data.label_space import ID2LABEL
from ..data.schema import Example, Span


def _example_from_hf_record(rec: dict) -> Example:
    return Example(
        text=rec["text"],
        spans=[Span(**s) for s in rec["spans"]],
        language=rec["language"],
        source=rec["source"],
        template_id=rec.get("template_id") or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to saved checkpoint directory (HF model)")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="HF DatasetDict directory")
    parser.add_argument("--split", type=str, default="test",
                        help="Split name to evaluate on (default: test)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional JSON output path for the model card")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from datasets import load_from_disk
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(args.checkpoint))
    model.eval()

    print(f"Loading dataset: {args.dataset_dir} (split={args.split})", flush=True)
    dd = load_from_disk(str(args.dataset_dir))
    if args.split not in dd:
        raise SystemExit(f"split {args.split!r} not in {list(dd)}")
    raw_split = dd[args.split]
    print(f"  {len(raw_split):,} chunks")

    def _encode_batch(records: dict[str, list]) -> dict[str, list]:
        out = {"input_ids": [], "attention_mask": [], "labels": [], "_lang": []}
        for i in range(len(records["text"])):
            ex = _example_from_hf_record({k: records[k][i] for k in records})
            enc = encode_example(ex, tokenizer, max_length=args.max_seq_length)
            out["input_ids"].append(enc["input_ids"])
            out["attention_mask"].append(enc["attention_mask"])
            out["labels"].append(enc["labels"])
            out["_lang"].append(ex.language)
        return out

    encoded = raw_split.map(
        _encode_batch, batched=True, batch_size=64,
        remove_columns=raw_split.column_names, desc="tokenize+align",
    )
    langs_per_row = encoded["_lang"]
    encoded = encoded.remove_columns(["_lang"])

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=IGNORE_INDEX)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/eval_only",
            per_device_eval_batch_size=args.batch_size,
            report_to=[],
            bf16=False,
        ),
        data_collator=collator,
    )

    print("Running prediction ...", flush=True)
    pred = trainer.predict(encoded)
    logits = pred.predictions
    label_ids = pred.label_ids
    preds = np.argmax(logits, axis=-1)

    # Build per-row label sequences
    true_per_row: list[list[str]] = []
    pred_per_row: list[list[str]] = []
    for p_row, l_row in zip(preds, label_ids, strict=True):
        t: list[str] = []
        p: list[str] = []
        for p_id, l_id in zip(p_row, l_row, strict=True):
            if l_id == IGNORE_INDEX:
                continue
            t.append(ID2LABEL[int(l_id)])
            p.append(ID2LABEL[int(p_id)])
        true_per_row.append(t)
        pred_per_row.append(p)

    # Overall metrics
    overall_f1 = float(f1_score(true_per_row, pred_per_row))
    overall_p = float(precision_score(true_per_row, pred_per_row))
    overall_r = float(recall_score(true_per_row, pred_per_row))
    print()
    print(f"OVERALL: F1={overall_f1:.4f}  P={overall_p:.4f}  R={overall_r:.4f}")

    # Per-category from seqeval classification_report
    rep = classification_report(true_per_row, pred_per_row, output_dict=True, zero_division=0)
    print()
    print("Per-category:")
    per_cat: dict[str, dict[str, float]] = {}
    for ent, sc in rep.items():
        if not isinstance(sc, dict):
            continue
        if ent in ("micro avg", "macro avg", "weighted avg"):
            continue
        per_cat[ent] = {
            "precision": float(sc.get("precision", 0.0)),
            "recall": float(sc.get("recall", 0.0)),
            "f1": float(sc.get("f1-score", 0.0)),
            "support": int(sc.get("support", 0)),
        }
        print(f"  {ent:<22} F1={per_cat[ent]['f1']:.4f}  P={per_cat[ent]['precision']:.4f}  "
              f"R={per_cat[ent]['recall']:.4f}  n={per_cat[ent]['support']}")

    # Per-language: rerun seqeval restricted to rows of each language
    print()
    print("Per-language F1:")
    per_lang_f1: dict[str, dict] = {}
    for lang in sorted(set(langs_per_row)):
        idx = [i for i, la in enumerate(langs_per_row) if la == lang]
        if not idx:
            continue
        t_sub = [true_per_row[i] for i in idx]
        p_sub = [pred_per_row[i] for i in idx]
        f = float(f1_score(t_sub, p_sub))
        pr = float(precision_score(t_sub, p_sub))
        re = float(recall_score(t_sub, p_sub))
        per_lang_f1[lang] = {"f1": f, "precision": pr, "recall": re, "n_chunks": len(idx)}
        print(f"  {lang:<8} F1={f:.4f}  P={pr:.4f}  R={re:.4f}  n={len(idx)}")

    out_blob = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "n_chunks": len(raw_split),
        "overall": {"f1": overall_f1, "precision": overall_p, "recall": overall_r},
        "per_category": per_cat,
        "per_language": per_lang_f1,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_blob, indent=2))
        print()
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
