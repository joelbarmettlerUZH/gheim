"""Compute a (language × category) char-level F1 matrix for a fine-tuned
HF token-classification checkpoint on a local HF dataset split.

Output JSON has the shape::

    {
      "model_id": "...",
      "dataset_split": "...",
      "n_chunks": int,
      "languages": [...],
      "categories": [...],
      "n_gold": {lang: {cat: int}},
      "n_chunks_per_lang": {lang: int},
      "f1": {lang: {cat: float | null}},
      "precision": {lang: {cat: float | null}},
      "recall": {lang: {cat: float | null}}
    }

A cell is ``null`` when there are no gold and no predicted spans of
that (lang, cat) — F1 is undefined and the paper renders ``n/a``.

Usage::

    uv run python -m gheim_training.eval.per_lang_cat \\
        --model checkpoints/stage2_xlmr \\
        --dataset-dir data/built_v2 --split test \\
        --out eval/per_lang_cat_gheim.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ._scoring import (
    char_level_counters,
    gold_spans_from_row,
    load_test_dataset,
    merge_collapsed_spans,
    spans_from_label_sequence,
    build_pred_id_translation,
)
from ._metrics import f1_pr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--remap-json", type=Path, default=None)
    args = p.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from ..data.bioes import IGNORE_INDEX, encode_example
    from ..data.label_space import ID2LABEL, LABEL2ID
    from ..data.schema import Example, Span

    print(f"Loading model: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    cfg_id2label = {int(k): v for k, v in model.config.id2label.items()}
    pred_translate = build_pred_id_translation(
        cfg_id2label, LABEL2ID, ID2LABEL, args.remap_json,
    )
    apply_merge = args.remap_json is not None

    raw = load_test_dataset(args.dataset_dir, args.split)
    n = len(raw)
    print(f"Scoring {n} chunks from {args.dataset_dir}/{args.split}", flush=True)

    counters: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    )
    n_gold: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_chunks_per_lang: dict[str, int] = defaultdict(int)

    bsz = args.batch_size
    t0 = time.time()
    for batch_start in range(0, n, bsz):
        rows = [raw[batch_start + i] for i in range(min(bsz, n - batch_start))]
        texts = [r["text"] for r in rows]
        enc = tok(texts, return_offsets_mapping=True, padding=True,
                  truncation=True, max_length=args.max_seq_length,
                  return_tensors="np")
        offsets_batch = enc.pop("offset_mapping")
        with torch.no_grad():
            ids = torch.from_numpy(np.asarray(enc["input_ids"]))
            am = torch.from_numpy(np.asarray(enc["attention_mask"]))
            if torch.cuda.is_available():
                ids = ids.cuda()
                am = am.cuda()
            logits = model(input_ids=ids, attention_mask=am).logits.float().cpu().numpy()
        preds = np.argmax(logits, axis=-1)

        for bi, row in enumerate(rows):
            offs = offsets_batch[bi].tolist()
            mask = enc["attention_mask"][bi].tolist()
            ex = Example(
                text=row["text"],
                spans=[Span(**s) for s in row["spans"]],
                language=row["language"],
                source=row["source"],
                template_id=row.get("template_id") or None,
            )
            enc_gold = encode_example(ex, tok, max_length=args.max_seq_length)
            gold_label_ids = enc_gold["labels"]
            pred_seq: list[str] = []
            char_offs: list[tuple[int, int]] = []
            for tok_i in range(len(offs)):
                if mask[tok_i] == 0:
                    continue
                a, b = offs[tok_i]
                if a == 0 and b == 0:
                    continue
                _ = (gold_label_ids[tok_i] if tok_i < len(gold_label_ids) else IGNORE_INDEX)
                p_lab = ID2LABEL[pred_translate.get(int(preds[bi, tok_i]), 0)]
                pred_seq.append(p_lab)
                char_offs.append((a, b))
            if apply_merge:
                pred_seq = merge_collapsed_spans(pred_seq)

            pred_spans = spans_from_label_sequence(pred_seq, char_offs)
            gold_spans = gold_spans_from_row(row)

            _, _, _, per_cat = char_level_counters(gold_spans, pred_spans)
            lang = ex.language
            for cat, c in per_cat.items():
                counters[lang][cat]["tp"] += c["tp"]
                counters[lang][cat]["fp"] += c["fp"]
                counters[lang][cat]["fn"] += c["fn"]
            for _s, _e, lab in gold_spans:
                n_gold[lang][lab] += 1
            n_chunks_per_lang[lang] += 1

        done = batch_start + len(rows)
        if done % (bsz * 25) == 0 or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            print(f"  {done:>6}/{n} · {rate:5.1f} chunks/s · elapsed {elapsed:5.0f}s",
                  flush=True)

    languages = sorted(counters)
    categories = sorted({c for lang in counters for c in counters[lang]} |
                        {c for lang in n_gold for c in n_gold[lang]})

    f1: dict[str, dict[str, float | None]] = {}
    pr: dict[str, dict[str, float | None]] = {}
    rc: dict[str, dict[str, float | None]] = {}
    for lang in languages:
        f1[lang] = {}
        pr[lang] = {}
        rc[lang] = {}
        for cat in categories:
            cell = counters[lang].get(cat, {"tp": 0, "fp": 0, "fn": 0})
            no_gold = (cell["tp"] + cell["fn"]) == 0
            no_pred = (cell["tp"] + cell["fp"]) == 0
            if no_gold and no_pred:
                f1[lang][cat] = None
                pr[lang][cat] = None
                rc[lang][cat] = None
            else:
                m = f1_pr(cell)
                f1[lang][cat] = m["f1"]
                pr[lang][cat] = m["precision"]
                rc[lang][cat] = m["recall"]

    blob: dict[str, Any] = {
        "model_id": args.model,
        "dataset_id": str(args.dataset_dir),
        "dataset_split": args.split,
        "n_chunks": n,
        "languages": languages,
        "categories": categories,
        "n_chunks_per_lang": dict(n_chunks_per_lang),
        "n_gold": {lang: dict(n_gold[lang]) for lang in languages},
        "f1": f1,
        "precision": pr,
        "recall": rc,
        "metric": "char-label-aware",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2))
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
