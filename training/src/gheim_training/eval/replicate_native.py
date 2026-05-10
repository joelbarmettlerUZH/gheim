"""Replicate an external model's published numbers on its native dataset.

Used as a sanity check on the eval harness: if we can reproduce dslim's
0.91 F1 on CoNLL-2003 or Davlan's 0.84 on WikiNeural, we know our scoring
methodology is correct, and our cross-dataset numbers are honest.

This is intentionally minimal — no schema remap, no language filter, no
custom encoding. Just: load model, load dataset's test split (which already
has tokens + ner_tags in the model's schema), tokenize via the model's
tokenizer with word-id alignment, predict, score with seqeval.

Usage:
    uv run --no-sync --project training python -m gheim_training.eval.replicate_native \\
        --model dslim/bert-base-NER \\
        --dataset tomaarsen/conll2003 \\
        --split test \\
        --batch-size 32 \\
        --out eval/replication_dslim_conll.json

For datasets with multiple test splits (WikiNeural test_de/test_fr/...),
score each split separately by passing --splits 'test_de,test_fr,test_it,test_en'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default=None,
                        help="Single split name (e.g. 'test'). Mutually exclusive with --splits.")
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma-separated split list. Each is scored separately.")
    parser.add_argument("--tokens-col", type=str, default="tokens")
    parser.add_argument("--tags-col", type=str, default="ner_tags")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.split and not args.splits:
        raise SystemExit("pass --split or --splits")
    splits = [args.split] if args.split else [s.strip() for s in args.splits.split(",")]

    import numpy as np
    import torch
    from datasets import load_dataset, load_from_disk
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    def _load(name: str, split: str):
        # Local-path before Hub-id ordering: if a directory of the same name
        # exists in cwd we'd otherwise hit it first via load_dataset's resolver.
        from pathlib import Path
        if Path(name).exists():
            dd = load_from_disk(name)
            if split not in dd:
                raise SystemExit(f"split {split!r} not in {list(dd)}")
            return dd[split]
        return load_dataset(name, split=split)

    print(f"model:   {args.model}", flush=True)
    print(f"dataset: {args.dataset}", flush=True)
    print(f"splits:  {splits}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                              trust_remote_code=args.trust_remote_code)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    ).eval().cuda()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    results: dict[str, dict] = {}
    for split in splits:
        print(f"\n=== {split} ===", flush=True)
        ds = _load(args.dataset, split)
        print(f"  {len(ds):,} rows", flush=True)

        # Resolve tag-column int → str if needed. Both branches bind their
        # lookup table via a default-arg so the closure captures by value, not
        # by reference (the loop reassigns `names`/`wn` on the next split).
        feat = ds.features[args.tags_col]
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            names = feat.feature.names
            def resolve_tags(t_list, _names=names):
                return [_names[int(t)] for t in t_list]
        else:
            # WikiNeural stores ints with no ClassLabel; fall back to the
            # canonical PER/ORG/LOC/MISC mapping.
            wn = {0: "O",
                  1: "B-PER", 2: "I-PER",
                  3: "B-ORG", 4: "I-ORG",
                  5: "B-LOC", 6: "I-LOC",
                  7: "B-MISC", 8: "I-MISC"}
            def resolve_tags(t_list, _wn=wn):
                return [_wn[int(t)] if isinstance(t, int) else t for t in t_list]

        true_per_row: list[list[str]] = []
        pred_per_row: list[list[str]] = []

        n = len(ds)
        bsz = args.batch_size
        import time
        t0 = time.time()
        for batch_start in range(0, n, bsz):
            rows = ds.select(range(batch_start, min(batch_start + bsz, n)))
            batch_tokens = [list(r[args.tokens_col]) for r in rows]
            batch_tags = [resolve_tags(r[args.tags_col]) for r in rows]

            enc = tokenizer(
                batch_tokens,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"].cuda(),
                    attention_mask=enc["attention_mask"].cuda(),
                ).logits.cpu().numpy()
            preds = np.argmax(logits, axis=-1)

            for i in range(len(batch_tokens)):
                word_ids = enc.word_ids(batch_index=i)
                t_seq: list[str] = []
                p_seq: list[str] = []
                seen_word: set[int] = set()
                for tok_idx, w_id in enumerate(word_ids):
                    # word_id == None marks special tokens (CLS/SEP/PAD); the
                    # `seen_word` guard takes only the first subword per word
                    # so the per-word label sequence stays aligned with batch_tags.
                    if w_id is None or w_id in seen_word:
                        continue
                    seen_word.add(w_id)
                    if w_id < len(batch_tags[i]):
                        t_seq.append(batch_tags[i][w_id])
                        p_seq.append(id2label.get(int(preds[i, tok_idx]), "O"))
                true_per_row.append(t_seq)
                pred_per_row.append(p_seq)

            done = batch_start + len(rows)
            if done % (bsz * 25) == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (n - done) / rate if rate else 0
                print(f"  {done:>5}/{n} · {rate:5.1f} rows/s · eta {eta:5.0f}s", flush=True)

        ov_f1 = float(f1_score(true_per_row, pred_per_row))
        ov_p = float(precision_score(true_per_row, pred_per_row))
        ov_r = float(recall_score(true_per_row, pred_per_row))
        print(f"\n{split} OVERALL: F1={ov_f1:.4f}  P={ov_p:.4f}  R={ov_r:.4f}", flush=True)

        rep = classification_report(
            true_per_row, pred_per_row, output_dict=True, zero_division=0
        )
        per_cat: dict[str, dict[str, float | int]] = {}
        print("Per-category:", flush=True)
        for ent, sc in rep.items():
            if not isinstance(sc, dict):
                continue
            if ent in ("micro avg", "macro avg", "weighted avg"):
                continue
            per_cat[ent] = {
                "precision": float(sc.get("precision", 0)),
                "recall": float(sc.get("recall", 0)),
                "f1": float(sc.get("f1-score", 0)),
                "support": int(sc.get("support", 0)),
            }
            print(f"  {ent:<10} F1={per_cat[ent]['f1']:.4f}  "
                  f"P={per_cat[ent]['precision']:.4f}  "
                  f"R={per_cat[ent]['recall']:.4f}  n={per_cat[ent]['support']}",
                  flush=True)

        results[split] = {
            "n_rows": n,
            "overall": {"f1": ov_f1, "precision": ov_p, "recall": ov_r},
            "per_category": per_cat,
        }

    if args.out:
        out_blob = {
            "model": args.model,
            "dataset": args.dataset,
            "splits": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_blob, indent=2))
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
