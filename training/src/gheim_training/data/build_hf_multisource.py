"""Build the multi-source HF DatasetDict for the cross-domain
transfer experiment.

Reads:
  - data/balanced_train.jsonl              (in-domain gheim train)
  - data/layer_external_train.jsonl        (openpii + WikiNeural + CoNLL train)
  - data/balanced_val.jsonl                (eval split, unchanged)
  - data/balanced_test.jsonl               (eval split, unchanged)

Writes ``data/built_multisource/`` with train concatenating both
sources, and val/test pointing at the same eval material so numbers
are directly comparable to the in-domain-only training run.

Run::

    uv run python -m gheim_training.data.build_hf_multisource
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_hf import _convert as _convert_indomain

IN_TRAIN_INDOMAIN = Path("data/balanced_train.jsonl")
IN_TRAIN_EXTERNAL = Path("data/layer_external_train.jsonl")
IN_VAL = Path("data/balanced_val.jsonl")
IN_TEST = Path("data/balanced_test.jsonl")
OUT_DIR = Path("data/built_multisource")


def _convert_external(rec: dict) -> dict:
    """Project an external-train record onto the same HF-friendly schema
    as the in-domain build, filling defaults for v3-only fields."""
    spans_out: list[dict] = []
    for sp in rec.get("spans", []):
        spans_out.append({
            "start": int(sp["start"]),
            "end": int(sp["end"]),
            "label": sp["label"],
            "value": rec["text"][int(sp["start"]):int(sp["end"])],
            "signals": ["external"],
            "confidence": 1.0,
            "regex_subtype": "",
        })
    subset = rec.get("source", "external")
    return {
        "id": rec.get("id", ""),
        "text": rec["text"],
        "language": rec.get("language", ""),
        "subset": subset,
        "source": subset,
        "doc_id": rec.get("id", ""),
        "spans": spans_out,
        "labelers": [subset],
        "build_pipeline": "external-train-mix-v1",
        "template_id": rec.get("template_id") or "",
    }


def _load(path: Path, convert) -> list[dict]:
    print(f"Reading: {path}", flush=True)
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(convert(json.loads(line)))
    print(f"  → {len(out):,} records")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict

    train_in  = _load(IN_TRAIN_INDOMAIN, _convert_indomain)
    train_ext = _load(IN_TRAIN_EXTERNAL, _convert_external)
    train = train_in + train_ext
    val   = _load(IN_VAL,  _convert_indomain)
    test  = _load(IN_TEST, _convert_indomain)

    print(f"\nTrain composition: in-domain={len(train_in):,}, external={len(train_ext):,}, "
          f"total={len(train):,}", flush=True)
    print("Building DatasetDict …", flush=True)
    dd = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })
    print(dd)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.out_dir))
    print(f"\nSaved DatasetDict → {args.out_dir}")


if __name__ == "__main__":
    main()
