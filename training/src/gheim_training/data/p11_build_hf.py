"""P11: Convert the gheim-pii JSONL splits + train_mix into a HF DatasetDict.

Reads:
  data/gheim_train_mix.jsonl                   (train: real + synthetic + AI4P)
  data/gheim_pii_balanced_validation.jsonl     (val:   real only, chmod 444)
  data/gheim_pii_balanced_test.jsonl           (test:  real only, chmod 444)

Writes:
  data/built_v2/                  (HF DatasetDict, load_from_disk-able)
    train/        — 156,041
    validation/   — 14,634
    test/         — 14,661

Schema mapping (gheim-pii → trainer's Example):
  text             → text
  spans[i].start/end/label → spans[i].start/end/label  (drop value, source)
  language         → language
  subset           → source
  synthetic_template (or None) → template_id

This removes per-record metadata the trainer doesn't use, keeping the on-disk
dataset compact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IN_TRAIN = Path("data/gheim_train_mix.jsonl")
IN_VAL = Path("data/gheim_pii_balanced_validation.jsonl")
IN_TEST = Path("data/gheim_pii_balanced_test.jsonl")
OUT_DIR = Path("data/built_v2")


def _convert(rec: dict) -> dict:
    return {
        "text": rec["text"],
        "spans": [
            {"start": int(s["start"]), "end": int(s["end"]), "label": s["label"]}
            for s in rec.get("spans", [])
        ],
        "language": rec.get("language", ""),
        "source": rec.get("subset", "unknown") or "unknown",
        "template_id": rec.get("synthetic_template") or "",
    }


def _load_jsonl_as_records(path: Path) -> list[dict]:
    print(f"Reading: {path}", flush=True)
    out = []
    with path.open() as f:
        for line in f:
            out.append(_convert(json.loads(line)))
    print(f"  → {len(out):,} records")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from datasets import Dataset, DatasetDict

    train = _load_jsonl_as_records(IN_TRAIN)
    val = _load_jsonl_as_records(IN_VAL)
    test = _load_jsonl_as_records(IN_TEST)

    print()
    print("Building DatasetDict (auto-infer features) ...", flush=True)
    dd = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })
    print(dd)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.out_dir))
    print()
    print(f"Saved DatasetDict → {args.out_dir}")


if __name__ == "__main__":
    main()
