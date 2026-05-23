"""V2-11a: Convert v2_balanced split JSONLs → HF DatasetDict.

Reads the three split files produced by ``split.py``:

- ``data/v2_balanced_train.jsonl``
- ``data/v2_balanced_val.jsonl``
- ``data/v2_balanced_test.jsonl``

and writes a HF DatasetDict at ``data/built_v2_balanced/`` with the
``train`` / ``validation`` / ``test`` keys (HF convention rename of
``val`` → ``validation``).

Schema preserved on disk
------------------------
Per-record fields kept:

- ``text``                — chunk text
- ``language``            — de_ch / fr_ch / it_ch / rm / en
- ``source``              — alias of ``subset`` for v1 train.py compat
- ``subset``              — v2 source tag (fineweb/curia_vista/entscheidsuche/
                            romansh/synthetic_l1/synthetic_l9)
- ``doc_id``              — source document id (or chunk id for synthetic)
- ``spans``               — list of v2 spans (full schema: start, end, label,
                            value, signals, confidence, regex_subtype)
- ``labelers``            — model identifiers that produced this chunk's labels
- ``build_pipeline``      — V2_PIPELINE_VERSION the chunk was built with
- ``template_id``         — synthetic template id (or empty string) for
                            v1 train.py compat

The full v2 schema flows through so the published dataset on HF Hub
has the same per-span provenance that drove the balancer + audit
decisions. The trainer's encoder projects spans to (start, end, label)
for BIOES alignment — the extra metadata is preserved but unused at
fit-time.

Run
---
    uv run python -m gheim_training.data.v2.build_hf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IN_TRAIN = Path("data/v2_balanced_train.jsonl")
IN_VAL = Path("data/v2_balanced_val.jsonl")
IN_TEST = Path("data/v2_balanced_test.jsonl")
OUT_DIR = Path("data/built_v2_balanced")


def _convert(rec: dict) -> dict:
    """Project a v2 JSONL record onto the HF-friendly schema. We retain
    every v2 field (spans get their full signals/confidence/regex_subtype
    triple) and also fill v1-compatible ``source`` + ``template_id``
    aliases so the existing train.py reader works without forking."""
    subset = rec.get("subset", "unknown") or "unknown"
    # Synthetic chunks carry template_id in meta; real chunks have None.
    meta = rec.get("meta") or {}
    template_id = meta.get("template_id") or ""
    spans_out: list[dict] = []
    for sp in rec.get("spans", []):
        spans_out.append({
            "start": int(sp["start"]),
            "end": int(sp["end"]),
            "label": sp["label"],
            "value": sp.get("value", ""),
            "signals": list(sp.get("signals", [])),
            "confidence": float(sp.get("confidence", 0.0)),
            "regex_subtype": sp.get("regex_subtype") or "",
        })
    return {
        "id": rec.get("id", ""),
        "text": rec["text"],
        "language": rec.get("language", ""),
        "subset": subset,
        "source": subset,  # v1 train.py reads "source"
        "doc_id": rec.get("doc_id") or rec.get("id", ""),
        "spans": spans_out,
        "labelers": list(rec.get("labelers", [])),
        "build_pipeline": rec.get("build_pipeline", ""),
        "template_id": template_id,
    }


def _load_jsonl_as_records(path: Path) -> list[dict]:
    print(f"Reading: {path}", flush=True)
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
    print("Building DatasetDict (auto-infer features) …", flush=True)
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
