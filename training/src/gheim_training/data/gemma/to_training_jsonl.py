"""Convert publishable layer5v3.jsonl → trainer-ready JSONL.

The labeler output (``layer5v3.jsonl``) is structured for HuggingFace
dataset publication: each row has ``id``, ``subset``, ``source_dataset``,
``doc_id``, ``chunk_index_in_doc``, ``labeler``, ``prompt_version``,
``labeled_at``, plus spans with both ``value`` and offsets.

The trainer's ``Example`` schema is leaner: ``text, spans[start/end/label],
language, source, template_id, meta``. This script produces that shape
while preserving all the publishable provenance under ``meta`` so it
survives stratified-split + tokenisation. Negative-control rows (no
spans) are kept — they're load-bearing for FP suppression.

Run:
    uv run python -m gheim_training.data.gemma.to_training_jsonl \\
        --in data/layer5v3.jsonl --out data/layer5v3_train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..schema import Example, Language, Span, write_jsonl


def _convert(rec: dict) -> Example:
    spans = [
        Span(start=int(s["start"]), end=int(s["end"]), label=s["label"])
        for s in rec.get("spans", [])
    ]
    lang: Language = rec["language"]
    return Example(
        text=rec["text"],
        spans=spans,
        language=lang,
        # Use the existing literal "apertus" — closest schema slot for
        # "LLM-labeled real text". Distinguish via template_id below.
        source="apertus",
        template_id=f"layer5v3:{rec['subset']}",
        meta={
            "id": rec["id"],
            "subset": rec["subset"],
            "doc_id": rec["doc_id"],
            "chunk_index_in_doc": rec["chunk_index_in_doc"],
            "source_dataset": rec.get("source_dataset"),
            "labeler": rec.get("labeler"),
            "prompt_version": rec.get("prompt_version"),
            "labeled_at": rec.get("labeled_at"),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n_in = n_out = n_dropped = 0
    examples: list[Example] = []
    with args.inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            try:
                ex = _convert(rec)
                ex.validate_offsets()
                examples.append(ex)
                n_out += 1
            except (ValueError, KeyError) as exc:
                n_dropped += 1
                if n_dropped <= 3:
                    print(f"  drop record {rec.get('id')}: {exc}")

    n = write_jsonl(args.out, examples)
    print(f"\nread {n_in} records, wrote {n} examples → {args.out} "
          f"(dropped {n_dropped})")


if __name__ == "__main__":
    main()
