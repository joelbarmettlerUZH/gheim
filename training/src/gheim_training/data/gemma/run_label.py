"""A1.3: Production runner — Gemma-label N chunks from an unlabeled JSONL.

Reads the output of ``sample_apertus_swiss`` (or any JSONL with the same
schema), runs each chunk through the GemmaLabeler, and writes a labeled
JSONL with full provenance metadata for the publishable Swiss-PII dataset.

Resumability
------------
The job is checkpointed: each batch's results are flushed to disk before
moving to the next. On crash/restart with the same ``--out``, already-
labeled rows are skipped (we read existing output, build a set of seen
(``doc_id``, ``chunk_index_in_doc``) pairs, and skip those in the input).
This is critical for an 8h job — we don't want to lose 6h of labeling
because of one OOM or one bad chunk.

Output JSONL schema (each row, suitable for HuggingFace dataset push):
  {
    "id": <doc_id>"#"<chunk_index>,
    "text": <chunk text>,
    "language": <gheim Language>,
    "subset": <entscheidsuche/curia_vista/fineweb/romansh>,
    "source_dataset": "swiss-ai/apertus-pretrain-swiss",
    "doc_id": <original doc id>,
    "spans": [{"start": int, "end": int, "label": str, "value": str}, ...],
    "labeler": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    "prompt_version": <git SHA or fixed string>,
    "labeled_at": <ISO timestamp>,
  }

Negative chunks (Gemma found no PII) are STILL written, with empty spans —
they're valid training data (PII-sparse real Swiss prose). The ``meta``
fields let downstream filter them in/out.

Run:
    uv run python -m gheim_training.data.gemma.run_label \\
        --in data/layer5v3_unlabeled.jsonl \\
        --out data/layer5v3.jsonl \\
        --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..schema import Language
from .labeler import GemmaLabeler

LABELER_MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
PROMPT_VERSION = "v3-localized-2026-05-06"


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _seen_ids_in_existing_output(path: Path) -> set[str]:
    """Read an existing output JSONL (if present) and return the set of ids
    already labeled. Used to skip on resume."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    for rec in _load_jsonl(path):
        rid = rec.get("id")
        if isinstance(rid, str):
            seen.add(rid)
    return seen


def _make_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}#{chunk_index}"


def _flush(out_path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to the output JSONL atomically (per-line append is fine
    for the use case — we don't read it concurrently)."""
    with out_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="Input JSONL of unlabeled chunks (sample_apertus_swiss output).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL of labeled chunks. Resumable.")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Chunks per Gemma batch. vLLM continuous-batches across these.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after labeling this many new chunks (0=no limit).")
    args = ap.parse_args()

    seen_ids = _seen_ids_in_existing_output(args.out)
    if seen_ids:
        print(f"Resuming: {len(seen_ids)} chunks already in {args.out}; will skip.")

    # Stream input into batches, skipping already-labeled
    print(f"Reading unlabeled chunks from {args.inp}…")
    pending: list[dict[str, Any]] = []
    n_input_total = 0
    n_skipped = 0
    for rec in _load_jsonl(args.inp):
        n_input_total += 1
        rid = _make_id(rec["doc_id"], rec["chunk_index_in_doc"])
        if rid in seen_ids:
            n_skipped += 1
            continue
        rec["_id"] = rid  # cache for the batch loop
        pending.append(rec)
    print(f"  input total: {n_input_total}, already-labeled: {n_skipped}, "
          f"to-label: {len(pending)}")

    if not pending:
        print("Nothing to label.")
        return
    if args.limit > 0:
        pending = pending[:args.limit]
        print(f"  limited to {args.limit} new chunks for this run")

    # Initialize labeler (loads vLLM model — 1-2 min cold start)
    print("\nLoading Gemma 4 26B-A4B-AWQ via vLLM tp=2…")
    labeler = GemmaLabeler()
    t_load_0 = time.time()
    assert labeler.client is not None
    labeler.client._load()
    print(f"  loaded in {time.time() - t_load_0:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_processed = 0
    total_with_spans = 0
    total_spans = 0
    t_run_0 = time.time()

    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(n_batches):
        batch = pending[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        chunks_text = [r["text"] for r in batch]
        languages: list[Language] = [r["language"] for r in batch]
        doc_ids = [r["doc_id"] for r in batch]

        t_b0 = time.time()
        examples = labeler.label_batch(
            chunks_text,
            languages=languages,
            doc_ids=doc_ids,
            subset="layer5v3",
        )
        elapsed_batch = time.time() - t_b0

        rows: list[dict[str, Any]] = []
        labeled_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        for rec, ex in zip(batch, examples, strict=True):
            spans_payload = []
            if ex is not None:
                for sp in ex.spans:
                    spans_payload.append({
                        "start": sp.start,
                        "end": sp.end,
                        "label": sp.label,
                        "value": ex.text[sp.start:sp.end],
                    })
            row = {
                "id": rec["_id"],
                "text": rec["text"],
                "language": rec["language"],
                "subset": rec["subset"],
                "source_dataset": rec.get("source_dataset",
                                          "swiss-ai/apertus-pretrain-swiss"),
                "doc_id": rec["doc_id"],
                "chunk_index_in_doc": rec["chunk_index_in_doc"],
                "spans": spans_payload,
                "labeler": LABELER_MODEL_ID,
                "prompt_version": PROMPT_VERSION,
                "labeled_at": labeled_at,
            }
            rows.append(row)
            if spans_payload:
                total_with_spans += 1
            total_spans += len(spans_payload)

        _flush(args.out, rows)
        total_processed += len(rows)
        run_elapsed = time.time() - t_run_0
        rate = total_processed / max(run_elapsed, 1)
        remaining = len(pending) - total_processed
        eta_sec = remaining / max(rate, 0.01)
        print(
            f"[batch {batch_idx + 1}/{n_batches}] "
            f"+{len(rows)} (with-spans={sum(1 for r in rows if r['spans'])}) "
            f"in {elapsed_batch:.0f}s · "
            f"total={total_processed}/{len(pending)} "
            f"({rate:.2f} chunks/s, ETA {eta_sec/60:.1f} min)",
            flush=True,
        )

    total_elapsed = time.time() - t_run_0
    print(f"\nDone. Labeled {total_processed} chunks in {total_elapsed:.0f}s "
          f"({total_processed / max(total_elapsed, 1):.2f} chunks/s).")
    print(f"  with spans: {total_with_spans} ({total_with_spans/max(total_processed,1):.0%})")
    print(f"  total spans: {total_spans}")
    print(f"  output → {args.out}")


if __name__ == "__main__":
    main()
