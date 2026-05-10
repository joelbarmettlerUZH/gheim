"""A1.1 PoC: hand-eyeball Gemma's labeling quality on 20 Swiss chunks.

Pulls 20 chunks from local apertus-pretrain-swiss parquet:
  - 5 entscheidsuche (court decisions)
  - 5 curia_vista (parliamentary records)
  - 5 fineweb-2 Swiss (web)
  - 5 romansh

Runs the GemmaLabeler with structured-output + verbatim verification.
Prints chunk + extracted spans side-by-side so you can eyeball precision
and recall before committing to a 50k-chunk run.

Run:
    uv run python -m gheim_training.data.gemma.poc_label_swiss
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from ..apertus_label.stream import Subset, _stream_local_parquet
from ..lang_detect import detect as detect_lang
from .labeler import GemmaLabeler

PARQUET_DIR = Path("data/raw/apertus-swiss")
SUBSETS: tuple[Subset, ...] = ("entscheidsuche", "curia_vista", "fineweb", "romansh")


def _grab(subset: Subset, n: int, max_skip: int = 5000) -> list[tuple[str, str]]:
    """Stream a subset from local parquet, return (chunk_text, doc_id) pairs."""
    out: list[tuple[str, str]] = []
    n_scanned = 0
    for chunk in _stream_local_parquet(
        PARQUET_DIR, subsets=(subset,),
        max_chunks=0, skip_first_n_records=0,
    ):
        n_scanned += 1
        if len(chunk.text) < 200 or len(chunk.text) > 1500:
            continue
        out.append((chunk.text, chunk.doc_id))
        if len(out) >= n:
            return out
        if n_scanned >= max_skip:
            return out
    return out


def main() -> None:
    print("=== A1.1: Gemma 4 26B-A4B-AWQ structured-output labeling PoC ===")
    print()
    print("Sampling 5 chunks from each of: entscheidsuche, curia_vista, "
          "fineweb, romansh…")
    chunks: list[tuple[str, str, str]] = []  # (text, doc_id, subset)
    for subset in SUBSETS:
        for text, did in _grab(subset, 5):
            chunks.append((text, did, subset))
        print(f"  {subset:<16} → got {sum(1 for c in chunks if c[2]==subset)}")
    print(f"Total: {len(chunks)} chunks")

    if not chunks:
        print("No chunks loaded — local parquet missing? Check data/raw/apertus-swiss/")
        return

    print()
    print("Detecting language for each chunk…")
    langs = [detect_lang(c[0], subset=c[2]) for c in chunks]
    by_lang: dict[str, int] = defaultdict(int)
    for la in langs:
        by_lang[la] += 1
    print(f"  {dict(by_lang)}")

    print()
    print("Loading Gemma 4 26B-A4B-AWQ via vLLM tp=2 (1-2 min cold start)…")
    labeler = GemmaLabeler()
    t0 = time.time()
    examples = labeler.label_batch(
        [c[0] for c in chunks],
        languages=langs,
        doc_ids=[c[1] for c in chunks],
        subset="poc",
    )
    elapsed = time.time() - t0
    print(f"  labeled {len(examples)} chunks in {elapsed:.1f}s "
          f"({len(examples)/elapsed:.2f} chunks/sec)")

    # Per-chunk side-by-side report
    n_with_pii = 0
    for i, (ex, (text, did, subset)) in enumerate(zip(examples, chunks, strict=True)):
        if ex is None:
            print(f"\n[{i:02d}] {subset:<14} {langs[i]:<5} {did}\n  TEXT: {text[:120]}…\n  SPANS: <None / no PII>")
            continue
        print(f"\n[{i:02d}] {subset:<14} {langs[i]:<5} {did}")
        print(f"  TEXT: {text[:200]}…")
        if not ex.spans:
            print("  SPANS: [empty — Gemma found no PII]")
        else:
            n_with_pii += 1
            for sp in ex.spans:
                surface = ex.text[sp.start:sp.end]
                print(f"  SPAN [{sp.label:<18}] {surface!r}")

    print()
    print("=== Summary ===")
    print(f"  chunks: {len(chunks)}")
    print(f"  with PII: {n_with_pii}")
    print(f"  without PII: {len(chunks) - n_with_pii}")
    total_spans = sum(len(e.spans) for e in examples if e is not None)
    print(f"  total spans: {total_spans}")
    print(f"  throughput: {len(chunks)/elapsed:.2f} chunks/sec → "
          f"50k chunks ≈ {50000/(len(chunks)/elapsed)/3600:.1f} hours on this hw")


if __name__ == "__main__":
    main()
