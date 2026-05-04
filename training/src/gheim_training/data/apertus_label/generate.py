"""Layer 5 CLI: stream → prefilter → label → verify → JSONL.

Run a small proof-of-concept (1k chunks):
    uv run python -m gheim_training.data.apertus_label.generate \\
        --max-chunks 1000 --batch-size 32 --out data/layer5_smoke.jsonl

Full run (uses much more compute):
    uv run python -m gheim_training.data.apertus_label.generate \\
        --max-chunks 50000 --batch-size 64 --out data/layer5.jsonl
"""
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from ..schema import Example, write_jsonl
from .label import ApertusLabeler
from .prefilter import filter_chunks
from .stream import Chunk, stream_chunks
from .verify import verify_and_combine


def _detect_language(chunk_text: str, subset: str) -> str:
    """Cheap heuristic for the (language, source) split key.

    Court decisions and parliamentary records are mostly DE; FR/IT corners
    exist. We score the chunk for a few high-frequency indicator words and
    pick the winner. Misclassifications are tolerated (training mix is
    aggregated across all languages anyway).
    """
    score = {"de_ch": 0, "fr_ch": 0, "it_ch": 0}
    t = " " + chunk_text.lower() + " "
    for w in (" und ", " der ", " die ", " ist ", " nicht ", " sich ", " auch "):
        if w in t:
            score["de_ch"] += 1
    for w in (" et ", " le ", " la ", " est ", " pas ", " que ", " avec "):
        if w in t:
            score["fr_ch"] += 1
    for w in (" e ", " il ", " la ", " è ", " che ", " con ", " del "):
        if w in t:
            score["it_ch"] += 1
    return max(score.items(), key=lambda kv: kv[1])[0]


def _batched(it, n: int):
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def run(
    *,
    out_path: Path,
    max_chunks: int,
    batch_size: int,
    subsets: tuple[str, ...] = ("entscheidsuche", "curia_vista"),
    min_structured_hits: int = 1,
    max_chunk_chars: int = 1500,
    local_parquet_dir: Path | None = None,
) -> int:
    src = f"local={local_parquet_dir}" if local_parquet_dir else "HF stream"
    print(f"Streaming chunks from {src}, subsets={subsets}, min_structured_hits={min_structured_hits}")
    raw_stream = stream_chunks(
        subsets=subsets, max_chunks=0, local_parquet_dir=local_parquet_dir,
    )
    filtered = filter_chunks(raw_stream, min_structured_hits=min_structured_hits)

    labeler = ApertusLabeler()
    examples: list[Example] = []
    n_input = n_kept = n_dropped = 0

    for batch in _batched(filtered, batch_size):
        # Defensive truncation: very long chunks blow up vLLM context budget
        truncated = [c for c in batch if len(c.text) <= max_chunk_chars]
        if not truncated:
            continue
        n_input += len(truncated)
        chunk_texts = [c.text for c in truncated]
        claims_per_chunk = labeler.label_batch(chunk_texts)
        for chunk, claims in zip(truncated, claims_per_chunk, strict=True):
            ex = verify_and_combine(
                chunk.text,
                claims,
                language=_detect_language(chunk.text, chunk.subset),
                doc_id=chunk.doc_id,
                subset=chunk.subset,
            )
            if ex is None:
                n_dropped += 1
                continue
            examples.append(ex)
            n_kept += 1
        print(
            f"  processed={n_input} kept={n_kept} dropped={n_dropped} "
            f"(yield={n_kept / max(1, n_input):.1%})"
        )
        if max_chunks and n_kept >= max_chunks:
            break

    written = write_jsonl(out_path, examples)
    print(f"\nwrote {written} examples → {out_path}")
    print(f"final yield: {n_kept}/{n_input} = {n_kept / max(1, n_input):.1%}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-chunks", type=int, default=1000,
                    help="Stop after this many KEPT chunks (post-verify).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--subsets", nargs="+",
                    default=["entscheidsuche", "curia_vista"])
    ap.add_argument("--min-structured-hits", type=int, default=1)
    ap.add_argument("--max-chunk-chars", type=int, default=1500)
    ap.add_argument("--local-parquet-dir", type=Path, default=None,
                    help="Read pre-downloaded parquet files instead of HF streaming.")
    args = ap.parse_args()

    run(
        out_path=args.out,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
        subsets=tuple(args.subsets),
        min_structured_hits=args.min_structured_hits,
        max_chunk_chars=args.max_chunk_chars,
        local_parquet_dir=args.local_parquet_dir,
    )


if __name__ == "__main__":
    main()
