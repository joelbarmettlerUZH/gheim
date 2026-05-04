"""Stream apertus-pretrain-swiss → PII-dense chunks ready for Apertus labeling.

The source dataset bundles four sub-corpora identified by the ``id`` prefix:

  - ``entscheidsuche_html/...``  — Swiss court decisions (4.5B tokens)
  - ``curia_vista/...``           — Parliamentary proceedings (0.5B tokens)
  - ``fineweb-2/...``             — Swiss-filtered web (1.8B tokens)
  - ``romansh/...``               — Romansh corpus (0.1B tokens)

For Layer 5 we filter to the first two — they're PII-dense by content type
(real party names, addresses, dates, IBAN/AHV references). Web content is
much sparser per token and will be picked up later by active learning.

Each document is chunked into 400-800 character windows on sentence
boundaries (period / newline). Chunks shorter than 200 chars are dropped.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

# Subset → id-prefix substring(s) that identify it. Conservative: we look for
# any of the listed substrings inside the record's id or metadata.file_path.
SUBSET_MARKERS: dict[str, tuple[str, ...]] = {
    "entscheidsuche": ("entscheidsuche",),
    # Curia Vista IDs use `curiavista` (one word), not `curia_vista`.
    "curia_vista": ("curiavista", "curia_vista", "curia-vista", "parlament"),
    "fineweb": ("fineweb-2", "fineweb_2", "fineweb"),
    # `roh_data` is the actual prefix used in apertus-pretrain-swiss for
    # Romansh content (announcements from Romansh-speaking municipalities).
    "romansh": ("romansh", "rumantsch", "roh_data", "/roh_"),
}

Subset = Literal["entscheidsuche", "curia_vista", "fineweb", "romansh"]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\n+")


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    doc_id: Annotated[str, "ID of the source document; preserved for audit traces."]
    subset: Subset


def _classify(rec: dict) -> Subset | None:
    rec_id = (rec.get("id") or "").lower()
    fp = ""
    md = rec.get("metadata")
    if isinstance(md, dict):
        fp = (md.get("file_path") or "").lower()
    haystack = rec_id + " " + fp
    for subset, markers in SUBSET_MARKERS.items():
        if any(m in haystack for m in markers):
            return cast(Subset, subset)
    return None


def _chunk_text(
    text: str,
    *,
    target_min: int = 400,
    target_max: int = 800,
    hard_min: int = 200,
) -> Iterator[str]:
    """Greedy: accumulate sentences until we cross target_min, emit at first
    chance to stay under target_max. Drops trailing fragments below hard_min.
    """
    sentences = _SENTENCE_BOUNDARY.split(text)
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        candidate = (buf + " " + s).strip() if buf else s
        if len(candidate) > target_max and buf:
            yield buf
            buf = s
            continue
        buf = candidate
        if len(buf) >= target_min:
            yield buf
            buf = ""
    if buf and len(buf) >= hard_min:
        yield buf


def stream_chunks(
    *,
    subsets: Annotated[tuple[Subset, ...], "Source corpora to include."] =
        ("entscheidsuche", "curia_vista"),
    max_chunks: Annotated[int, "Stop after producing this many chunks (0 = no limit)."] = 0,
    dataset_name: str = "swiss-ai/apertus-pretrain-swiss",
    skip_first_n_records: int = 0,
    local_parquet_dir: Annotated[
        Path | None, "If set, iterate parquet files here instead of HF streaming."
    ] = None,
) -> Iterator[Chunk]:
    """Yield Layer-5-eligible chunks.

    By default uses HF ``streaming=True``; set ``local_parquet_dir`` to read
    pre-downloaded parquet files (much faster, no rate-limit risk).
    """
    if local_parquet_dir is not None:
        yield from _stream_local_parquet(local_parquet_dir, subsets, max_chunks, skip_first_n_records)
        return
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)
    n_emitted = 0
    for i, rec in enumerate(ds):
        if i < skip_first_n_records:
            continue
        subset = _classify(rec)
        if subset is None or subset not in subsets:
            continue
        text = rec.get("text") or ""
        if not text:
            continue
        doc_id = rec.get("id") or f"unknown_{i}"
        for chunk in _chunk_text(text):
            yield Chunk(text=chunk, doc_id=doc_id, subset=subset)
            n_emitted += 1
            if max_chunks and n_emitted >= max_chunks:
                return


def _stream_local_parquet(
    parquet_dir: Path,
    subsets: tuple[Subset, ...],
    max_chunks: int,
    skip_first_n_records: int,
) -> Iterator[Chunk]:
    import pyarrow.parquet as pq
    paths = sorted(parquet_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no .parquet files under {parquet_dir}")
    n_emitted = 0
    skipped = 0
    for path in paths:
        # Read in row-group batches so we don't blow up memory on a 460 MB file.
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=1000, columns=["text", "id"]):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            for rid, text in zip(ids, texts, strict=True):
                if skipped < skip_first_n_records:
                    skipped += 1
                    continue
                rec = {"id": rid, "text": text}
                subset = _classify(rec)
                if subset is None or subset not in subsets:
                    continue
                if not text:
                    continue
                doc_id = rid or "unknown"
                for chunk in _chunk_text(text):
                    yield Chunk(text=chunk, doc_id=doc_id, subset=subset)
                    n_emitted += 1
                    if max_chunks and n_emitted >= max_chunks:
                        return
