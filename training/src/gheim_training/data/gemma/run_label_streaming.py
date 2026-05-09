"""A1.7+A1.8: Stream the whole 4.5M-record corpus through Gemma, save
all positives and capped negatives.

Architecture (in one process to keep it simple and resumable):

   ┌────────────────────────────────────────────────────────────────────┐
   │ HF streaming load                                                  │
   │   apertus-pretrain-swiss (default, all subsets, 3.92M)             │
   │   apertus-pretrain-romansh (monolingual, ~? records)               │
   │ → interleave with equal probability (over-samples RM relative to   │
   │   its size — wanted, since RM is our biggest gap)                  │
   │ → shuffle(buffer_size=10k) for in-stream randomisation             │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ Chunk + filter pipeline (sequential, lightweight)                  │
   │   - Sentence-bound chunk to 400-800 chars (chunk_sentences)        │
   │   - Length filter [200, 1500]                                       │
   │   - Per-doc-id cap (default 5 chunks per doc)                       │
   │   - Subset classify (entscheidsuche/curia_vista/fineweb/romansh)   │
   │   - fasttext language detect                                       │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼ accumulate batch
   ┌────────────────────────────────────────────────────────────────────┐
   │ Gemma label_batch (size=256, ~6s per batch on tp=2)                │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼ accept/reject per chunk
   ┌────────────────────────────────────────────────────────────────────┐
   │ Acceptance:                                                        │
   │   spans non-empty  → ALWAYS save                                   │
   │   spans empty      → save iff (lang) negative bucket non-full      │
   │ Stop when:                                                         │
   │   all negative buckets full AND spans-empty rate looks normal      │
   │   OR scan-cap hit (default: 5M records)                            │
   │   OR data exhausted                                                │
   └──────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ Resumable JSONL output (append-mode)                               │
   │   - One row per accepted chunk                                     │
   │   - Sidecar state file every N batches                             │
   └────────────────────────────────────────────────────────────────────┘

Run:
    uv run python -m gheim_training.data.gemma.run_label_streaming \\
        --out data/layer5v4.jsonl \\
        --neg-de 10000 --neg-fr 8000 --neg-it 6000 --neg-rm 8000 \\
        --max-records 5000000 \\
        --batch-size 256

Resumes from existing --out file (skips already-seen ids on restart).
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..apertus_label.stream import _classify, chunk_sentences
from ..lang_detect import detect as detect_lang
from ..schema import Language
from .labeler import GemmaLabeler

LABELER_MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
PROMPT_VERSION = "v3-localized-2026-05-06"

SOURCE_SWISS = "swiss-ai/apertus-pretrain-swiss"
SOURCE_ROMANSH = "swiss-ai/apertus-pretrain-romansh"


@dataclass(frozen=True, slots=True)
class _Candidate:
    text: str
    doc_id: str
    subset: str
    language: Language


def _stream_swiss(shuffle_buffer: int = 10_000):
    """Stream apertus-pretrain-swiss (all subsets) with in-stream shuffle."""
    from datasets import load_dataset
    ds = load_dataset(SOURCE_SWISS, split="train", streaming=True)
    return ds.shuffle(buffer_size=shuffle_buffer, seed=17)


def _stream_romansh(shuffle_buffer: int = 5_000):
    """Stream apertus-pretrain-romansh `monolingual` config (real RM text)."""
    from datasets import load_dataset
    ds = load_dataset(SOURCE_ROMANSH, "monolingual", split="train", streaming=True)
    return ds.shuffle(buffer_size=shuffle_buffer, seed=17)


def _candidate_chunks_from_record(rec: dict, default_subset: str) -> Iterator[_Candidate]:
    """Convert one source record into 0+ candidate chunks for labeling."""
    text = rec.get("text") or ""
    if not text:
        return
    doc_id = rec.get("id") or rec.get("uid") or rec.get("filename") or ""

    if default_subset:
        subset_str = default_subset
    else:
        subset_obj = _classify(rec)
        subset_str = subset_obj if subset_obj else "unknown"

    for chunk in chunk_sentences(text):
        if not (200 <= len(chunk) <= 1500):
            continue
        lang = detect_lang(chunk, subset=subset_str)
        yield _Candidate(text=chunk, doc_id=str(doc_id), subset=subset_str,
                         language=lang)


def _interleaved_candidates(
    per_doc_cap: int,
    max_records: int,
) -> Iterator[_Candidate]:
    """Yield candidate chunks from swiss + romansh streams in parallel.

    Round-robin alternation per record so every batch sees a mix of
    sources. Per-doc cap is enforced across the whole run (not per
    source) — same doc_id from different streams is unlikely.
    """
    from itertools import zip_longest
    swiss_iter = iter(_stream_swiss())
    rm_iter = iter(_stream_romansh())
    n_records = 0
    per_doc_count: dict[str, int] = {}
    for swiss_rec, rm_rec in zip_longest(swiss_iter, rm_iter, fillvalue=None):
        for rec, default_subset in [(swiss_rec, ""), (rm_rec, "romansh")]:
            if rec is None:
                continue
            n_records += 1
            for c in _candidate_chunks_from_record(rec, default_subset):
                if per_doc_count.get(c.doc_id, 0) >= per_doc_cap:
                    continue
                per_doc_count[c.doc_id] = per_doc_count.get(c.doc_id, 0) + 1
                yield c
            if n_records >= max_records:
                return


@dataclass
class _NegBuckets:
    targets: dict[str, int]
    filled: dict[str, int]

    def admit_negative(self, lang: str) -> bool:
        target = self.targets.get(lang, 0)
        if self.filled.get(lang, 0) < target:
            self.filled[lang] = self.filled.get(lang, 0) + 1
            return True
        return False

    def all_full(self) -> bool:
        return all(self.filled.get(k, 0) >= v for k, v in self.targets.items())


def _seen_ids(path: Path) -> set[str]:
    """Read existing output for resume support."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line).get("id")
            except json.JSONDecodeError:
                continue
            if isinstance(rid, str):
                seen.add(rid)
    return seen


def _make_id(doc_id: str, chunk_idx: int) -> str:
    return f"{doc_id}#{chunk_idx}"


def _flush(out_path: Path, rows: list[dict[str, Any]]) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--per-doc-cap", type=int, default=5)
    ap.add_argument("--max-records", type=int, default=5_000_000,
                    help="Safety cap on records scanned (4.5M is the corpus).")
    ap.add_argument("--neg-de", type=int, default=10_000)
    ap.add_argument("--neg-fr", type=int, default=8_000)
    ap.add_argument("--neg-it", type=int, default=6_000)
    ap.add_argument("--neg-rm", type=int, default=8_000)
    ap.add_argument("--neg-en", type=int, default=2_000)
    ap.add_argument("--print-every-batches", type=int, default=10)
    args = ap.parse_args()

    seen = _seen_ids(args.out)
    if seen:
        print(f"Resuming: {len(seen)} chunks already in {args.out}; will skip.")

    # Restore neg-bucket fill state by counting empty-spans rows in existing output.
    neg = _NegBuckets(
        targets={
            "de_ch": args.neg_de, "fr_ch": args.neg_fr,
            "it_ch": args.neg_it, "rm": args.neg_rm,
            "en": args.neg_en,
        },
        filled={},
    )
    if args.out.exists():
        with args.out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("spans"):
                    la = rec.get("language", "")
                    neg.filled[la] = neg.filled.get(la, 0) + 1
        print(f"  resume neg state: {dict(neg.filled)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Gemma 4 26B-A4B-AWQ via vLLM tp=2…")
    labeler = GemmaLabeler()
    t_load_0 = time.time()
    assert labeler.client is not None
    labeler.client._load()
    print(f"  loaded in {time.time() - t_load_0:.0f}s\n")

    # Per-doc chunk index counter (deterministic id assignment)
    chunk_idx_per_doc: dict[str, int] = {}

    # Pull candidates and accumulate batches
    batch: list[_Candidate] = []
    n_input = n_skipped_seen = 0
    n_pos_saved = n_neg_saved = n_neg_dropped = 0
    n_batches = 0
    t_run_0 = time.time()
    pos_per_lang: dict[str, int] = {}
    cat_counts: dict[str, int] = {}

    def _process_batch(batch: list[_Candidate]) -> None:
        nonlocal n_pos_saved, n_neg_saved, n_neg_dropped, n_batches
        chunks_text = [c.text for c in batch]
        languages = [c.language for c in batch]
        doc_ids = [c.doc_id for c in batch]
        examples = labeler.label_batch(
            chunks_text, languages=languages, doc_ids=doc_ids,
            subset="layer5v4",
        )
        rows: list[dict[str, Any]] = []
        labeled_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        for cand, ex in zip(batch, examples, strict=True):
            cidx = chunk_idx_per_doc.get(cand.doc_id, 0)
            chunk_idx_per_doc[cand.doc_id] = cidx + 1
            rid = _make_id(cand.doc_id, cidx)
            spans = ex.spans if ex is not None else []
            if not spans:
                # Negative — admit only if bucket non-full
                if not neg.admit_negative(cand.language):
                    n_neg_dropped += 1
                    continue
                n_neg_saved += 1
            else:
                n_pos_saved += 1
                pos_per_lang[cand.language] = pos_per_lang.get(cand.language, 0) + 1
                for sp in spans:
                    cat_counts[sp.label] = cat_counts.get(sp.label, 0) + 1

            spans_payload = [
                {"start": sp.start, "end": sp.end, "label": sp.label,
                 "value": cand.text[sp.start:sp.end]}
                for sp in spans
            ]
            rows.append({
                "id": rid, "text": cand.text,
                "language": cand.language, "subset": cand.subset,
                "source_dataset": SOURCE_ROMANSH if cand.subset == "romansh"
                                   else SOURCE_SWISS,
                "doc_id": cand.doc_id, "chunk_index_in_doc": cidx,
                "spans": spans_payload,
                "labeler": LABELER_MODEL_ID,
                "prompt_version": PROMPT_VERSION,
                "labeled_at": labeled_at,
            })
        _flush(args.out, rows)
        n_batches += 1
        if n_batches % args.print_every_batches == 0:
            elapsed = time.time() - t_run_0
            rate = (n_pos_saved + n_neg_saved) / max(elapsed, 1)
            print(
                f"[batch {n_batches}] +saved {len(rows)} "
                f"(pos={n_pos_saved}, neg={n_neg_saved}, dropped_neg={n_neg_dropped}) "
                f"· input {n_input}, skipped_seen={n_skipped_seen} "
                f"· {rate:.2f} saved/s "
                f"· neg buckets: {dict(neg.filled)}",
                flush=True,
            )

    for cand in _interleaved_candidates(args.per_doc_cap, args.max_records):
        n_input += 1
        cidx_pre = chunk_idx_per_doc.get(cand.doc_id, 0)
        rid_pre = _make_id(cand.doc_id, cidx_pre)
        if rid_pre in seen:
            n_skipped_seen += 1
            chunk_idx_per_doc[cand.doc_id] = cidx_pre + 1
            continue
        batch.append(cand)
        if len(batch) >= args.batch_size:
            _process_batch(batch)
            batch.clear()
    # final partial batch
    if batch:
        _process_batch(batch)

    elapsed = time.time() - t_run_0
    print(f"\nDone. {n_pos_saved + n_neg_saved} chunks saved "
          f"(pos={n_pos_saved}, neg={n_neg_saved}) "
          f"in {elapsed/60:.1f} min "
          f"({(n_pos_saved + n_neg_saved) / max(elapsed, 1):.2f} saved/s).")
    print(f"  input candidates: {n_input}, skipped (already seen): {n_skipped_seen}")
    print(f"  negatives dropped (bucket full): {n_neg_dropped}")
    print(f"  positives per language: {pos_per_lang}")
    print(f"  negative buckets: {dict(neg.filled)} / {neg.targets}")
    print(f"  category span counts: {cat_counts}")
    print(f"  output → {args.out}")


if __name__ == "__main__":
    main()
