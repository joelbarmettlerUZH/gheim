"""P1a-i: Re-stream apertus + regex-mine rare-PII chunks not already in layer5v4.

The Gemma run capped at per_doc_cap=5 chunks per source doc. Many docs
have IBANs / AHV numbers / phones in chunks beyond the first 5. This
script re-streams apertus-pretrain-swiss + -romansh with NO per-doc cap,
runs the validated regex catalog on every chunk, and saves only chunks
that:
  (a) contain at least one regex+checksum-validated PII hit, AND
  (b) are NOT already in data/layer5v4.jsonl (deduped by chunk id).

Output: data/layer5v4_regex_new.jsonl with the same schema as layer5v4
(text, spans, language, subset, source_dataset, doc_id,
chunk_index_in_doc, labeler="regex+checksum", prompt_version, labeled_at).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Reuse the streaming + chunking infrastructure
from .gemma.run_label_streaming import (
    _candidate_chunks_from_record,
    _make_id,
    _stream_romansh,
    _stream_swiss,
)

# Add packages/gheim-py/src for the validated regex catalog
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "packages" / "gheim-py" / "src"))
from gheim.detectors.composite import _find_regex_spans  # type: ignore[import-not-found]  # noqa: E402, I001

SEEN_PATH = Path("data/layer5v4.jsonl")
OUT_PATH = Path("data/layer5v4_regex_new.jsonl")
LABELER = "regex+checksum:gheim.detectors.composite._find_regex_spans"
PROMPT_VERSION = "regex-v1-2026-05-08"


def _load_seen_ids(path: Path) -> set[str]:
    print(f"Loading seen ids from {path} ...", flush=True)
    seen: set[str] = set()
    n = 0
    with path.open() as f:
        for line in f:
            n += 1
            rec_id = json.loads(line).get("id")
            if rec_id:
                seen.add(rec_id)
            if n % 200_000 == 0:
                print(f"  ... {n:,}", flush=True)
    print(f"Loaded {len(seen):,} ids from {n:,} records", flush=True)
    return seen


def _stream_with_high_cap():
    """Yield (cand, source_label) with NO per-doc cap.

    Round-robin alternation across swiss + romansh, mirroring the
    original run_label_streaming logic but without the per_doc_cap.
    """
    from itertools import zip_longest
    swiss_iter = iter(_stream_swiss())
    rm_iter = iter(_stream_romansh())
    n_records = 0
    for swiss_rec, rm_rec in zip_longest(swiss_iter, rm_iter, fillvalue=None):
        for rec, default_subset in [(swiss_rec, ""), (rm_rec, "romansh")]:
            if rec is None:
                continue
            n_records += 1
            yield from _candidate_chunks_from_record(rec, default_subset)
        if n_records % 100_000 == 0:
            print(f"  ... scanned {n_records:,} source records", flush=True)


def main() -> None:
    seen = _load_seen_ids(SEEN_PATH)

    n_chunks_seen = 0
    n_chunks_new = 0
    n_chunks_with_hit = 0
    n_spans_total = 0
    by_lang_cat: Counter[tuple[str, str]] = Counter()
    by_subset: Counter[str] = Counter()
    chunk_idx_per_doc: dict[str, int] = {}
    label_meta = {
        "labeler": LABELER,
        "prompt_version": PROMPT_VERSION,
        "labeled_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"Writing: {OUT_PATH}", flush=True)
    with OUT_PATH.open("w") as fout:
        for cand in _stream_with_high_cap():
            n_chunks_seen += 1
            # Reconstruct the id the same way run_label_streaming did:
            # incrementing chunk_idx per doc_id.
            cidx = chunk_idx_per_doc.get(cand.doc_id, 0)
            chunk_idx_per_doc[cand.doc_id] = cidx + 1
            chunk_id = _make_id(cand.doc_id, cidx)

            if chunk_id in seen:
                continue
            n_chunks_new += 1

            spans = _find_regex_spans(cand.text)
            if not spans:
                continue

            n_chunks_with_hit += 1
            n_spans_total += len(spans)

            text = cand.text
            span_dicts = [
                {
                    "start": sp.start,
                    "end": sp.end,
                    "label": sp.label,
                    "value": text[sp.start:sp.end],
                    "source": "regex",
                }
                for sp in spans
            ]
            for sp in spans:
                by_lang_cat[(cand.language, sp.label)] += 1
            by_subset[cand.subset] += 1

            rec_out = {
                "id": chunk_id,
                "text": text,
                "language": cand.language,
                "subset": cand.subset,
                "source_dataset": "swiss-ai/apertus-pretrain-swiss"
                                 if cand.subset != "romansh"
                                 else "swiss-ai/apertus-pretrain-romansh",
                "doc_id": cand.doc_id,
                "chunk_index_in_doc": cidx,
                "spans": span_dicts,
                **label_meta,
            }
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

            if n_chunks_with_hit % 5_000 == 0:
                print(f"  written {n_chunks_with_hit:,} new regex-hit chunks "
                      f"(scanned {n_chunks_seen:,}, new {n_chunks_new:,})",
                      flush=True)

    print()
    print("DONE")
    print(f"  total chunks scanned: {n_chunks_seen:,}")
    print(f"  new (not in layer5v4): {n_chunks_new:,}")
    print(f"  with regex+checksum hit: {n_chunks_with_hit:,}")
    print(f"  total spans added: {n_spans_total:,}")
    print(f"  output: {OUT_PATH}")
    print()
    print("By (lang × cat):")
    LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
    cats = sorted({c for _, c in by_lang_cat})
    for cat in cats:
        row = " ".join(f"{by_lang_cat.get((la, cat), 0):>8}" for la in LANGS)
        print(f"  {cat:<22} {row}")
    print()
    print("By subset:")
    for s, c in by_subset.most_common():
        print(f"  {s:<20} {c:>10,}")


if __name__ == "__main__":
    main()
