"""A1.4 quality gate: curated 100-chunk evaluation of Gemma's labeling.

The 20-chunk PoC happened to land on PII-sparse anonymised legal text and
under-sampled person/address content. This gate is more aggressive:

  - 25 chunks per subset (entscheidsuche / curia_vista / fineweb / romansh)
  - Within each subset, prefer chunks where the regex prefilter found at
    least one structured PII hit (date / IBAN / email / phone / capitalised
    person-shape) — that's a cheap signal that real PII is present.
  - Allow longer chunks (up to 1500 chars) since real PII often appears in
    longer documents.
  - Mix in 5 prefilter-empty chunks per subset as negative controls.

Output:
  - eval/gemma_quality_gate.json — per-chunk results
  - Console: per-(subset × category) span counts, top-30 sample spans,
    list of empty/disagreement chunks.

Decision rule (gate to pass before A1.3 50k production run):
  - Person/address spans surface in ≥ 30% of likely-PII chunks
  - Zero hallucinations (every span verbatim-matches the source — the
    labeler enforces this; we just verify the gate didn't slip)
  - Negative-control chunks return ``{"spans": []}``

Run:
    uv run python -m gheim_training.data.gemma.quality_gate_100
"""
from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from ..apertus_label.prefilter import find_structured
from ..apertus_label.stream import Subset, _stream_local_parquet
from ..lang_detect import detect as detect_lang
from .labeler import GemmaLabeler

PARQUET_DIR = Path("data/raw/apertus-swiss")
SUBSETS: tuple[Subset, ...] = ("entscheidsuche", "curia_vista", "fineweb", "romansh")

LIKELY_PII_PER_SUBSET = 20
NEGATIVES_PER_SUBSET = 5  # chunks the regex prefilter found nothing in


def _stream_with_classification(subset: Subset) -> Iterator[tuple[str, str, int]]:
    """Yield (chunk_text, doc_id, n_regex_hits) for chunks of the subset.

    ``n_regex_hits`` lets the caller pick PII-likely chunks vs negative
    controls cheaply.
    """
    for chunk in _stream_local_parquet(
        PARQUET_DIR, subsets=(subset,),
        max_chunks=0, skip_first_n_records=0,
    ):
        if len(chunk.text) < 200 or len(chunk.text) > 1500:
            continue
        n = len(find_structured(chunk.text))
        yield (chunk.text, chunk.doc_id, n)


def _curated_sample(subset: Subset) -> list[tuple[str, str, str]]:
    """Sample 20 likely-PII + 5 negative-control chunks for one subset."""
    likely: list[tuple[str, str, str]] = []
    negatives: list[tuple[str, str, str]] = []
    seen_docs_likely: set[str] = set()
    seen_docs_neg: set[str] = set()
    for text, doc_id, n_hits in _stream_with_classification(subset):
        # Diversify by source document so we don't pull 25 chunks from
        # one giant court decision.
        if n_hits >= 1 and doc_id not in seen_docs_likely \
                and len(likely) < LIKELY_PII_PER_SUBSET:
            likely.append((text, doc_id, subset))
            seen_docs_likely.add(doc_id)
        elif n_hits == 0 and doc_id not in seen_docs_neg \
                and len(negatives) < NEGATIVES_PER_SUBSET:
            negatives.append((text, doc_id, subset))
            seen_docs_neg.add(doc_id)
        if (len(likely) >= LIKELY_PII_PER_SUBSET
                and len(negatives) >= NEGATIVES_PER_SUBSET):
            break
    return likely + negatives


def main() -> None:
    print("=== A1.4: Gemma quality gate on 100 curated Swiss chunks ===\n")
    print("Sampling 20 likely-PII + 5 negative-control chunks per subset…")
    chunks: list[tuple[str, str, str]] = []
    for subset in SUBSETS:
        sample = _curated_sample(subset)
        chunks.extend(sample)
        n_distinct_docs = len({did for _, did, _ in sample})
        print(f"  {subset:<16} → {len(sample)} chunks from "
              f"{n_distinct_docs} distinct docs")
    print(f"Total: {len(chunks)} chunks\n")

    print("Detecting language for each chunk…")
    langs = [detect_lang(c[0], subset=c[2]) for c in chunks]
    by_lang = Counter(langs)
    print(f"  {dict(by_lang)}\n")

    print("Loading Gemma 4 26B-A4B-AWQ…")
    labeler = GemmaLabeler()
    t0 = time.time()
    examples = labeler.label_batch(
        [c[0] for c in chunks],
        languages=langs,
        doc_ids=[c[1] for c in chunks],
        subset="quality_gate",
    )
    elapsed = time.time() - t0
    print(f"  labeled {len(examples)} chunks in {elapsed:.0f}s "
          f"({len(examples)/elapsed:.2f} chunks/sec)\n")

    # Aggregate per-(subset, category, language)
    per_subset_cat: dict[tuple[str, str], int] = Counter()
    per_lang_cat: dict[tuple[str, str], int] = Counter()
    per_cat: dict[str, int] = Counter()
    n_with_pii_likely = 0
    n_with_pii_negative = 0
    n_likely_total = 0
    n_negative_total = 0
    samples_per_cat: dict[str, list[tuple[str, str, str]]] = {}

    for ex, (text, _doc_id, subset), la in zip(examples, chunks, langs, strict=True):
        is_likely = len(find_structured(text)) >= 1
        if is_likely:
            n_likely_total += 1
        else:
            n_negative_total += 1
        if ex is None or not ex.spans:
            continue
        if is_likely:
            n_with_pii_likely += 1
        else:
            n_with_pii_negative += 1
        for sp in ex.spans:
            surface = ex.text[sp.start:sp.end]
            per_subset_cat[(subset, sp.label)] += 1
            per_lang_cat[(la, sp.label)] += 1
            per_cat[sp.label] += 1
            samples_per_cat.setdefault(sp.label, []).append(
                (subset, surface, text[:80])
            )

    print("=== Per-category span counts ===")
    for cat in sorted(per_cat):
        print(f"  {cat:<22} {per_cat[cat]}")
    print()
    print("=== Per-(subset × category) ===")
    for (subset, cat), n in sorted(per_subset_cat.items()):
        print(f"  {subset:<16} {cat:<22} {n}")
    print()
    print("=== Per-(language × category) ===")
    for (lang, cat), n in sorted(per_lang_cat.items()):
        print(f"  {lang:<8} {cat:<22} {n}")
    print()

    print("=== Sample spans per category (up to 5 each) ===")
    for cat in sorted(samples_per_cat):
        print(f"\n  --- {cat} ---")
        for subset, surface, snippet in samples_per_cat[cat][:5]:
            print(f"    [{subset}] {surface!r}  in  {snippet!r}…")

    print("\n=== Gate summary ===")
    rate_likely = n_with_pii_likely / max(n_likely_total, 1)
    rate_negative = n_with_pii_negative / max(n_negative_total, 1)
    print(f"  Likely-PII chunks with ≥1 span: {n_with_pii_likely}/{n_likely_total} "
          f"({rate_likely:.0%})")
    print(f"  Negative-control chunks with ≥1 span: {n_with_pii_negative}/{n_negative_total} "
          f"({rate_negative:.0%})")
    print()
    person_total = per_cat.get("private_person", 0)
    address_total = per_cat.get("private_address", 0)
    print(f"  Person spans across all 100 chunks: {person_total}")
    print(f"  Address spans across all 100 chunks: {address_total}")

    # Save full report
    out_path = Path("eval/gemma_quality_gate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_chunks": len(chunks),
        "elapsed_seconds": elapsed,
        "throughput_chunks_per_sec": len(chunks) / elapsed,
        "per_category_counts": dict(per_cat),
        "per_subset_category": {f"{k[0]}/{k[1]}": v for k, v in per_subset_cat.items()},
        "per_language_category": {f"{k[0]}/{k[1]}": v for k, v in per_lang_cat.items()},
        "n_likely_with_pii": n_with_pii_likely,
        "n_likely_total": n_likely_total,
        "n_negative_with_pii": n_with_pii_negative,
        "n_negative_total": n_negative_total,
        "examples": [
            {
                "text": ex.text,
                "subset": chunks[i][2],
                "doc_id": chunks[i][1],
                "language": langs[i],
                "spans": [
                    {"start": sp.start, "end": sp.end, "label": sp.label,
                     "surface": ex.text[sp.start:sp.end]}
                    for sp in (ex.spans if ex else [])
                ],
                "n_regex_hits": len(find_structured(chunks[i][0])),
            }
            for i, ex in enumerate(examples) if ex is not None
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()
