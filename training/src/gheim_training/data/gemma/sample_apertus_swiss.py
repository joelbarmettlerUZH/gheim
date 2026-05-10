"""A1.2: Stratified sampler for apertus-pretrain-swiss.

Pulls a balanced sample of chunks across (subset × language), diversified
across source documents. Output is a flat JSONL of unlabeled chunks ready
for the Gemma labeling runner (A1.3).

Stratification target (50k default):
  entscheidsuche  9k  (de/fr/it mixed, real legal language)
  curia_vista     9k  (de/fr/it mixed, parliamentary)
  fineweb        15k  (de/fr/it Swiss web — broader register)
  romansh        17k  (oversampled — smallest source, biggest gap in our model)

Per-(subset × language) quotas are filled by language-detecting each
streamed chunk and routing it to the appropriate cell. We cap per-doc
chunks (default 8) so a single long court decision can't dominate a
quota.

Output JSONL fields per chunk (no labels yet):
  - text, doc_id, subset, language
  - chunk_index_in_doc (for de-dup if same doc reappears across runs)
  - source_dataset = "swiss-ai/apertus-pretrain-swiss"

Run:
    uv run python -m gheim_training.data.gemma.sample_apertus_swiss \\
        --out data/layer5v3_unlabeled.jsonl --target 50000
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from ..apertus_label.stream import Subset, _stream_local_parquet
from ..lang_detect import detect as detect_lang
from ..schema import Language

PARQUET_DIR = Path("data/raw/apertus-swiss")
SOURCE_DATASET = "swiss-ai/apertus-pretrain-swiss"


@dataclass(frozen=True, slots=True)
class _UnlabeledChunk:
    text: str
    doc_id: str
    subset: Subset
    language: Language
    chunk_index_in_doc: int


# Per-(subset × language) target chunk counts. Sums to ~50k. RM is over-
# sampled deliberately because it's our biggest gap (0 RM examples in the
# gheim-2 gold core).
_TARGETS: dict[tuple[str, str], int] = {
    ("entscheidsuche", "de_ch"): 4000,
    ("entscheidsuche", "fr_ch"): 3000,
    ("entscheidsuche", "it_ch"): 2000,
    ("curia_vista", "de_ch"):    4000,
    ("curia_vista", "fr_ch"):    3000,
    ("curia_vista", "it_ch"):    2000,
    ("fineweb", "de_ch"):        8000,
    ("fineweb", "fr_ch"):        4000,
    ("fineweb", "it_ch"):        3000,
    ("romansh", "rm"):          17000,
}

# Maximum chunks to take from any one source document. Protects against a
# single long court decision filling 90% of a quota with near-duplicate
# legalese from the same case.
_MAX_PER_DOC = 8

# Maximum chunks to scan per subset before giving up on remaining quota
# (some quotas may be unfillable for the smaller corpora).
_MAX_SCAN_PER_SUBSET: dict[Subset, int] = {
    "entscheidsuche": 200_000,
    "curia_vista":    200_000,
    "fineweb":        200_000,
    "romansh":        200_000,
}


def sample(
    target_subsets: tuple[Subset, ...] = ("entscheidsuche", "curia_vista",
                                          "fineweb", "romansh"),
    targets: Annotated[
        dict[tuple[str, str], int],
        "Per-(subset, language) chunk targets. Defaults to a 50k stratification.",
    ] = _TARGETS,
    max_per_doc: int = _MAX_PER_DOC,
    min_chars: int = 200,
    max_chars: int = 1500,
) -> Iterator[_UnlabeledChunk]:
    """Stream stratified chunks honouring per-(subset, language) quotas."""
    filled: Counter[tuple[str, str]] = Counter()
    per_doc: dict[str, int] = defaultdict(int)
    chunk_idx_per_doc: dict[str, int] = defaultdict(int)

    for subset in target_subsets:
        # Quotas active for this subset
        active_cells = {(s, lang) for (s, lang) in targets if s == subset}
        if not active_cells:
            continue
        scan_cap = _MAX_SCAN_PER_SUBSET.get(subset, 200_000)
        n_scanned = 0
        print(f"  scanning subset={subset}…", flush=True)
        t0 = time.time()
        for chunk in _stream_local_parquet(
            PARQUET_DIR, subsets=(subset,),
            max_chunks=0, skip_first_n_records=0,
        ):
            n_scanned += 1
            if n_scanned >= scan_cap:
                print(f"    scan-cap {scan_cap} reached for {subset}; "
                      f"stopping (quotas filled: "
                      f"{[(c, filled[c], targets[c]) for c in active_cells]})",
                      flush=True)
                break
            if not (min_chars <= len(chunk.text) <= max_chars):
                continue
            # Drop chunks from docs that already filled the per-doc cap
            if per_doc[chunk.doc_id] >= max_per_doc:
                continue
            lang = detect_lang(chunk.text, subset=chunk.subset)
            cell = (chunk.subset, lang)
            if cell not in active_cells:
                continue
            if filled[cell] >= targets[cell]:
                continue
            per_doc[chunk.doc_id] += 1
            chunk_idx_per_doc[chunk.doc_id] += 1
            filled[cell] += 1
            yield _UnlabeledChunk(
                text=chunk.text,
                doc_id=chunk.doc_id,
                subset=chunk.subset,
                language=lang,
                chunk_index_in_doc=chunk_idx_per_doc[chunk.doc_id] - 1,
            )
            # Periodic progress output every 1000 chunks emitted from this subset
            n_in_subset = sum(filled[c] for c in active_cells)
            if n_in_subset % 1000 == 0:
                _print_progress(filled, active_cells, targets,
                                elapsed=time.time() - t0, scanned=n_scanned)
            # Early-exit: all quotas for this subset filled
            if all(filled[c] >= targets[c] for c in active_cells):
                print(f"    subset={subset} all quotas filled "
                      f"(scanned {n_scanned} records)", flush=True)
                break


def _print_progress(filled, active_cells, targets, elapsed, scanned):
    parts = ", ".join(
        f"{c[1]}={filled[c]}/{targets[c]}" for c in sorted(active_cells)
    )
    print(f"    [{parts}] in {elapsed:.0f}s ({scanned} scanned)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=50_000,
                    help="Total chunk target (proportionally scales _TARGETS).")
    ap.add_argument("--max-per-doc", type=int, default=_MAX_PER_DOC)
    args = ap.parse_args()

    # Scale the default per-cell targets to hit args.target total
    scale = args.target / sum(_TARGETS.values())
    scaled_targets = {k: max(1, int(round(v * scale))) for k, v in _TARGETS.items()}
    print(f"Target: {args.target} chunks total. Per-cell targets:")
    for k, v in sorted(scaled_targets.items()):
        print(f"  {k[0]:<16} {k[1]:<6} {v}")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    by_cell: Counter[tuple[str, str]] = Counter()
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as f:
        for c in sample(targets=scaled_targets, max_per_doc=args.max_per_doc):
            payload = {
                "text": c.text,
                "doc_id": c.doc_id,
                "subset": c.subset,
                "language": c.language,
                "chunk_index_in_doc": c.chunk_index_in_doc,
                "source_dataset": SOURCE_DATASET,
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
            n_written += 1
            by_cell[(c.subset, c.language)] += 1

    elapsed = time.time() - t0
    print(f"\nwrote {n_written} chunks → {args.out} (in {elapsed:.0f}s)")
    print("Per-(subset, language) emitted:")
    for k, v in sorted(by_cell.items()):
        target = scaled_targets.get(k, 0)
        flag = "" if v >= target else "  ← short of target"
        print(f"  {k[0]:<16} {k[1]:<6} {v:>6} / target {target}{flag}")


if __name__ == "__main__":
    main()
