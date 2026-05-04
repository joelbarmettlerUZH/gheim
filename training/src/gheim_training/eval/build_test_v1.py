"""Day 3 surfacer for test_v1: stream non-prod chunks, prefilter, stratify.

Produces ``data/test_v1_pool.jsonl`` — the candidate pool that feeds the
4-way subagent labeling pipeline (Days 4–5). Each pool record contains a
chunk's text, its detected language, its source corpus, and a
``presence`` flag (``pii_dense`` if the prefilter found ≥1 structured PII
hit, ``control`` if not). The two flags map to two different labeler
prompts: dense chunks get "find all PII"; controls get "verify there is
no PII" (false-positive measurement).

Stratification target (defaults):
  - 4 Swiss languages: de_ch, fr_ch, it_ch, rm
  - PII-dense per language: 150 chunks
  - PII-free controls per language: 50 chunks
  - Total pool: ~800 chunks
  - Expected gold after labeling + agreement: ≥600 chunks, ≥1500 spans

EN is intentionally excluded — gheim's primary deployment is Swiss; English
regression is measured separately on Day 15 against a layer4_en validation
slice.

Source: ``data/raw/apertus-swiss-holdout/`` (a separate parquet shard
reserved for held-out evals). All FineWeb-2-Swiss content. Defensive
dedup against ``data/prod_inputs.jsonl`` doc_ids in case the holdout
shard accidentally overlaps.

Run:
    uv run python -m gheim_training.eval.build_test_v1 \\
        --out data/test_v1_pool.jsonl \\
        --target-per-lang 150 \\
        --controls-per-lang 50
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

from ..data.apertus_label.prefilter import passes
from ..data.apertus_label.stream import Chunk, Subset, chunk_sentences, stream_chunks
from ..data.lang_detect import detect as detect_lang
from ..data.schema import Language

# Languages we stratify on. EN is excluded by design (see module docstring).
TestV1Language = Literal["de_ch", "fr_ch", "it_ch", "rm"]
_TARGET_LANGS: tuple[TestV1Language, ...] = ("de_ch", "fr_ch", "it_ch", "rm")

# Per-chunk "is there PII or not" flag drives which labeler prompt the
# subagent uses downstream and how its output is scored (FP vs F1).
Presence = Literal["pii_dense", "control"]


@dataclass(frozen=True, slots=True)
class PoolRecord:
    """One candidate chunk, ready to hand to a subagent labeler."""
    chunk_id: Annotated[
        str,
        "Stable T-prefixed counter. Used as the join key when subagent gold "
        "comes back, and as the chunk identifier in the final test_v1.jsonl.",
    ]
    text: Annotated[str, "The chunk text — passed verbatim to the labeler."]
    language: Annotated[
        Language,
        "Lingua-detected language. May be any gheim Language but the surfacer "
        "stratifies only on de_ch / fr_ch / it_ch / rm.",
    ]
    subset: Annotated[
        Subset,
        "Source corpus tag (entscheidsuche / curia_vista / fineweb / romansh). "
        "Recorded so the final test set can be cut by source in eval reports.",
    ]
    doc_id: Annotated[str, "Upstream document id from the parquet shard."]
    presence: Annotated[
        Presence,
        "pii_dense = prefilter saw ≥1 structured PII hit; control = none. "
        "Drives the subagent prompt and the FP-vs-recall score it goes into.",
    ]


def _load_prod_doc_ids(path: Path) -> set[str]:
    """Build the set of doc_ids that fed the prod labeling pipeline so we
    can defensively skip them. ``prod_inputs.jsonl`` is ~200 MB / 318k
    records; the doc_id column is a string short enough to fit in memory."""
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_id = d.get("doc_id")
            if doc_id:
                out.add(str(doc_id))
    return out


def _iter_candidates(
    holdout_dir: Path,
    prod_doc_ids: set[str],
) -> Iterator[tuple[Chunk, Language, Presence]]:
    """Stream chunks from the holdout parquet, yield (chunk, language, presence).

    Skips prod-overlap chunks defensively. ``stream_chunks`` already enforces
    the ``Subset`` Literal so chunk.subset is safe to use unfiltered downstream.
    """
    for chunk in stream_chunks(
        subsets=("entscheidsuche", "curia_vista", "fineweb", "romansh"),
        local_parquet_dir=holdout_dir,
    ):
        if chunk.doc_id in prod_doc_ids:
            continue
        # detect_lang uses chunk.subset only as a fallback hint for very
        # short text; on FineWeb chunks (~400-800 chars) lingua dominates.
        lang = detect_lang(chunk.text, subset=chunk.subset)
        presence: Presence = "pii_dense" if passes(chunk.text, min_hits=1) else "control"
        yield chunk, lang, presence


def _iter_romansh_candidates(
    rm_files: tuple[Path, ...],
    prod_doc_ids: set[str],
) -> Iterator[tuple[Chunk, Language, Presence]]:
    """Stream chunks from Romansh-specific JSONL.gz files (e.g. rm_wikipedia).

    The romansh corpus uses a different schema than the parquet shards
    (text, id, language="roh", idiom, …). We re-use ``chunk_sentences`` for
    sentence-bound chunking and tag each chunk with subset="romansh" so
    downstream code treats them like the existing romansh subset.
    """
    for path in rm_files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_id = str(d.get("id") or "")
                if doc_id in prod_doc_ids:
                    continue
                text = d.get("text") or ""
                if not text:
                    continue
                for chunk_text in chunk_sentences(text):
                    chunk = Chunk(text=chunk_text, doc_id=doc_id, subset="romansh")
                    # subset="romansh" forces the RM detection branch in
                    # detect() — protects against lingua mis-classifying
                    # short Romansh sentences as Italian.
                    lang = detect_lang(chunk.text, subset="romansh")
                    presence: Presence = (
                        "pii_dense" if passes(chunk.text, min_hits=1) else "control"
                    )
                    yield chunk, lang, presence


def _stratify_and_sample(
    candidates: list[tuple[Chunk, Language, Presence]],
    target_per_lang: int,
    controls_per_lang: int,
    rng: random.Random,
) -> list[tuple[Chunk, Language, Presence]]:
    """Bucket candidates by (language, presence), then sample-without-replacement
    per bucket. Buckets that fall short of target are taken in full (with a
    warning); they don't borrow from other languages because we want
    per-language CIs to be honest."""
    buckets: dict[tuple[Language, Presence], list[tuple[Chunk, Language, Presence]]] = (
        defaultdict(list)
    )
    for cand in candidates:
        chunk, lang, presence = cand
        if lang not in _TARGET_LANGS:
            continue
        buckets[(lang, presence)].append(cand)

    out: list[tuple[Chunk, Language, Presence]] = []
    for lang in _TARGET_LANGS:
        for presence, target in (("pii_dense", target_per_lang),
                                  ("control", controls_per_lang)):
            pool = buckets[(lang, presence)]
            if len(pool) < target:
                print(
                    f"  WARN: lang={lang} presence={presence} only has "
                    f"{len(pool)} candidates, target was {target}; using all"
                )
                out.extend(pool)
            else:
                out.extend(rng.sample(pool, target))
    return out


def _write_pool(path: Path, records: list[PoolRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/test_v1_pool.jsonl"),
                    help="Output JSONL path.")
    ap.add_argument("--holdout-dir", type=Path,
                    default=Path("data/raw/apertus-swiss-holdout/data"),
                    help="Directory of parquet files to stream from.")
    ap.add_argument("--prod-inputs", type=Path,
                    default=Path("data/prod_inputs.jsonl"),
                    help="JSONL of chunks fed to the prod labeling pipeline; "
                         "doc_ids in this file are defensively skipped.")
    ap.add_argument("--romansh-files", nargs="*", type=Path,
                    default=[Path("data/raw/apertus-romansh/monolingual/"
                                  "rm_wikipedia_new.jsonl.gz")],
                    help="Additional Romansh JSONL.gz sources (FineWeb-Swiss "
                         "is RM-poor; we top up from rm.wikipedia by default).")
    ap.add_argument("--target-per-lang", type=int, default=150,
                    help="Number of PII-dense chunks per language.")
    ap.add_argument("--controls-per-lang", type=int, default=50,
                    help="Number of PII-free control chunks per language.")
    ap.add_argument("--seed", type=int, default=17,
                    help="RNG seed for reproducible stratified sampling.")
    args = ap.parse_args()

    print(f"Loading prod doc_ids from {args.prod_inputs}…")
    prod_doc_ids = _load_prod_doc_ids(args.prod_inputs)
    print(f"  {len(prod_doc_ids):,} distinct prod doc_ids loaded")

    print(f"Streaming candidates from {args.holdout_dir}…")
    candidates: list[tuple[Chunk, Language, Presence]] = []
    for i, (chunk, lang, presence) in enumerate(
        _iter_candidates(args.holdout_dir, prod_doc_ids), start=1,
    ):
        candidates.append((chunk, lang, presence))
        if i % 5000 == 0:
            print(f"  …seen {i:,}, kept {len(candidates):,}")

    if args.romansh_files:
        print(f"Streaming Romansh candidates from {len(args.romansh_files)} file(s)…")
        rm_kept_before = sum(1 for _, lang, _ in candidates if lang == "rm")
        for i, (chunk, lang, presence) in enumerate(
            _iter_romansh_candidates(tuple(args.romansh_files), prod_doc_ids),
            start=1,
        ):
            candidates.append((chunk, lang, presence))
            if i % 1000 == 0:
                rm_kept = sum(1 for _, lng, _ in candidates if lng == "rm")
                print(f"  …rm seen {i:,}, total rm chunks {rm_kept:,}")
        rm_kept = sum(1 for _, lang, _ in candidates if lang == "rm")
        print(f"  RM additions: {rm_kept - rm_kept_before:,} (was {rm_kept_before})")
    n_total = len(candidates)

    # Diagnostic: per-(lang, presence) bucket sizes BEFORE sampling
    pre_counts: Counter[tuple[Language, Presence]] = Counter(
        (lang, presence) for _, lang, presence in candidates
    )
    print(f"\nCandidate counts ({n_total:,} total):")
    for (lang, presence), n in sorted(pre_counts.items()):
        print(f"  {lang:<6} {presence:<10} {n:>6}")

    # Stratified sample
    rng = random.Random(args.seed)
    sampled = _stratify_and_sample(
        candidates, args.target_per_lang, args.controls_per_lang, rng,
    )

    # Wrap as PoolRecords with stable T-prefixed ids
    records = [
        PoolRecord(
            chunk_id=f"T{i + 1:06d}",
            text=chunk.text,
            language=lang,
            subset=chunk.subset,
            doc_id=chunk.doc_id,
            presence=presence,
        )
        for i, (chunk, lang, presence) in enumerate(sampled)
    ]
    _write_pool(args.out, records)

    # Final report
    final_counts: Counter[tuple[Language, Presence]] = Counter(
        (r.language, r.presence) for r in records
    )
    print(f"\nWrote {len(records):,} pool records → {args.out}")
    print("Final bucket sizes:")
    for (lang, presence), n in sorted(final_counts.items()):
        print(f"  {lang:<6} {presence:<10} {n:>6}")


if __name__ == "__main__":
    main()
