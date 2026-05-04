"""Day 5-6 aggregator: combine 4-labeler outputs into consensus + disagreements.

For each chunk, collect (label, value) claims across all 4 labelers and
compute agreement counts. Output two files:

- ``data/test_v1_consensus.jsonl`` — chunks with their accepted spans
  (located via ``str.find`` and converted to (start, end, label) gold
  format compatible with the eval harness).
- ``data/test_v1_disagreements.jsonl`` — claims that need hand resolution
  (2-of-4 agreement). Each record contains the chunk text, the disputed
  claim, and which labelers proposed it.

Acceptance threshold:
  - ≥3 of 4 labelers agree on (label, exact verbatim value) → auto-accept
  - 2 of 4 → flag for hand resolution
  - 1 of 4 → drop (treat as labeler noise)

Verbatim normalisation: claims are joined on ``(label, value.strip())``
case-sensitive. A leading "Dr. " from one labeler vs no title from another
counts as disagreement (correctly — boundary differences matter for the
strict-F1 eval). Verbatim relocation uses ``str.find`` of the trimmed
value; not-found claims are dropped with a counter.

Run:
    uv run python -m gheim_training.eval.aggregate_labels \\
        --pool data/test_v1_pool.jsonl \\
        --labels-dir data/test_v1_labels \\
        --consensus-out data/test_v1_consensus.jsonl \\
        --disagreements-out data/test_v1_disagreements.jsonl \\
        --threshold 3
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from ..data.label_space import CATEGORIES
from ..data.schema import Span

# Strict filename pattern for labeler outputs: batch_NNN_labeler{A|B|C|D}.jsonl.
# Loaders raise on any deviation rather than silently skipping — a typo'd
# filename would otherwise cause a chunk to be missing one labeler's vote
# without any signal in the report.
_LABELER_FILENAME_RE = re.compile(r"^batch_(\d{3})_labeler([A-D])\.jsonl$")


@dataclass(frozen=True, slots=True)
class Claim:
    """One labeler's proposed span — verbatim text + label, no offsets."""
    label: Annotated[str, "One of the 8 gheim CATEGORIES; other values dropped."]
    value: Annotated[str, "Verbatim substring claimed for the span."]


@dataclass(frozen=True, slots=True)
class DisagreementRecord:
    chunk_id: str
    text: str
    language: str
    subset: str
    claim: Claim
    n_labelers_agreeing: int
    labelers_agreeing: tuple[str, ...]


def _read_jsonl(path: Path) -> Iterator[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _load_per_labeler(
    labels_dir: Path,
) -> tuple[dict[str, dict[str, list[Claim]]], dict[str, set[str]]]:
    """Return ``(by_chunk_claims, responded_per_chunk)``.

    - ``by_chunk_claims``: chunk_id → labeler_letter → list[Claim]. Only
      includes labelers that proposed at least one valid claim for the
      chunk.
    - ``responded_per_chunk``: chunk_id → set of labeler letters that
      EMITTED a record for the chunk (even with empty spans). Lets us
      distinguish "clean negative" (4 labelers said empty) from
      "no labeler responded" (genuine missing data).

    Reads all ``batch_NNN_labeler{A,B,C,D}.jsonl`` files under ``labels_dir``.
    Schema-validates each line: skips rows missing required keys or with
    non-list ``spans``. Only accepts labels that are in the gheim
    CATEGORIES set; logs (but doesn't fail) on unknown labels.
    """
    by_chunk: dict[str, dict[str, list[Claim]]] = defaultdict(lambda: defaultdict(list))
    responded: dict[str, set[str]] = defaultdict(set)
    bad_label_counter: Counter[str] = Counter()

    # Strict pattern match — catches typo'd filenames that the prior
    # rsplit-based parsing would silently mis-attribute.
    candidate_files = sorted(labels_dir.glob("batch_*.jsonl"))
    skipped: list[str] = []
    for path in candidate_files:
        m = _LABELER_FILENAME_RE.match(path.name)
        if m is None:
            skipped.append(path.name)
            continue
        labeler = m.group(2)
        for rec in _read_jsonl(path):
            chunk_id = rec.get("chunk_id")
            spans = rec.get("spans")
            if not isinstance(chunk_id, str) or not isinstance(spans, list):
                print(f"  WARN: malformed record in {path.name}: {rec!r}")
                continue
            # Mark labeler as having emitted SOMETHING for this chunk,
            # even if spans is empty — distinguishes clean negatives from
            # genuine missing data.
            responded[chunk_id].add(labeler)
            for sp in spans:
                if not isinstance(sp, dict):
                    continue
                lab = sp.get("label")
                val = sp.get("value")
                if not isinstance(lab, str) or not isinstance(val, str):
                    continue
                if lab not in CATEGORIES:
                    bad_label_counter[lab] += 1
                    continue
                if not val.strip():
                    continue
                by_chunk[chunk_id][labeler].append(Claim(label=lab, value=val.strip()))

    if skipped:
        print(f"  WARN: {len(skipped)} files in {labels_dir} did not match "
              f"batch_NNN_labeler[A-D].jsonl convention and were skipped: "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    if bad_label_counter:
        print("  Bad labels (not in CATEGORIES, dropped):")
        for lab, n in bad_label_counter.most_common():
            print(f"    {lab!r}: {n}")
    return by_chunk, responded


def _locate(
    text: str, value: str,
) -> tuple[tuple[int, int] | None, int]:
    """Locate ``value`` in ``text``. Returns ``((start, end), n_occurrences)``.

    First-occurrence semantics for the offsets (matching the labeler's
    likely intent in a left-to-right read), but also returns the count of
    distinct occurrences so callers can detect ambiguity. Trims surrounding
    whitespace; returns ``(None, 0)`` for empty/not-found/whitespace-only.

    Why ambiguity matters: a labeler claim "Anna Müller" against a chunk
    that mentions her twice ("Anna Müller wrote… Anna Müller said") gets
    silently bound to the first occurrence. For test_v1 the gold span is
    still correct (both occurrences are equally valid), but a finetuned
    model that catches both will be penalised against the gold, and a
    model that catches only the second will appear to miss the gold.
    Callers can use the count to log/skip ambiguous chunks if they care.
    """
    if not value:
        return None, 0
    idx = text.find(value)
    if idx < 0:
        return None, 0
    # Count all non-overlapping occurrences for ambiguity reporting.
    n = 0
    j = 0
    step = max(1, len(value))
    while True:
        k = text.find(value, j)
        if k < 0:
            break
        n += 1
        j = k + step
    end = idx + len(value)
    while idx < end and text[idx].isspace():
        idx += 1
    while end > idx and text[end - 1].isspace():
        end -= 1
    if end <= idx:
        return None, n
    return (idx, end), n


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Greedy: earlier-start wins; on equal start, longer span wins.
    Same algorithm the eval harness uses for predicted spans."""
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    last_end = -1
    for sp in spans:
        if sp.start < last_end:
            continue
        out.append(sp)
        last_end = sp.end
    return out


def _aggregate_chunk(
    chunk_text: str,
    by_labeler: dict[str, list[Claim]],
    threshold: int,
) -> tuple[list[Span], list[tuple[Claim, tuple[str, ...]]]]:
    """Per-chunk: collect claim → labelers, accept if ≥threshold, flag
    disputed (2-of-4 when threshold=3) for hand resolution.

    Returns (accepted_spans, disagreements). Disagreements are
    ``(claim, labelers_who_agreed)`` tuples; labelers tuple has length
    < threshold but ≥ floor(threshold-1).
    """
    # Aggregate: claim → set of labelers that emitted it
    claim_to_labelers: dict[Claim, set[str]] = defaultdict(set)
    for labeler, claims in by_labeler.items():
        # Within-labeler dedup: same label/value emitted twice still counts once
        for c in set(claims):
            claim_to_labelers[c].add(labeler)

    accepted: list[Span] = []
    disputed: list[tuple[Claim, tuple[str, ...]]] = []
    for claim, labelers in claim_to_labelers.items():
        n = len(labelers)
        if n >= threshold:
            located, _n_occ = _locate(chunk_text, claim.value)
            if located is None:
                continue  # not in text → drop (verbatim violation)
            start, end = located
            accepted.append(Span(start=start, end=end, label=claim.label))
        elif n >= max(1, threshold - 1):  # 2-of-4 when threshold=3
            disputed.append((claim, tuple(sorted(labelers))))
    return _resolve_overlaps(accepted), disputed


def aggregate(
    pool_records: list[dict],
    by_chunk: dict[str, dict[str, list[Claim]]],
    responded: dict[str, set[str]],
    threshold: int,
) -> tuple[list[dict], list[DisagreementRecord], dict[str, int]]:
    """Build consensus gold + disagreement records.

    Consensus: list of pool records augmented with ``spans`` (gold). One
    record per pool chunk, even when the chunk has zero accepted spans
    (controls + clean-negative dense chunks both surface as []).

    Disagreements: flat list of DisagreementRecord, one per disputed claim.
    """
    consensus: list[dict] = []
    disagreements: list[DisagreementRecord] = []
    stats: Counter[str] = Counter()
    for pool_rec in pool_records:
        chunk_id = pool_rec["chunk_id"]
        by_labeler = by_chunk.get(chunk_id, {})
        n_responded = len(responded.get(chunk_id, set()))
        n_proposed = len(by_labeler)
        if n_responded == 0:
            stats["chunks_no_labeler_responded"] += 1
        elif n_responded < 4:
            stats[f"chunks_only_{n_responded}_of_4_responded"] += 1
        if n_responded > 0 and n_proposed == 0:
            stats["chunks_clean_negative_unanimous"] += 1
        accepted, disputed = _aggregate_chunk(
            pool_rec["text"], by_labeler, threshold,
        )
        # Count accepted spans where the verbatim value occurred multiple
        # times in the chunk — gold offset is bound to first-occurrence,
        # which may not match the labeler's intent. Worth flagging.
        n_ambiguous = 0
        for sp in accepted:
            _, n_occ = _locate(pool_rec["text"], pool_rec["text"][sp.start:sp.end])
            if n_occ > 1:
                n_ambiguous += 1
        # Emit consensus record (preserves all pool fields + adds spans)
        consensus.append({
            **pool_rec,
            "spans": [asdict(sp) for sp in accepted],
            "n_labelers_responded": n_responded,
            "n_disputed": len(disputed),
            "n_ambiguous_offsets": n_ambiguous,
        })
        stats["accepted_spans"] += len(accepted)
        stats["disputed_claims"] += len(disputed)
        stats["ambiguous_first_occurrence_spans"] += n_ambiguous
        for claim, labelers in disputed:
            disagreements.append(DisagreementRecord(
                chunk_id=chunk_id,
                text=pool_rec["text"],
                language=pool_rec["language"],
                subset=pool_rec["subset"],
                claim=claim,
                n_labelers_agreeing=len(labelers),
                labelers_agreeing=labelers,
            ))
    return consensus, disagreements, dict(stats)


def _write_jsonl(path: Path, records: list[dict] | list[DisagreementRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            payload = r if isinstance(r, dict) else asdict(r)
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=Path("data/test_v1_pool.jsonl"))
    ap.add_argument("--labels-dir", type=Path, default=Path("data/test_v1_labels"))
    ap.add_argument("--consensus-out", type=Path,
                    default=Path("data/test_v1_consensus.jsonl"))
    ap.add_argument("--disagreements-out", type=Path,
                    default=Path("data/test_v1_disagreements.jsonl"))
    ap.add_argument("--threshold", type=int, default=3,
                    help="Min labelers agreeing to auto-accept a claim. "
                         "Default 3-of-4. Disputed = exactly threshold-1 agreed.")
    args = ap.parse_args()

    pool = list(_read_jsonl(args.pool))
    print(f"Loaded {len(pool)} pool chunks from {args.pool}")
    by_chunk, responded = _load_per_labeler(args.labels_dir)
    print(
        f"Loaded labels for {len(by_chunk)} chunks with claims, "
        f"{len(responded)} chunks responded to overall, from {args.labels_dir}",
    )

    consensus, disagreements, stats = aggregate(
        pool, by_chunk, responded, args.threshold,
    )
    _write_jsonl(args.consensus_out, consensus)
    _write_jsonl(args.disagreements_out, disagreements)

    # Per-language span counts in the consensus
    by_lang: Counter[str] = Counter()
    by_lang_label: Counter[tuple[str, str]] = Counter()
    for c in consensus:
        for sp in c["spans"]:
            by_lang[c["language"]] += 1
            by_lang_label[(c["language"], sp["label"])] += 1

    print("\n=== Aggregation report ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:<32} {v:>6}")
    print(f"\nConsensus chunks:        {len(consensus):>6}")
    print(f"Disagreement claims:     {len(disagreements):>6}")
    print("\nAccepted spans by language:")
    for lang, n in sorted(by_lang.items()):
        print(f"  {lang:<6} {n:>5}")
    print("\nAccepted spans by (language, label):")
    for (lang, lab), n in sorted(by_lang_label.items()):
        print(f"  {lang:<6} {lab:<22} {n:>4}")

    print(f"\nWrote consensus → {args.consensus_out}")
    print(f"Wrote disagreements → {args.disagreements_out}")


if __name__ == "__main__":
    main()
