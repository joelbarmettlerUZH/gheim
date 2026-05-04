"""Day 6 finalizer: merge 3+/4 consensus with hand-resolved 2-of-4 spans.

Produces ``data/test_v1.jsonl`` — the frozen gold test set for the bake-off.
The format matches ``audit_gold.jsonl`` and ``address_gold_v1.jsonl`` so the
existing eval harness reads it without modification.

Inputs:
- ``data/test_v1_consensus.jsonl`` — chunks with their accepted (3+/4)
  spans plus pool metadata (chunk_id, language, subset, presence, …).
- ``data/test_v1_resolved.jsonl`` — hand-resolution decisions for the
  2-of-4 disagreements: ``{chunk_id, claim {label, value}, decision,
  reason}``. Only ``decision == "accept"`` claims are added back.

Output schema per JSONL line::

    {
      "chunk_id": "T000001",
      "text": "...",
      "language": "de_ch",
      "source": "fineweb",  # was "subset" in pool, mapped to harness "source"
      "spans": [{"start": 0, "end": 11, "label": "private_person"}, ...],
      "presence": "pii_dense",
      "doc_id": "..."
    }

After write the file is locked ``chmod 444`` and backed up to
``data/_backups/``. Subsequent attempts to re-run will fail loudly unless
the file is unlinked first — intentional, this is the held-out set.

Run:
    uv run python -m gheim_training.eval.finalize_test_v1 \\
        --consensus data/test_v1_consensus.jsonl \\
        --resolved data/test_v1_resolved.jsonl \\
        --out data/test_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import stat
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from ..data.label_space import CATEGORIES
from ..data.schema import Span


def _read_jsonl(path: Path) -> Iterator[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _locate(text: str, value: str) -> tuple[int, int] | None:
    """First-occurrence locate of ``value`` in ``text``, with whitespace trim."""
    if not value:
        return None
    idx = text.find(value)
    if idx < 0:
        return None
    end = idx + len(value)
    while idx < end and text[idx].isspace():
        idx += 1
    while end > idx and text[end - 1].isspace():
        end -= 1
    if end <= idx:
        return None
    return idx, end


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Greedy overlap resolution: earlier-start, then longer-span wins."""
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    last_end = -1
    for sp in spans:
        if sp.start < last_end:
            continue
        out.append(sp)
        last_end = sp.end
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--consensus", type=Path,
                    default=Path("data/test_v1_consensus.jsonl"))
    ap.add_argument("--resolved", type=Path,
                    default=Path("data/test_v1_resolved.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/test_v1.jsonl"))
    ap.add_argument("--backup-dir", type=Path,
                    default=Path("data/_backups"))
    args = ap.parse_args()

    # Hard-fail if test_v1 already exists (held-out set should be written
    # exactly once; any re-run is a discipline error, not a routine action).
    if args.out.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE {args.out}. The held-out test set should "
            f"only be written once. Delete the file manually if you really "
            f"intend to regenerate it."
        )

    # Index the resolved decisions by (chunk_id, label, value) — accept-only
    accepted_resolutions: dict[tuple[str, str, str], str] = {}
    n_resolved_accept = n_resolved_reject = 0
    if args.resolved.exists():
        for r in _read_jsonl(args.resolved):
            cid = r["chunk_id"]
            claim = r["claim"]
            decision = r.get("decision")
            if decision == "accept":
                key = (cid, claim["label"], claim["value"].strip())
                accepted_resolutions[key] = r.get("reason", "")
                n_resolved_accept += 1
            elif decision == "reject":
                n_resolved_reject += 1
            else:
                print(f"  WARN: unknown decision {decision!r} for {cid} — treating as reject")
                n_resolved_reject += 1
    print(
        f"Loaded {n_resolved_accept} accepted + {n_resolved_reject} rejected "
        f"hand-resolutions from {args.resolved}",
    )

    # Walk consensus chunks; for each, append accepted resolutions and
    # re-resolve overlaps so the gold is clean.
    out_records: list[dict] = []
    counts: Counter[str] = Counter()
    by_lang_label: Counter[tuple[str, str]] = Counter()
    by_chunk_lookup: dict[str, dict] = {}
    for c in _read_jsonl(args.consensus):
        by_chunk_lookup[c["chunk_id"]] = c

    n_consensus_spans = 0
    n_added_resolutions = 0
    for cid, c in by_chunk_lookup.items():
        text = c["text"]
        spans = [Span(**sp) for sp in c["spans"]]
        n_consensus_spans += len(spans)

        # Add accepted resolutions for this chunk
        chunk_keys = [k for k in accepted_resolutions if k[0] == cid]
        for _, label, value in chunk_keys:
            if label not in CATEGORIES:
                counts["dropped_bad_label"] += 1
                continue
            located = _locate(text, value)
            if located is None:
                counts["dropped_unlocatable"] += 1
                continue
            spans.append(Span(start=located[0], end=located[1], label=label))
            n_added_resolutions += 1

        # Resolve any overlaps between consensus + new accepts
        spans = _resolve_overlaps(spans)

        # Map "subset" → "source" to match harness convention
        out_records.append({
            "chunk_id": cid,
            "text": text,
            "language": c["language"],
            "source": c["subset"],
            "spans": [{"start": sp.start, "end": sp.end, "label": sp.label}
                       for sp in spans],
            "presence": c["presence"],
            "doc_id": c.get("doc_id", ""),
        })
        for sp in spans:
            by_lang_label[(c["language"], sp.label)] += 1

    # Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

    # Lock + backup
    args.out.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 444
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = args.backup_dir / args.out.name.replace(".jsonl", ".jsonl.bak")
    if backup_path.exists():
        backup_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
    shutil.copy(args.out, backup_path)
    backup_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    # Final stats
    final_total_spans = sum(len(r["spans"]) for r in out_records)
    final_chunks_with_spans = sum(1 for r in out_records if r["spans"])
    print("\n=== test_v1 finalized ===")
    print(f"Chunks:                     {len(out_records)}")
    print(f"Spans (consensus):          {n_consensus_spans}")
    print(f"Spans (added from resolved):{n_added_resolutions}")
    print(f"Spans (final, post-overlap):{final_total_spans}")
    print(f"Chunks with spans:          {final_chunks_with_spans}")
    print(f"Clean negatives:            {len(out_records) - final_chunks_with_spans}")
    if counts:
        print("Dropped during merge:")
        for k, n in counts.most_common():
            print(f"  {k:<28} {n}")
    print("\nFinal per-(language, label):")
    for lang, label in sorted(by_lang_label.keys()):
        print(f"  {lang:<6} {label:<22} {by_lang_label[(lang, label)]:>4}")
    by_lang: dict[str, int] = defaultdict(int)
    for (lang, _), n in by_lang_label.items():
        by_lang[lang] += n
    print("\nPer-language totals:")
    for lang in sorted(by_lang):
        print(f"  {lang:<6} {by_lang[lang]:>4}")

    print(f"\nLocked:  {args.out} (chmod 444)")
    print(f"Backed up: {backup_path} (chmod 444)")


if __name__ == "__main__":
    main()
