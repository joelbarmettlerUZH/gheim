"""V3-3: post-process v2_assembled.jsonl with the v3 merge fixes.

Reads ``data/v2_assembled.jsonl``, applies the v3 post-merge fixes
(`_apply_swiss_place_gazetteer` + `_resolve_label_conflicts`) to each
chunk's spans, and writes ``data/v3_assembled.jsonl``.

This is functionally equivalent to re-running V2-6 ``assemble.py`` with
``apply_v3_fixes=True``, but ~20× faster because we operate on
already-merged V2Spans instead of re-loading + re-merging the source
gemma/qwen/nemotron JSONL files. Same outputs, much less I/O.

Reports the impact of the v3 fixes per chunk + in aggregate (how many
gazetteer demotions, how many overlap resolutions).

Run
---
    uv run python -m gheim_training.data.v2.v3_postprocess
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .merge import _apply_swiss_place_gazetteer, _resolve_label_conflicts
from .schema import V2Example, V2Span

DEFAULT_IN = Path("data/v2_assembled.jsonl")
DEFAULT_OUT = Path("data/v3_assembled.jsonl")
DEFAULT_REPORT = Path("data/v3_postprocess_report.json")
N_CANDIDATE_SIGNALS = 4  # the standard v2 build (gemma + qwen + nemotron + regex)


def _process_chunk(spans: list[V2Span]) -> tuple[list[V2Span], dict]:
    """Apply v3 fixes to one chunk's spans. Returns (new_spans, stats)."""
    n_before = len(spans)
    # Count what the gazetteer changes
    n_gazetteer_demoted = sum(
        1 for sp in spans
        if sp.label == "private_person"
        and sp.value.casefold().strip()
        in _PLACE_SET_CACHE.get_cached_set()
    )
    out = _apply_swiss_place_gazetteer(spans)
    n_after_gazetteer = len(out)
    out = _resolve_label_conflicts(out, n_candidate_signals=N_CANDIDATE_SIGNALS)
    n_after_resolve = len(out)
    return out, {
        "n_before": n_before,
        "n_after_gazetteer": n_after_gazetteer,
        "n_after_resolve": n_after_resolve,
        "n_demoted": n_gazetteer_demoted,
        "n_resolved": n_after_gazetteer - n_after_resolve,
    }


class _PlaceSetCache:
    """Cache the Geonames place-name set once for the entire run."""
    _set: frozenset[str] | None = None

    def get_cached_set(self) -> frozenset[str]:
        if self._set is None:
            from ..synthetic.swiss_geo import all_places
            self._set = frozenset(
                p.city.casefold().strip() for p in all_places()
            )
        return self._set


_PLACE_SET_CACHE = _PlaceSetCache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    # Warm the place-set cache before iterating
    print(f"Loading Swiss place-name gazetteer…", flush=True)
    place_set = _PLACE_SET_CACHE.get_cached_set()
    print(f"  {len(place_set):,} unique place names")
    print()

    print(f"Reading {args.in_path}…", flush=True)
    n_chunks = 0
    n_chunks_changed = 0
    total_spans_before = 0
    total_spans_after = 0
    total_demotions = 0
    total_resolutions = 0
    demotions_by_lang: Counter[str] = Counter()
    resolutions_by_label_pair: Counter[tuple[str, str]] = Counter()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.in_path.open() as f_in, args.out_path.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            ex = V2Example.from_json(line)
            n_chunks += 1
            total_spans_before += len(ex.spans)

            # Track resolutions by their pre-change label pairs (so we
            # can report "address × person" type collisions in aggregate)
            spans_before = list(ex.spans)
            # First check what gazetteer will demote (for stats)
            demoted_ids = [
                i for i, sp in enumerate(spans_before)
                if sp.label == "private_person"
                and sp.value.casefold().strip() in place_set
            ]
            # Pre-compute (start, end) → labels for conflict reporting
            by_pos_before: dict[tuple[int, int], list[str]] = {}
            for sp in spans_before:
                by_pos_before.setdefault((sp.start, sp.end), []).append(sp.label)
            n_conflicts = sum(
                1 for labels in by_pos_before.values()
                if len(set(labels)) > 1
            )

            new_spans, _stats = _process_chunk(ex.spans)
            ex.spans = new_spans
            total_spans_after += len(new_spans)
            n_changed = (len(spans_before) != len(new_spans)
                         or any(a.label != b.label
                                for a, b in zip(spans_before, new_spans)))
            if n_changed:
                n_chunks_changed += 1

            if demoted_ids:
                total_demotions += len(demoted_ids)
                demotions_by_lang[ex.language] += len(demoted_ids)
            if n_conflicts:
                total_resolutions += n_conflicts
                for labels in by_pos_before.values():
                    s = set(labels)
                    if len(s) > 1:
                        pair = tuple(sorted(s))
                        resolutions_by_label_pair[(pair[0], pair[1])] += 1

            f_out.write(ex.to_json())
            f_out.write("\n")

            if n_chunks % 250_000 == 0:
                print(f"  processed {n_chunks:,} chunks…", flush=True)

    print()
    print(f"Wrote {args.out_path}")
    print()
    print("=" * 60)
    print("V3-3 post-process summary")
    print("=" * 60)
    print(f"Chunks processed:           {n_chunks:,}")
    print(f"Chunks with at least one change: {n_chunks_changed:,} "
          f"({100*n_chunks_changed/n_chunks:.3f}%)")
    print()
    print(f"Spans before:               {total_spans_before:,}")
    print(f"Spans after:                {total_spans_after:,}")
    print(f"  delta:                    {total_spans_after - total_spans_before:+,}")
    print()
    print(f"Gazetteer demotions (private_person → private_address): {total_demotions:,}")
    print(f"  by language: {dict(demotions_by_lang)}")
    print()
    print(f"Label-conflict resolutions at same (start, end): {total_resolutions:,}")
    print(f"  by label pair:")
    for (a, b), n in resolutions_by_label_pair.most_common():
        print(f"    {a:<22} × {b:<22}  {n:>5}")

    report = {
        "_meta": {
            "in_path": str(args.in_path),
            "out_path": str(args.out_path),
            "n_candidate_signals": N_CANDIDATE_SIGNALS,
            "place_set_size": len(place_set),
        },
        "n_chunks": n_chunks,
        "n_chunks_changed": n_chunks_changed,
        "spans_before": total_spans_before,
        "spans_after": total_spans_after,
        "gazetteer_demotions": total_demotions,
        "gazetteer_demotions_by_language": dict(demotions_by_lang),
        "label_conflict_resolutions": total_resolutions,
        "label_conflict_pairs": {
            f"{a}__{b}": n
            for (a, b), n in resolutions_by_label_pair.most_common()
        },
    }
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.report_path}")


if __name__ == "__main__":
    main()
