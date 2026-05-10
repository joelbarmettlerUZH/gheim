"""Coverage gate: refuse to train if any (language × category) cell is too thin.

The gheim-1 dataset had cells with literally zero training examples for
some categories (e.g. ``secret`` only existed in Layer 1 synthetic across
the whole 363k-example corpus). The model learned to predict ``O`` on
those cells.

This module reads the assembled gold-core layer files, prints a
per-(language × category) coverage matrix, and exits non-zero if any cell
falls below ``--min-per-cell`` (default 1000) — unless explicitly waived
with ``--waive lang/cat`` arguments.

Run before any training build:
    uv run python -m gheim_training.data.coverage_gate \\
        --layers data/layer1.jsonl data/layer2_v2.jsonl data/layer4_v2.jsonl \\
        --min-per-cell 1000

Or via build.py which calls this automatically.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .label_space import CATEGORIES
from .schema import read_jsonl

_LANGUAGES: tuple[str, ...] = ("de_ch", "fr_ch", "it_ch", "rm", "gsw", "en")


def count_examples(layers: Iterable[Path]) -> dict[tuple[str, str], int]:
    """Count examples per (language, category) — an example contributes
    once per category it contains, not once per span. Sparse cells get 0."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for path in layers:
        if not path.exists():
            print(f"[coverage_gate] WARN: {path} does not exist; skipping",
                  file=sys.stderr)
            continue
        for ex in read_jsonl(path):
            cats_in_example: set[str] = {sp.label for sp in ex.spans}
            for cat in cats_in_example:
                counts[(ex.language, cat)] += 1
    return counts


def report(
    counts: dict[tuple[str, str], int],
    *,
    min_per_cell: int,
    waivers: set[tuple[str, str]],
) -> tuple[bool, list[str]]:
    """Print per-cell counts. Return (passed, failures)."""
    failures: list[str] = []
    print(f"\n{'category':<22}  " + "  ".join(f"{lang:>8}" for lang in _LANGUAGES))
    print("-" * (24 + 10 * len(_LANGUAGES)))
    for cat in CATEGORIES:
        cells = []
        for lang in _LANGUAGES:
            n = counts.get((lang, cat), 0)
            waived = (lang, cat) in waivers
            if n >= min_per_cell:
                marker = "✓"
            elif waived:
                marker = "w"
            else:
                marker = "✗"
                failures.append(f"{lang}/{cat}: {n} (need {min_per_cell})")
            cells.append(f"{marker}{n:>7}")
        print(f"  {cat:<20}  " + "  ".join(cells))
    print()
    print("Legend: ✓ ≥ threshold   w waived   ✗ below threshold")
    return (not failures, failures)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=Path, nargs="+", required=True,
                    help="JSONL files to count across.")
    ap.add_argument("--min-per-cell", type=int, default=1000,
                    help="Minimum examples per (language × category).")
    ap.add_argument("--waive", action="append", default=[],
                    help="Cells to waive: lang/category (e.g. gsw/secret). Repeatable.")
    args = ap.parse_args()

    waivers: set[tuple[str, str]] = set()
    for w in args.waive:
        try:
            lang, cat = w.split("/", 1)
        except ValueError:
            print(f"--waive expects lang/category, got {w!r}", file=sys.stderr)
            sys.exit(2)
        waivers.add((lang, cat))

    counts = count_examples(args.layers)
    passed, failures = report(counts, min_per_cell=args.min_per_cell,
                              waivers=waivers)
    if passed:
        print(f"PASS: every (language × category) cell ≥ {args.min_per_cell} or waived.")
        sys.exit(0)
    print(f"\nFAIL: {len(failures)} cells below threshold:")
    for f in failures:
        print(f"  - {f}")
    print("\nOptions:")
    print("  - Add training data for the missing cells (Phase B gap-fill)")
    print("  - Waive cells where you accept the gap (e.g. gsw/secret) "
          "via --waive lang/category")
    sys.exit(1)


if __name__ == "__main__":
    main()
