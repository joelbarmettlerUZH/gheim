"""P17: Cleanup the published-dataset schema.

Drops chunk-level fields that were only ever pipeline-internal:
  - labeler          (per-span 'source' now covers provenance)
  - prompt_version   (internal)
  - labeled_at       (internal)
  - language_old     (pre-redetection label; pipeline trauma)
  - language_raw     (raw NLLB label, redundant with 'language' for in-scope langs)

Keeps:
  - id, text, language, language_confidence,
    subset, source_dataset, doc_id, chunk_index_in_doc,
    spans, synthetic, synthetic_template,
    ai4p_region, ai4p_lang_code (for AI4Privacy-augmented chunks in train_mix only)

Per-span: ensures every span has an explicit 'source' field. For Gemma-labelled
spans (which previously had no source field), adds source: "gemma". Existing
"regex" and "synthetic" sources are preserved.

Operates in-place on:
  data/gheim_pii_balanced_train.jsonl
  data/gheim_pii_balanced_validation.jsonl   (chmod 444 → 644 → 444)
  data/gheim_pii_balanced_test.jsonl         (chmod 444 → 644 → 444)
  data/gheim_synthetic.jsonl
  data/gheim_train_mix.jsonl

Safe to run while training is in progress: all training jobs read from
data/built_v2 (loaded into memory at job start), not from these JSONL
files directly. Use P11 to rebuild data/built_v2 after this.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
from collections import Counter
from pathlib import Path

DROP_FIELDS = (
    "labeler",
    "prompt_version",
    "labeled_at",
    "language_old",
    "language_raw",
)

FILES = [
    Path("data/gheim_pii_balanced_train.jsonl"),
    Path("data/gheim_pii_balanced_validation.jsonl"),
    Path("data/gheim_pii_balanced_test.jsonl"),
    Path("data/gheim_synthetic.jsonl"),
    Path("data/gheim_train_mix.jsonl"),
]


def _writable(path: Path) -> None:
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _is_locked_eval(path: Path) -> bool:
    return path.name in ("gheim_pii_balanced_validation.jsonl",
                         "gheim_pii_balanced_test.jsonl")


def _convert(rec: dict) -> tuple[dict, dict[str, int]]:
    """Return (cleaned_record, per-source span counter)."""
    out = {k: v for k, v in rec.items() if k not in DROP_FIELDS}

    # Per-span: ensure 'source' field is present.
    src_counts: Counter[str] = Counter()
    new_spans = []
    for sp in out.get("spans", []):
        sp = dict(sp)
        if "source" not in sp:
            sp["source"] = "gemma"
        new_spans.append(sp)
        src_counts[sp["source"]] += 1
    out["spans"] = new_spans
    return out, src_counts


def _process_file(path: Path, *, write: bool) -> dict:
    if not path.exists():
        return {"path": str(path), "skipped": True}

    print(f"Reading: {path}", flush=True)
    n_total = 0
    src_global: Counter[str] = Counter()
    cleaned: list[str] = []  # list of JSON lines
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            new_rec, src_counts = _convert(rec)
            src_global.update(src_counts)
            cleaned.append(json.dumps(new_rec, ensure_ascii=False) + "\n")

    if write:
        was_locked = _is_locked_eval(path)
        if was_locked:
            _writable(path)
        with path.open("w") as f:
            f.writelines(cleaned)
        if was_locked:
            _readonly(path)

    return {
        "path": str(path),
        "n_chunks": n_total,
        "spans_by_source": dict(src_global),
        "wrote": write,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Rewrite files in place (default: dry-run)")
    args = parser.parse_args()

    print("Schema cleanup:")
    print(f"  drop chunk-level fields: {DROP_FIELDS}")
    print(f"  ensure every span has 'source': defaults to 'gemma' if missing")
    print()

    results = []
    for path in FILES:
        results.append(_process_file(path, write=args.write))

    print()
    print(f"{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'file':<48} {'chunks':>10} {'gemma':>8} {'regex':>8} {'synth':>8}")
    for r in results:
        if r.get("skipped"):
            print(f"{r['path']:<48} (missing)")
            continue
        sb = r.get("spans_by_source", {})
        print(f"{r['path']:<48} {r['n_chunks']:>10,} "
              f"{sb.get('gemma', 0):>8,} {sb.get('regex', 0):>8,} "
              f"{sb.get('synthetic', 0):>8,}")

    if not args.write:
        print()
        print("[DRY-RUN] No files modified. Re-run with --write.")
    else:
        print()
        print("Files rewritten. Run P11 next to rebuild data/built_v2.")


if __name__ == "__main__":
    main()
