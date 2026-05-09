"""P1a-ii: Regex-augment existing layer5v4 chunks.

For each chunk in data/layer5v4.jsonl, run the validated regex catalog
(from gheim.detectors.composite._find_regex_spans) and merge any new
spans that don't overlap an existing Gemma-labeled span. Preserves all
original Gemma spans.

Output: data/layer5v4_regex_aug.jsonl with merged spans, plus a stats
report on per-(lang × cat) regex hits.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Add packages/gheim-py/src to the path so we can import the validators.
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "packages" / "gheim-py" / "src"))

from gheim.detectors.composite import _find_regex_spans  # type: ignore[import-not-found]


IN_PATH = Path("data/layer5v4.jsonl")
OUT_PATH = Path("data/layer5v4_regex_aug.jsonl")


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def main() -> None:
    n_chunks = 0
    n_chunks_with_regex = 0
    n_existing_spans = 0
    n_added_spans = 0
    n_dropped_overlap = 0

    added_by_lang_cat: Counter[tuple[str, str]] = Counter()
    chunks_changed_by_lang: Counter[str] = Counter()

    with IN_PATH.open() as fin, OUT_PATH.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            n_chunks += 1
            lang = rec.get("language", "")
            text = rec["text"]
            existing = rec.get("spans", [])
            n_existing_spans += len(existing)

            regex_spans = _find_regex_spans(text)
            if not regex_spans:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            additions: list[dict] = []
            for sp in regex_spans:
                # Drop if overlaps any existing Gemma span (Gemma's call wins).
                if any(
                    _spans_overlap(sp.start, sp.end, e["start"], e["end"])
                    for e in existing
                ):
                    n_dropped_overlap += 1
                    continue
                additions.append({
                    "start": sp.start,
                    "end": sp.end,
                    "label": sp.label,
                    "value": text[sp.start:sp.end],
                    "source": "regex",
                })
                added_by_lang_cat[(lang, sp.label)] += 1

            if additions:
                # Mark existing spans as gemma-source for traceability.
                merged = []
                for e in existing:
                    e2 = dict(e)
                    e2.setdefault("source", "gemma")
                    merged.append(e2)
                merged.extend(additions)
                merged.sort(key=lambda s: s["start"])
                rec["spans"] = merged
                n_chunks_with_regex += 1
                n_added_spans += len(additions)
                chunks_changed_by_lang[lang] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if n_chunks % 100_000 == 0:
                print(f"  scanned {n_chunks:,} chunks, "
                      f"{n_chunks_with_regex:,} augmented, "
                      f"+{n_added_spans:,} spans added", flush=True)

    print()
    print(f"DONE — {n_chunks:,} chunks scanned")
    print(f"  existing Gemma spans:       {n_existing_spans:,}")
    print(f"  regex spans added:          {n_added_spans:,}")
    print(f"  regex spans dropped (overlap with Gemma): {n_dropped_overlap:,}")
    print(f"  chunks gained ≥1 regex span: {n_chunks_with_regex:,}")
    print(f"  output: {OUT_PATH}")
    print()
    print("Added spans by (lang × category):")
    LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
    cats = sorted({c for _, c in added_by_lang_cat})
    for cat in cats:
        row = " ".join(
            f"{added_by_lang_cat.get((la, cat), 0):>7}" for la in LANGS
        )
        print(f"  {cat:<22} {row}")
    print()
    print("Chunks changed by language:")
    for la in LANGS:
        print(f"  {la:<8} {chunks_changed_by_lang.get(la, 0):,}")


if __name__ == "__main__":
    main()
