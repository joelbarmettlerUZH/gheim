"""Build a tokens+ner_tags eval set from ai4privacy/pii-masking-200k for
replicating Isotonic's published F1=0.955 claim.

The dataset only ships a `train` split, but Isotonic was trained on it and
reports F1 on a held-out portion. We sample N records from the END of the
train split (approximate held-out) and produce whitespace-tokens with
BIO tags projected from `privacy_mask` (char-level spans).

Output is consumable by replicate_native.py with --tokens-col tokens
--tags-col ner_tags.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


def whitespace_tokens_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Return list of (token, start, end) using a simple word/punct tokenizer."""
    out = []
    for m in re.finditer(r"\S+", text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def project_spans_to_bio(
    tokens: list[tuple[str, int, int]],
    spans: list[dict],
) -> list[str]:
    """Project char-offset spans onto whitespace tokens as BIO tags.

    A token is in the span if their character ranges intersect at all (≥1
    char overlap), not just full containment — handles the case where the
    annotator's span ends mid-token (rare but does happen in pii-200k).
    """
    tags = ["O"] * len(tokens)
    for sp in spans:
        s_start = int(sp.get("start", -1))
        s_end = int(sp.get("end", -1))
        label = sp.get("label")
        if not label or s_start < 0 or s_end <= s_start:
            continue
        first = True
        for i, (_, t_start, t_end) in enumerate(tokens):
            if t_end <= s_start or t_start >= s_end:
                continue
            tags[i] = ("B-" if first else "I-") + label
            first = False
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000,
                        help="Sample size (taken from end of train; rough held-out proxy)")
    parser.add_argument("--language", type=str, default="en",
                        help="Language code to filter (Isotonic is en-only)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from datasets import Dataset, DatasetDict, load_dataset

    print(f"Streaming pii-masking-200k train, sampling {args.n} END rows in {args.language!r} ...",
          flush=True)
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True)
    rows = []
    label_counts: Counter[str] = Counter()
    n_seen = 0
    # Streaming dataset has no len, so we materialise the whole language slice
    # and keep the last N rows below as a rough proxy for the held-out tail.
    for r in ds:
        n_seen += 1
        if r.get("language") != args.language:
            continue
        rows.append(r)
        if n_seen % 50_000 == 0:
            print(f"  scanned {n_seen:,} (kept {len(rows):,})", flush=True)
    rows = rows[-args.n:]
    print(f"  total {n_seen:,} scanned; using last {len(rows)} {args.language!r} rows", flush=True)

    flat = []
    for r in rows:
        text = r.get("source_text") or ""
        if not text:
            continue
        toks = whitespace_tokens_with_offsets(text)
        bio = project_spans_to_bio(toks, r.get("privacy_mask") or [])
        for t in bio:
            label_counts[t] += 1
        flat.append({
            "tokens": [t[0] for t in toks],
            "ner_tags": bio,
        })

    print(f"\n  built {len(flat):,} rows; tag distribution (top 15):", flush=True)
    for k, v in label_counts.most_common(15):
        print(f"    {k:<20} {v}")

    dd = DatasetDict({"test": Dataset.from_list(flat)})
    args.out.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(args.out))
    print(f"\n  saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
