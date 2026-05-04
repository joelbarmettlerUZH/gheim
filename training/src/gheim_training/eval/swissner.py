"""ZurichNLP SwissNER loader → Examples (only PERSON/LOC mapped to our categories).

Used as a held-out behavioural eval set for names + addresses. SwissNER
labels: PER, LOC, ORG. We drop ORG (no gheim category) and treat LOC as
private_address (a stretch — LOC includes place names not full addresses —
but consistent with how SwissBERT-NER's LOC predictions are scored).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.schema import Example, Span, write_jsonl

LABEL_MAP = {"PER": "private_person", "LOC": "private_address"}


def load(dataset_name: str = "ZurichNLP/swissner", split: str = "test") -> list[Example]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    out: list[Example] = []
    for rec in ds:
        # Schema varies between SwissNER releases; expect either token+tag arrays
        # or a flat (text, spans) form. Handle the token+tag form here.
        tokens = rec.get("tokens") or rec.get("words")
        tags = rec.get("ner_tags") or rec.get("tags") or rec.get("labels")
        if not tokens or not tags:
            continue
        if isinstance(tags[0], int):
            tag_names = ds.features["ner_tags"].feature.names
            tags = [tag_names[t] for t in tags]
        text = " ".join(tokens)
        # Reconstruct spans by walking the BIO tag sequence.
        spans: list[Span] = []
        cursor = 0
        cur_cat: str | None = None
        cur_start: int | None = None
        cur_end: int | None = None
        for tok, tag in zip(tokens, tags, strict=True):
            tok_start = cursor
            cursor += len(tok)
            tok_end = cursor
            cursor += 1  # the joining space (last iter overshoots; harmless)
            if tag == "O":
                if cur_cat:
                    spans.append(Span(start=cur_start, end=cur_end, label=cur_cat))
                cur_cat = cur_start = cur_end = None
                continue
            prefix, _, ent = tag.partition("-")
            cat = LABEL_MAP.get(ent)
            if cat is None:
                if cur_cat:
                    spans.append(Span(start=cur_start, end=cur_end, label=cur_cat))
                cur_cat = cur_start = cur_end = None
                continue
            if prefix == "B" or cur_cat != cat:
                if cur_cat:
                    spans.append(Span(start=cur_start, end=cur_end, label=cur_cat))
                cur_cat = cat
                cur_start = tok_start
                cur_end = tok_end
            else:  # I-, same cat
                cur_end = tok_end
        if cur_cat:
            spans.append(Span(start=cur_start, end=cur_end, label=cur_cat))
        text = text.rstrip()  # remove the trailing extra space cursor adds
        if not spans:
            continue
        ex = Example(text=text, spans=spans, language="de_ch", source="real")
        try:
            ex.validate_offsets()
        except ValueError:
            continue
        out.append(ex)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    examples = load(split=args.split)
    n = write_jsonl(args.out, examples)
    print(f"wrote {n} SwissNER examples → {args.out}")


if __name__ == "__main__":
    main()
