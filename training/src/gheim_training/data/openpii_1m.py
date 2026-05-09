"""Loader for ``ai4privacy/pii-masking-openpii-1m`` → gheim Examples.

This is the gheim-2 successor to ``ai4privacy.py`` (which loaded the older
300k release with a v1-only REMAP). The 1m release is larger (1.43M vs
177k records), uses a richer schema (98 entity types vs ~30), and exposes
a ``region`` field that lets us pull native Swiss records (de/CH, fr/CH,
it/CH) as well as the broader EU/UK/US English coverage we need for the
"first-class English" goal.

Language mapping
----------------
The model's label space (see ``schema.Language``) only knows
``de_ch``/``fr_ch``/``it_ch``/``rm``/``gsw``/``en`` — it does not separate
DE-from-Germany from DE-from-Switzerland. We collapse all DE/FR/IT regions
to the Swiss labels because:
  - Standard German/French/Italian PII patterns are largely transferable
    across regions — names, dates, emails, etc. behave the same way.
  - Swiss-specific patterns (CH-IBAN, AHV, +41 phone, Swiss addresses)
    come from Layer 1 (Faker_CH synthetic).
  - Mixing Swiss and non-Swiss records gives the model more coverage of
    the general PII shape vocabulary without diluting Swiss specificity.

The original ``region`` is preserved in ``Example.meta["region"]`` so
downstream analysis (per-region eval splits, region-targeted oversampling)
remains possible.

Run:
    uv run python -m gheim_training.data.openpii_1m \\
        --out data/layer2_v2.jsonl \\
        --languages de,fr,it
    uv run python -m gheim_training.data.openpii_1m \\
        --out data/layer4_v2.jsonl \\
        --languages en

Set ``GHEIM_REMAP_STRICT=1`` to fail loudly on any unmapped label.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated

from .ai4privacy_remap import remap
from .schema import Example, Language, Span, write_jsonl

DATASET_NAME = "ai4privacy/pii-masking-openpii-1m"
SOURCE_TAG = "ai4privacy"

# 1m language codes → gheim Language. None means "skip" (we don't have a
# matching label and won't collapse arbitrary languages into our scheme).
_LANG_TO_GHEIM: dict[str, Language] = {
    "de": "de_ch",
    "fr": "fr_ch",
    "it": "it_ch",
    "en": "en",
    # No mapping for: bg, cs, da, el, es, et, fi, hr, hu, lt, lv, nl, pl,
    # pt, ro, sk, sl, sr, sv. Add here if/when we want them.
}


def _spans_from_record(rec: dict) -> list[Span]:
    """Convert a 1m record's privacy_mask into gheim Spans, applying REMAP."""
    text = rec.get("source_text") or ""
    mask = rec.get("privacy_mask") or []
    if not text or not mask:
        return []
    out: list[Span] = []
    for m in mask:
        lab = m.get("label") or m.get("entity_type") or m.get("entity")
        s = m.get("start")
        e = m.get("end")
        if lab is None or s is None or e is None:
            continue
        cat = remap(lab, source=SOURCE_TAG)
        if cat is None:
            continue
        if e <= s or e > len(text):
            continue
        if not text[s:e].strip():
            continue
        out.append(Span(start=int(s), end=int(e), label=cat))
    return out


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """First-by-start wins; later overlaps dropped (matches ai4privacy.py)."""
    spans.sort(key=lambda sp: (sp.start, -sp.end))
    out: list[Span] = []
    last_end = -1
    for sp in spans:
        if sp.start < last_end:
            continue
        out.append(sp)
        last_end = sp.end
    return out


def load(
    *,
    languages: Annotated[Iterable[str], "1m language codes: de/fr/it/en/…"] = (
        "de", "fr", "it", "en",
    ),
    regions: Annotated[
        Iterable[str] | None,
        "Optional region whitelist (CH/DE/AT/IT/FR/...). None = all regions.",
    ] = None,
    splits: Annotated[
        Iterable[str], "1m splits to draw from. ``validation`` is held out for English regression eval — never include it for training builds."
    ] = ("train",),
    max_per_language: Annotated[int | None, "Cap per gheim Language."] = None,
    skip_layer4_train_collisions: Annotated[
        Path | None,
        "If set: hash-exclude any record whose source_text matches a record "
        "in this file (used to keep test_en clean of training data).",
    ] = None,
) -> Iterator[Example]:
    """Stream Examples from openpii-1m, applying the unified REMAP."""
    from datasets import load_dataset

    excl_hashes = _load_text_hashes(skip_layer4_train_collisions) \
        if skip_layer4_train_collisions else set()
    region_filter = set(regions) if regions is not None else None
    lang_filter = set(languages)

    counts: dict[str, int] = {}
    for split in splits:
        ds = load_dataset(DATASET_NAME, split=split, streaming=True)
        for rec in ds:
            lang_code = (rec.get("language") or "").lower()
            if lang_code not in lang_filter:
                continue
            if lang_code not in _LANG_TO_GHEIM:
                continue
            gheim_lang = _LANG_TO_GHEIM[lang_code]
            region = rec.get("region") or ""
            if region_filter is not None and region not in region_filter:
                continue
            if max_per_language and counts.get(gheim_lang, 0) >= max_per_language:
                continue
            text = rec.get("source_text") or ""
            if not text:
                continue
            if excl_hashes and _hash(text) in excl_hashes:
                continue
            spans = _resolve_overlaps(_spans_from_record(rec))
            if not spans:
                continue
            ex = Example(
                text=text,
                spans=spans,
                language=gheim_lang,
                source=SOURCE_TAG,
                meta={
                    "openpii_1m_id": rec.get("uid"),
                    "region": region,
                    "lang_code": lang_code,
                    "split": split,
                },
            )
            try:
                ex.validate_offsets()
            except ValueError:
                continue
            counts[gheim_lang] = counts.get(gheim_lang, 0) + 1
            yield ex


def _hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_text_hashes(path: Path) -> set[str]:
    """Read a JSONL of Examples and hash every text — used to exclude
    training-set records from a held-out eval build."""
    import json
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = rec.get("text") or ""
        if text:
            out.add(_hash(text))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--languages", type=str, default="de,fr,it,en",
                    help="Comma-separated 1m language codes")
    ap.add_argument("--regions", type=str, default=None,
                    help="Optional comma-separated region whitelist (CH,DE,AT,IT,FR,...)")
    ap.add_argument("--splits", type=str, default="train",
                    help="Comma-separated splits (train/validation)")
    ap.add_argument("--max-per-language", type=int, default=None)
    ap.add_argument("--skip-layer4-collisions", type=Path, default=None,
                    help="Path to a JSONL whose texts should be excluded by hash.")
    args = ap.parse_args()

    examples = load(
        languages=args.languages.split(","),
        regions=args.regions.split(",") if args.regions else None,
        splits=tuple(args.splits.split(",")),
        max_per_language=args.max_per_language,
        skip_layer4_train_collisions=args.skip_layer4_collisions,
    )
    n = write_jsonl(args.out, examples)
    print(f"wrote {n} examples → {args.out}")


if __name__ == "__main__":
    main()
