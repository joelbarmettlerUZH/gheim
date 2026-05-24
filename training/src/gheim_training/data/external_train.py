"""Build an external-NER training-mix layer.

This module materialises a JSONL of training chunks drawn from public
external NER / PII datasets (ai4privacy/openpii-1m, Babelscape WikiNeural,
CoNLL-2003) in the gheim chunk schema, for use as an additional layer
in the train split of a multi-source training experiment.

Schema mapping is shared with the eval-time external scorer
(``gheim_training.eval.eval_on_external``) so a chunk that the eval
pipeline would score one way is labelled the same way for training.

ZurichNLP/swissner has no train split (test-only benchmark), so it
is deliberately NOT included here — adding swissner training material
would invalidate the swissner cross-domain eval.

Run::

    uv run python -m gheim_training.data.external_train

Outputs ``data/layer_external_train.jsonl``. Append it to the train
split via :mod:`gheim_training.data.build_multisource_train` (separate
module that concatenates and rebuilds the HF DatasetDict).
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

# Reuse the exact same schema-remap tables and token-stream converter
# that the eval scorer uses — guarantees train-time and eval-time
# label decisions are consistent.
from ..eval.eval_on_external import (
    OPENPII_REMAP,
    PER_REMAP,
    WIKINEURAL_INT_TO_TAG,
    tokens_tags_to_text_spans,
)

OUT_PATH = Path("data/layer_external_train.jsonl")

# Per-language target sample sizes (per train split).
OPENPII_PER_LANG = 10_000        # × 4 langs = 40k
WIKINEURAL_PER_LANG = 10_000     # × 4 langs = 40k  (research variant only)
WIKIANN_PER_LANG = 20_000        # × 4 langs ≤ 80k  (commercial variant)
CONLL_TOTAL = 14_000             # ~full train      (research variant only)

SEED = 20260524


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _emit_chunk(out_fp, text: str, spans: list[dict], language: str,
                source: str) -> None:
    rec = {
        "id": _new_id(source.split("/")[-1]),
        "text": text,
        "language": language,
        "source": source,
        "spans": spans,
        "synthetic": True,  # treat external as augmentation, not gold gheim-domain
        "template_id": None,
    }
    out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _ai4privacy_lang_to_gheim(lang: str) -> str:
    """openpii uses ISO-639-1 codes. Map to our (de_ch/fr_ch/it_ch/en) labels.
    de→de_ch, fr→fr_ch, it→it_ch keeps the lang-strata aligned with our
    internal balancer; en stays en."""
    return {"de": "de_ch", "fr": "fr_ch", "it": "it_ch", "en": "en"}.get(lang, lang)


def _wikineural_lang_to_gheim(lang: str) -> str:
    return {"de": "de_ch", "fr": "fr_ch", "it": "it_ch", "en": "en"}.get(lang, lang)


def _load_openpii_train(out_fp, per_lang: int, rng: random.Random) -> int:
    from datasets import load_dataset
    targets = {"de", "fr", "it", "en"}
    print(f"\n=== openpii-1m train (sampling {per_lang}/lang in {sorted(targets)}) ===", flush=True)
    ds = load_dataset("ai4privacy/pii-masking-openpii-1m",
                      split="train", streaming=True)
    bucket = {la: 0 for la in targets}
    n_scanned = n_written = 0
    for row in ds:
        n_scanned += 1
        lang = row.get("language")
        if lang not in targets or bucket[lang] >= per_lang:
            if all(bucket[la] >= per_lang for la in targets):
                break
            continue
        text = row.get("source_text") or ""
        if not text:
            continue
        spans = []
        for s in row.get("privacy_mask") or []:
            target = OPENPII_REMAP.get(s.get("label"))
            if not target or target == "O":
                continue
            start, end = int(s.get("start", -1)), int(s.get("end", -1))
            if start < 0 or end <= start or end > len(text):
                continue
            spans.append({"start": start, "end": end, "label": target})
        if not spans:
            continue  # positive-only: skip chunks with no in-schema PII
        _emit_chunk(out_fp, text, spans, _ai4privacy_lang_to_gheim(lang),
                    "ai4privacy/openpii-1m")
        bucket[lang] += 1
        n_written += 1
        if n_scanned % 5000 == 0:
            sizes = ", ".join(f"{la}={bucket[la]}" for la in sorted(targets))
            print(f"  scanned {n_scanned:,}; kept {n_written:,}; bucket: {sizes}", flush=True)
    print(f"  done: scanned {n_scanned:,}, wrote {n_written:,}", flush=True)
    return n_written


def _load_wikineural_train(out_fp, per_lang: int, rng: random.Random) -> int:
    from datasets import load_dataset
    print(f"\n=== WikiNeural train (sampling up to {per_lang}/lang) ===", flush=True)
    n_written = 0
    for split, lang in (("train_de", "de"), ("train_fr", "fr"),
                        ("train_it", "it"), ("train_en", "en")):
        print(f"  loading {split} ...", flush=True)
        ds = load_dataset("Babelscape/wikineural", split=split)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[: per_lang * 3]  # over-sample; we'll filter for PER hits
        kept = 0
        for i in idxs:
            if kept >= per_lang:
                break
            row = ds[i]
            tokens = row["tokens"]
            tags = [WIKINEURAL_INT_TO_TAG[int(t)] for t in row["ner_tags"]]
            text, spans = tokens_tags_to_text_spans(tokens, tags, PER_REMAP)
            if not spans:
                continue
            _emit_chunk(out_fp, text, spans, _wikineural_lang_to_gheim(lang),
                        "Babelscape/wikineural")
            kept += 1
        print(f"  kept {kept:,} {lang} chunks (PER-bearing)", flush=True)
        n_written += kept
    return n_written


def _wikiann_lang_to_gheim(lang: str) -> str:
    return {"de": "de_ch", "fr": "fr_ch", "it": "it_ch",
            "en": "en", "rm": "rm"}.get(lang, lang)


def _load_wikiann_train(out_fp, per_lang: int, rng: random.Random) -> int:
    """WikiAnn (also distributed as PAN-X) — Wikipedia-derived multilingual
    NER (CC BY 4.0 / ODC-BY wrapper over CC BY-SA Wikipedia content).
    Commercial-variant substitute for Babelscape WikiNeural (which is
    CC BY-NC-SA 4.0 and therefore non-commercial only)."""
    from datasets import load_dataset
    NAMES = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")
    print(f"\n=== WikiAnn train (sampling up to {per_lang}/lang) ===", flush=True)
    n_written = 0
    for lang in ("de", "fr", "it", "en", "rm"):
        print(f"  loading {lang} ...", flush=True)
        ds = load_dataset("unimelb-nlp/wikiann", lang, split="train")
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[: min(per_lang * 3, len(idxs))]  # over-sample; filter to PER hits
        kept = 0
        for i in idxs:
            if kept >= per_lang:
                break
            row = ds[i]
            tokens = row["tokens"]
            tag_ints = row["ner_tags"]
            tags = [NAMES[int(t)] if 0 <= int(t) < len(NAMES) else "O" for t in tag_ints]
            text, spans = tokens_tags_to_text_spans(tokens, tags, PER_REMAP)
            if not spans:
                continue
            _emit_chunk(out_fp, text, spans, _wikiann_lang_to_gheim(lang),
                        "unimelb-nlp/wikiann")
            kept += 1
        print(f"  kept {kept:,} {lang} chunks (PER-bearing)", flush=True)
        n_written += kept
    return n_written


def _load_conll2003_train(out_fp, target_n: int) -> int:
    from datasets import load_dataset
    print(f"\n=== CoNLL-2003 train (up to {target_n}, en, PER-bearing) ===", flush=True)
    ds = load_dataset("tomaarsen/conll2003", split="train")
    names = ds.features["ner_tags"].feature.names
    print(f"  {len(ds):,} rows; label set: {names}", flush=True)
    kept = 0
    for row in ds:
        if kept >= target_n:
            break
        tokens = row["tokens"]
        tags = [names[i] for i in row["ner_tags"]]
        text, spans = tokens_tags_to_text_spans(tokens, tags, PER_REMAP)
        if not spans:
            continue
        _emit_chunk(out_fp, text, spans, "en", "conll2003")
        kept += 1
    print(f"  kept {kept:,} PER-bearing rows", flush=True)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--variant", choices=["research", "commercial"],
                    default="research",
                    help=("'research' uses WikiNeural+CoNLL-2003 "
                          "(non-commercial); 'commercial' substitutes "
                          "WikiAnn (CC BY 4.0 / ODC-BY) and drops "
                          "CoNLL-2003 (Reuters research-only)."))
    ap.add_argument("--openpii-per-lang", type=int, default=OPENPII_PER_LANG)
    ap.add_argument("--wikineural-per-lang", type=int, default=WIKINEURAL_PER_LANG)
    ap.add_argument("--wikiann-per-lang", type=int, default=WIKIANN_PER_LANG)
    ap.add_argument("--conll-total", type=int, default=CONLL_TOTAL)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w") as out_fp:
        total += _load_openpii_train(out_fp, args.openpii_per_lang, rng)
        if args.variant == "research":
            total += _load_wikineural_train(out_fp, args.wikineural_per_lang, rng)
            total += _load_conll2003_train(out_fp, args.conll_total)
        else:  # commercial
            total += _load_wikiann_train(out_fp, args.wikiann_per_lang, rng)
    print(f"\nWrote {total:,} chunks → {args.out} (variant={args.variant})",
          flush=True)


if __name__ == "__main__":
    main()
