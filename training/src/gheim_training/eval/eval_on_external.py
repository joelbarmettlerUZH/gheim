"""Score gheim against external PII / NER benchmarks in one pass.

Replaces the four ``build_*_eval`` siblings (openpii, swissner,
conll2003, wikineural). Each dataset's converter is wrapped in a
registry entry; the script loads the source HF dataset, projects it into
our 8-cat schema in memory, and runs the HF transformers backend
through the same scoring code as :mod:`eval_on_ours`. The resulting
JSON has the same shape so :mod:`report` consumes both uniformly.

Usage::

    eval_on_external.py --dataset openpii    --model joelbarmettler/gheim-ch-560m \\
        [--per-lang 2000] --out eval/external_gheim_openpii.json
    eval_on_external.py --dataset swissner   --model joelbarmettler/gheim-ch-560m \\
        --out eval/external_gheim_swissner.json
    eval_on_external.py --dataset conll2003  --model joelbarmettler/gheim-ch-560m \\
        --out eval/external_gheim_conll2003.json
    eval_on_external.py --dataset wikineural --model joelbarmettler/gheim-ch-560m \\
        [--per-lang 2000] --out eval/external_gheim_wikineural.json

Only the HF transformers backend is supported here — Direction B of the
positioning matrix only ever runs gheim against external data, never
the other way around.
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._scoring import (
    CharSpan,
    ScoreAccumulator,
    build_pred_id_translation,
    gold_spans_from_row,
    merge_collapsed_spans,
    spans_from_label_sequence,
    write_result,
)

# ---------------------------------------------------------------------------
# Schema remaps
# ---------------------------------------------------------------------------

# openpii-1m label → our 8-cat schema. Carried over verbatim from the
# legacy build_openpii_eval.py so re-runs reproduce the published numbers.
OPENPII_REMAP: dict[str, str] = {
    "GIVENNAME":            "private_person",
    "SURNAME":              "private_person",
    "TITLE":                "private_person",
    "USERNAME":             "private_person",
    "EMAIL":                "private_email",
    "TELEPHONENUM":         "private_phone",
    "PHONENUM":             "private_phone",
    "STREET":               "private_address",
    "BUILDINGNUM":          "private_address",
    "CITY":                 "private_address",
    "STATE":                "private_address",
    "ZIPCODE":              "private_address",
    "COUNTY":               "private_address",
    "SECONDARYADDRESS":     "private_address",
    "GEOCOORD":             "private_address",
    "DATE":                 "private_date",
    "DOB":                  "private_date",
    "TIME":                 "private_date",
    "URL":                  "private_url",
    "IPV4":                 "private_url",
    "IPV6":                 "private_url",
    "CREDITCARDNUMBER":     "account_number",
    "CREDITCARDISSUER":     "account_number",
    "CREDITCARDCVV":        "account_number",
    "IBAN":                 "account_number",
    "BIC":                  "account_number",
    "ACCOUNTNUM":           "account_number",
    "TAXNUM":               "account_number",
    "DRIVERLICENSENUM":     "account_number",
    "IDCARDNUM":            "account_number",
    "SOCIALNUM":            "account_number",
    "PASSPORTNUM":          "account_number",
    "BITCOINADDRESS":       "account_number",
    "ETHEREUMADDRESS":      "account_number",
    "LITECOINADDRESS":      "account_number",
    "PIN":                  "account_number",
    "PASSWORD":             "secret",
    "MAC":                  "secret",
    "USERAGENT":            "secret",
    "PHONEIMEI":            "secret",
    "VEHICLEVIN":           "secret",
    "VEHICLEVRM":           "secret",
    "AGE":                  "O",
    "GENDER":               "O",
    "SEX":                  "O",
    "HEIGHT":               "O",
    "EYECOLOR":             "O",
    "JOBAREA":              "O",
    "JOBTITLE":             "O",
    "JOBTYPE":              "O",
    "ORDINALDIRECTION":     "O",
    "CURRENCYCODE":         "O",
    "CURRENCYSYMBOL":       "O",
    "CURRENCYNAME":         "O",
    "CURRENCY":             "O",
    "AMOUNT":               "O",
    "COMPANYNAME":          "O",
    "ACCOUNTNAME":          "O",
    "MIDDLENAME":           "private_person",
    "LASTNAME":             "private_person",
    "FIRSTNAME":            "private_person",
    "PREFIX":               "private_person",
}

# Pure-NER datasets: only PER maps to a gheim category.
PER_REMAP: dict[str, str] = {
    "PER": "private_person",
    "LOC": "O",
    "ORG": "O",
    "MISC": "O",
}

# Standard BIO int → string scheme for WikiNeural (and any dataset that
# follows {0:O,1:B-PER,2:I-PER,3:B-ORG,4:I-ORG,5:B-LOC,6:I-LOC,7:B-MISC,8:I-MISC}).
WIKINEURAL_INT_TO_TAG: dict[int, str] = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC",
}


# ---------------------------------------------------------------------------
# Token-stream → text + char-span helper
# ---------------------------------------------------------------------------

def tokens_tags_to_text_spans(
    tokens: list[str], tags: list[str], remap: dict[str, str],
) -> tuple[str, list[dict]]:
    """Detokenize with single spaces, walk the BIO tags to build char
    spans in the remapped schema. The output ``spans`` list matches
    the row schema used by :mod:`eval_on_ours` (``start``, ``end``,
    ``label``).

    Punctuation is joined as-is with a leading space (small over-
    tokenisation quirk, accepted for simplicity to match the legacy
    build_*_eval.py behaviour).
    """
    text_parts: list[str] = []
    char_offsets: list[int] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            text_parts.append(" ")
            cursor += 1
        char_offsets.append(cursor)
        text_parts.append(tok)
        cursor += len(tok)
    text = "".join(text_parts)

    spans: list[dict] = []
    cur_label: str | None = None
    cur_start: int = -1
    cur_end: int = -1
    # Append a sentinel "O" so the final span (if any) is flushed by the
    # close-condition without needing a duplicate flush after the loop.
    for i, tag in enumerate(tags + ["O"]):
        prefix = tag.split("-", 1)[0]
        bare = tag.split("-", 1)[1] if "-" in tag else "O"
        target = remap.get(bare, "O") if bare != "O" else None

        if cur_label is not None:
            close = (
                target is None
                or target != cur_label
                or prefix == "B"
            )
            if close:
                spans.append({"start": cur_start, "end": cur_end, "label": cur_label})
                cur_label = None
                cur_start = cur_end = -1

        if target and target != "O":
            tok_start = char_offsets[i] if i < len(char_offsets) else cursor
            tok_end = tok_start + len(tokens[i])
            if cur_label is None:
                cur_label = target
                cur_start = tok_start
                cur_end = tok_end
            else:
                cur_end = tok_end

    return text, spans


# ---------------------------------------------------------------------------
# Per-dataset loaders. Each returns a list[dict] in our row schema:
#   {text, spans:[{start,end,label}], language, source, template_id}
# ---------------------------------------------------------------------------

def _load_openpii(per_lang: int, languages: list[str], seed: int) -> tuple[list[dict], str]:
    from datasets import load_dataset

    targets = set(languages)
    print(f"Streaming openpii-1m validation; sampling {per_lang} per lang in {sorted(targets)} ...",
          flush=True)
    ds = load_dataset("ai4privacy/pii-masking-openpii-1m", split="validation", streaming=True)

    bucket: dict[str, list[dict]] = defaultdict(list)
    unmapped: Counter[str] = Counter()
    n_scanned = 0
    for row in ds:
        n_scanned += 1
        lang = row.get("language")
        if lang not in targets:
            continue
        if len(bucket[lang]) >= per_lang:
            if all(len(bucket[la]) >= per_lang for la in targets):
                break
            continue
        text = row.get("source_text") or ""
        if not text:
            continue
        gold_spans: list[dict] = []
        for s in row.get("privacy_mask") or []:
            label = s.get("label")
            target = OPENPII_REMAP.get(label)
            if target is None:
                unmapped[label] += 1
                continue
            if target == "O":
                continue
            start = int(s.get("start", -1))
            end = int(s.get("end", -1))
            if start < 0 or end <= start or end > len(text):
                continue
            gold_spans.append({"start": start, "end": end, "label": target})
        bucket[lang].append({
            "text": text,
            "spans": gold_spans,
            "language": lang,
            "source": "ai4privacy/openpii-1m",
            "template_id": None,
        })
        if n_scanned % 5000 == 0:
            sizes = ", ".join(f"{la}={len(bucket[la])}" for la in sorted(targets))
            print(f"  scanned {n_scanned:,}; bucket sizes: {sizes}", flush=True)
    if unmapped:
        print(f"  unmapped openpii labels (treated as O): {dict(unmapped.most_common(15))}",
              flush=True)

    flat: list[dict] = []
    for la in sorted(targets):
        flat.extend(bucket[la])
    print(f"  total: {len(flat):,} chunks "
          f"({', '.join(f'{la}={len(bucket[la])}' for la in sorted(targets))})",
          flush=True)
    # --seed is accepted for CLI parity with the wikineural loader; the
    # streamed ai4privacy split is itself shuffled deterministically
    # server-side, so the flag is currently a no-op here.
    _ = seed
    return flat, "ai4privacy/pii-masking-openpii-1m"


def _load_swissner(_per_lang: int, _languages: list[str], _seed: int) -> tuple[list[dict], str]:
    from datasets import load_dataset

    print("Loading ZurichNLP/swissner ...", flush=True)
    ds = load_dataset("ZurichNLP/swissner")

    flat: list[dict] = []
    span_counts: Counter[str] = Counter()
    per_lang_n: Counter[str] = Counter()
    for split in ("test_de", "test_fr", "test_it", "test_rm"):
        if split not in ds:
            print(f"  skipping {split} (not present)", flush=True)
            continue
        lang_code = split.split("_", 1)[1]
        for row in ds[split]:
            tokens: list[str] = row["tokens"]
            tags = row["ner_tags"]
            if tags and isinstance(tags[0], int):
                names = ds[split].features["ner_tags"].feature.names
                tags = [names[i] for i in tags]
            text, spans = tokens_tags_to_text_spans(tokens, tags, PER_REMAP)
            for s in spans:
                span_counts[s["label"]] += 1
            flat.append({
                "text": text,
                "spans": spans,
                "language": lang_code,
                "source": "ZurichNLP/swissner",
                "template_id": None,
            })
            per_lang_n[lang_code] += 1
    print(f"  rows: {dict(per_lang_n)}", flush=True)
    print(f"  spans (post-remap, PER → private_person): {dict(span_counts)}", flush=True)
    return flat, "ZurichNLP/swissner"


def _load_conll2003(_per_lang: int, _languages: list[str], _seed: int) -> tuple[list[dict], str]:
    from datasets import load_dataset

    print("Loading tomaarsen/conll2003 test split ...", flush=True)
    ds = load_dataset("tomaarsen/conll2003", split="test")
    names = ds.features["ner_tags"].feature.names
    print(f"  {len(ds)} rows; label set: {names}", flush=True)
    flat: list[dict] = []
    n_spans = 0
    for row in ds:
        tokens = row["tokens"]
        tags = [names[i] for i in row["ner_tags"]]
        text, spans = tokens_tags_to_text_spans(tokens, tags, PER_REMAP)
        n_spans += len(spans)
        flat.append({
            "text": text,
            "spans": spans,
            "language": "en",
            "source": "conll2003",
            "template_id": None,
        })
    print(f"  {n_spans:,} private_person spans extracted", flush=True)
    return flat, "tomaarsen/conll2003"


def _load_wikineural(per_lang: int, _languages: list[str], seed: int) -> tuple[list[dict], str]:
    from datasets import load_dataset

    rng = random.Random(seed)
    flat: list[dict] = []
    n_spans = 0
    per_lang_n: dict[str, int] = {}
    for split in ("test_en", "test_de", "test_fr", "test_it"):
        lang = split.split("_", 1)[1]
        print(f"Loading {split} ...", flush=True)
        ds = load_dataset("Babelscape/wikineural", split=split)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[: per_lang]
        print(f"  {len(ds):,} rows; sampling {len(idxs):,}", flush=True)
        kept = 0
        for i in idxs:
            row = ds[i]
            tokens = row["tokens"]
            tags = [WIKINEURAL_INT_TO_TAG[int(t)] for t in row["ner_tags"]]
            text, spans = tokens_tags_to_text_spans(tokens, tags, PER_REMAP)
            n_spans += len(spans)
            flat.append({
                "text": text,
                "spans": spans,
                "language": lang,
                "source": "Babelscape/wikineural",
                "template_id": None,
            })
            kept += 1
        per_lang_n[lang] = kept
        print(f"  kept {kept} rows", flush=True)
    print(f"\n  total: {len(flat)} rows; {n_spans} private_person spans", flush=True)
    print(f"  per-lang: {per_lang_n}", flush=True)
    return flat, "Babelscape/wikineural"


# Registry: dataset name → (loader, default per_lang, default languages,
# whether --per-lang is meaningful). Keep this as a module-level table so
# new datasets can be wired in by appending one line.
LoaderResult = tuple[list[dict], str]
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "openpii": {
        "load": _load_openpii,
        "default_per_lang": 2000,
        "default_languages": ["en", "de", "fr", "it"],
        "supports_per_lang": True,
    },
    "swissner": {
        "load": _load_swissner,
        "default_per_lang": 0,
        "default_languages": [],
        "supports_per_lang": False,
    },
    "conll2003": {
        "load": _load_conll2003,
        "default_per_lang": 0,
        "default_languages": [],
        "supports_per_lang": False,
    },
    "wikineural": {
        "load": _load_wikineural,
        "default_per_lang": 2000,
        "default_languages": ["en", "de", "fr", "it"],
        "supports_per_lang": True,
    },
}


# ---------------------------------------------------------------------------
# In-memory dataset wrapper + HF scoring loop
# ---------------------------------------------------------------------------

def _score_rows_with_hf(
    rows: list[dict],
    *,
    model_id: str,
    max_seq_length: int,
    batch_size: int,
    trust_remote_code: bool,
    remap_json: Path | None,
) -> ScoreAccumulator:
    """Run the HF model over an in-memory list of rows.

    Mirrors the loop in :mod:`eval_on_ours` but without round-tripping
    through ``Dataset.from_list`` / ``save_to_disk`` — Direction B
    doesn't need persistence, the data lives only as long as the run.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from ..data.bioes import IGNORE_INDEX, encode_example
    from ..data.label_space import ID2LABEL, LABEL2ID
    from ..data.schema import Example, Span

    print(f"Loading HF model: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, trust_remote_code=trust_remote_code,
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    cfg_id2label = {int(k): v for k, v in model.config.id2label.items()}
    pred_translate = build_pred_id_translation(
        cfg_id2label, LABEL2ID, ID2LABEL, remap_json,
    )
    apply_merge = remap_json is not None

    acc = ScoreAccumulator(strict_method="seqeval-token-level")
    n = len(rows)
    t0 = time.time()
    for batch_start in range(0, n, batch_size):
        batch = rows[batch_start: batch_start + batch_size]
        texts = [r["text"] for r in batch]
        enc = tokenizer(
            texts,
            return_offsets_mapping=True,
            padding=True, truncation=True,
            max_length=max_seq_length,
            return_tensors="np",
        )
        offsets_batch = enc.pop("offset_mapping")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        with torch.no_grad():
            ids_t = torch.from_numpy(np.asarray(input_ids))
            am_t = torch.from_numpy(np.asarray(attention_mask))
            if torch.cuda.is_available():
                ids_t = ids_t.cuda()
                am_t = am_t.cuda()
            logits = model(input_ids=ids_t, attention_mask=am_t).logits.float().cpu().numpy()
        preds = np.argmax(logits, axis=-1)

        for bi, row in enumerate(batch):
            offs = offsets_batch[bi].tolist()
            mask = attention_mask[bi].tolist()
            ex = Example(
                text=row["text"],
                spans=[Span(**s) for s in row["spans"]],
                language=row.get("language", "?"),
                source=row.get("source", "?"),
                template_id=row.get("template_id") or None,
            )
            enc_gold = encode_example(ex, tokenizer, max_length=max_seq_length)
            gold_label_ids = enc_gold["labels"]
            true_seq: list[str] = []
            pred_seq: list[str] = []
            char_offs: list[tuple[int, int]] = []
            for tok_i in range(len(offs)):
                if mask[tok_i] == 0:
                    continue
                a, b = offs[tok_i]
                if a == 0 and b == 0:
                    continue
                g_id = gold_label_ids[tok_i] if tok_i < len(gold_label_ids) else IGNORE_INDEX
                g_lab = "O" if g_id == IGNORE_INDEX else ID2LABEL[int(g_id)]
                p_lab = ID2LABEL[pred_translate.get(int(preds[bi, tok_i]), 0)]
                true_seq.append(g_lab)
                pred_seq.append(p_lab)
                char_offs.append((a, b))
            if apply_merge:
                pred_seq = merge_collapsed_spans(pred_seq)
            pred_spans = spans_from_label_sequence(pred_seq, char_offs)
            gold_spans: list[CharSpan] = gold_spans_from_row(row)
            acc.add_token_row(true_seq, pred_seq, gold_spans, pred_spans, ex.language)

        done = batch_start + len(batch)
        if done % (batch_size * 25) == 0 or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            print(f"  {done:>6}/{n} · {rate:5.1f} chunks/s · elapsed {elapsed:5.0f}s",
                  flush=True)
    return acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASET_REGISTRY), required=True)
    p.add_argument("--model", type=str, required=True,
                   help="HF model id or local checkpoint dir.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-lang", type=int, default=None,
                   help="Per-language sample size (only meaningful for "
                        "datasets with the supports_per_lang flag).")
    p.add_argument("--languages", nargs="+", default=None,
                   help="Override the dataset's default language list.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--remap-json", type=Path, default=None,
                   help="Optional remap JSON (rare; the per-dataset converters "
                        "already project labels into our schema).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    spec = DATASET_REGISTRY[args.dataset]
    loader: Callable[[int, list[str], int], LoaderResult] = spec["load"]

    per_lang = args.per_lang if args.per_lang is not None else spec["default_per_lang"]
    languages = list(args.languages) if args.languages else list(spec["default_languages"])
    rows, dataset_id = loader(per_lang, languages, args.seed)
    if not rows:
        raise SystemExit(f"loader for {args.dataset} returned 0 rows")

    acc = _score_rows_with_hf(
        rows,
        model_id=args.model,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        remap_json=args.remap_json,
    )
    metrics = acc.finalize()
    write_result(
        args.out,
        model_id=args.model,
        backend="hf",
        dataset_id=dataset_id,
        dataset_split="test",
        n_chunks=len(rows),
        metrics=metrics,
        extra={"per_lang": per_lang} if spec["supports_per_lang"] else None,
    )

    ov = metrics["overall"]
    print()
    print(f"OVERALL  strict_span F1={ov['strict_span']['f1']:.4f}  "
          f"char F1={ov['char']['f1']:.4f}", flush=True)
    pp = metrics["per_category"].get("private_person")
    if pp:
        print(f"PER      strict_span F1={pp['strict_span']['f1']:.4f}  "
              f"char F1={pp['char']['f1']:.4f}", flush=True)
    print()
    print("PER-LANGUAGE:")
    for lang, cell in sorted(metrics["per_language"].items()):
        print(f"  {lang:<8} strict={cell['strict_span']['f1']:.4f}  "
              f"char={cell['char']['f1']:.4f}  n={cell['n_chunks']}")
    print()
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
