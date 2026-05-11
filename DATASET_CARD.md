---
language:
  - de
  - fr
  - it
  - rm
  - en
multilinguality:
  - multilingual
license: cc-by-4.0
task_categories:
  - token-classification
task_ids:
  - named-entity-recognition
annotations_creators:
  - machine-generated
language_creators:
  - found
  - machine-generated
source_datasets:
  - swiss-ai/apertus-pretrain-swiss
  - swiss-ai/apertus-pretrain-romansh
tags:
  - pii
  - swiss
  - swiss-german
  - ner
  - privacy
  - de-CH
  - fr-CH
  - it-CH
  - rm
  - synthetic-augmented
  - gdpr
size_categories:
  - 100K<n<1M
pretty_name: gheim-ch-pii-171k
configs:
  - config_name: default
    data_files:
      - split: train
        path: gheim_pii_balanced_train.jsonl
      - split: validation
        path: gheim_pii_balanced_validation.jsonl
      - split: test
        path: gheim_pii_balanced_test.jsonl
    default: true
---

<p align="center">
  <img src="https://github.com/joelbarmettlerUZH/gheim/blob/main/assets/logo.png" alt="gheim" width="360">
</p>

# gheim-ch-pii-171k

> **Summary.** 171,336-chunk multilingual PII NER dataset covering the
> four official Swiss languages and English. Approximately 85% is real
> text from the Apertus pretrain corpus (Swiss court rulings, federal
> parliament records, Swiss-filtered web text, Romansh corpus); the
> remaining 15% is LLM-generated synthetic prose used to populate
> categories where real-text coverage was insufficient for training
> or evaluation. Annotations are machine-generated. Eight PII categories
> are encoded in a 33-class BIOES tag scheme. Train, validation, and
> test splits are document-isolated for the real-chunk portion.
> Released under CC BY 4.0.

**171,336 chunks · 8 PII categories · 5 languages · 5 source types · doc-level isolated splits.**

> **Full report:** the curation pipeline (LLM labelling + checksum-validated
> regex augmentation + slot-fill verifier), per-cell distribution analysis,
> related-dataset comparison, and the labelling-policy / label-noise discussion
> are documented in
> [`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
> (arXiv preprint forthcoming). This card is the deployment-facing summary.

---

## Quick start

The dataset is a single configuration on the HuggingFace Hub with three
splits (`train` / `validation` / `test`). Load it with the standard
`datasets` library and inspect the first row:

```python
from datasets import load_dataset

ds = load_dataset("joelbarmettler/gheim-ch-pii-171k")
ds["train"][0]
# → {"id": "...", "text": "...", "language": "de_ch",
#    "spans": [{"start": 138, "end": 146, "label": "private_person", "value": "..."}],
#    "synthetic": False, ...}

# Filter to real chunks only (excludes LLM-generated gap-fill):
ds["train"].filter(lambda r: not r["synthetic"])
```

To train a token-classification model with HF Trainer, build a 33-class
BIOES label space from the eight category names and the standard B/I/E/S
prefixes (one outside class plus four BIOES positions for each category)
and align spans to tokenizer offsets. Reference implementation:
[`training/src/gheim_training/data/bioes.py`](https://github.com/joelbarmettlerUZH/gheim/blob/main/training/src/gheim_training/data/bioes.py).

---

## Dataset summary

`gheim-ch-pii-171k` combines real Swiss-text content from the Apertus
pretrain corpora with a synthetic gap-fill layer that addresses cells
too sparse in the real corpus (notably `secret`, English-language
entities, and Italian/Romansh `account_number`/`private_email`).
146,436 chunks come from the `entscheidsuche` (court rulings),
`curia_vista` (federal parliament), Swiss-filtered FineWeb-2, and
Romansh subsets of the Apertus pretrain corpora; 24,900 are
LLM-generated (Gemma 4 26B-A4B) multilingual prose with embedded PII
values verified by exact substring match. Every record carries a
`synthetic` boolean field for filtering.

The label space is byte-aligned with the
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)
33-class BIOES head, so a fine-tune started from that base preserves
the pretrained classifier-head weights.

| | |
|---|---|
| Size | 171,336 chunks (146,436 real + 24,900 synthetic) |
| Languages | de_CH, fr_CH, it_CH, rm, en |
| Categories | 8 (account_number, private_address, private_date, private_email, private_person, private_phone, private_url, secret) |
| Tag scheme | BIOES, 33 classes (O + B/I/E/S × 8) |
| Source corpora | apertus-pretrain-swiss, apertus-pretrain-romansh, gheim-synthetic |
| Negative-chunk ratio | ~7.5% |
| License | CC BY 4.0 |
| Splits | `train` (139,641) / `validation` (15,834) / `test` (15,861); document-level isolation for real chunks |

---

## Languages

The four official Swiss languages are represented in proportions that
roughly track Swiss-Hub web text rather than census population, with
English added as a regression anchor. Distribution across the
171,336 chunks:

| Code | Variant | Chunks | % |
|---|---|---:|---:|
| `de_ch` | Swiss German written standard | 48,564 | 28.3 |
| `fr_ch` | Swiss French | 48,097 | 28.1 |
| `it_ch` | Swiss Italian (Ticino/Grisons) | 45,462 | 26.5 |
| `rm` | Romansh (Rumantsch Grischun + idiom variants) | 17,171 | 10.0 |
| `en` | English (international, Swiss-context) | 12,042 | 7.0 |

Spoken-form Swiss German (GSW) is **not** explicitly represented:
the corpus is written-standard German as published in Swiss legal
and journalistic contexts.

---

## Categories

The label space has eight PII categories, each encoded in BIOES (Begin,
Inside, Outside, End, Single) for 33 classes overall. Categories were
chosen to cover the most common PII flagged for anonymisation in
Swiss-market LLM round-trips:

| Category | Examples |
|---|---|
| `account_number` | CH-IBAN, AHV/AVS, VAT-CHE, credit-card numbers |
| `private_address` | Bahnhofstrasse 1, 8001 Zürich |
| `private_date` | 1990-01-02, 14. März 2026 |
| `private_email` | jbarmettler@proton.me |
| `private_person` | Joel Barmettler, Dr. Schmidt |
| `private_phone` | +41 44 268 12 34 |
| `private_url` | joelbarmettler.xyz |
| `secret` | API keys, bearer tokens |

The `private_` prefix is historical and does **not** mean the entity has
been verified to be private; all eight categories are applied with a
recall-oriented labelling policy and may flag publicly-listed
institutional entities (e.g. court switchboard numbers, parliament
email addresses, public-official names). Applications needing stricter
precision should pair the dataset with a downstream public-vs-private
classifier.

---

## Dataset structure

### Data instance

Each row is a JSON object: a chunk of source text plus a list of
character-aligned PII spans. Provenance fields (`subset`,
`source_dataset`, `synthetic`) let you filter to a specific
sub-population. A representative real-text row from the
parliament-proceedings subset:

```jsonc
{
  "id": "/curia_vista_2024/...0192.jsonl.gz/15#0",
  "text": "M. le Conseiller fédéral Berset a indiqué dans sa réponse écrite du 4 mai 2023 que le bureau zurichois (Werdstrasse 36, 8004 Zürich) avait reçu un courriel à finance@parl.admin.ch ...",
  "language": "fr_ch",
  "subset": "curia_vista",
  "source_dataset": "swiss-ai/apertus-pretrain-swiss",
  "spans": [
    {"start": 25, "end": 31, "label": "private_person", "value": "Berset", "source": "gemma"},
    {"start": 73, "end": 84, "label": "private_date", "value": "4 mai 2023", "source": "regex"},
    {"start": 116, "end": 142, "label": "private_address", "value": "Werdstrasse 36, 8004 Zürich", "source": "gemma"},
    {"start": 161, "end": 184, "label": "private_email", "value": "finance@parl.admin.ch", "source": "regex"}
  ],
  "synthetic": false
}
```

### Span structure

Each span is a five-field object describing which slice of the chunk
text carries which PII category and where the label came from:

| Field | Type | Description |
|---|---|---|
| `start` | int | Character offset, inclusive. |
| `end` | int | Character offset, exclusive. `text[start:end] == value`. |
| `label` | string | One of the 8 categories listed above. |
| `value` | string | The literal substring. |
| `source` | string | Provenance: `gemma`, `regex`, `synthetic`, or `ai4privacy`. |

### Splits

The 171,336 chunks are split roughly 88 / 6 / 6 between training,
validation and test, with each split keeping the same ratio of real
to synthetic chunks:

| Split | Chunks | Real | Synthetic | Notes |
|---|---:|---:|---:|---|
| `train` | 139,641 | 117,141 | 22,500 | For fine-tuning |
| `validation` | 15,834 | 14,634 | 1,200 | Early stopping + best-model selection |
| `test` | 15,861 | 14,661 | 1,200 | Final eval, touched once per model |

The real-chunk split is document-level isolated: every real `doc_id`
lives in exactly one split, so chunks from the same source document
never leak across splits. Stratification is on (language × subset).
Synthetic chunks are placed deterministically (seed=17) and have
disjoint chunk IDs across splits.

The bulk of synthetic chunks live in train; the 2,400 in val/test
populate cells where real Swiss coverage was too sparse to evaluate
(`secret` across all five languages and `en × {email/address/account/phone}`).
Filter to `synthetic == false` for a real-only evaluation.

---

## Considerations for using the data

The dataset's labelling policy favours recall over precision: a pattern
that *looks* like PII is flagged even when context indicates it is not
private (court switchboards, public officials' names, federal IBANs,
publicly-listed parliamentary emails). This matches the production
stance of a redaction tool, where false negatives leak data and false
positives only over-mask. Applications requiring stricter precision
should layer a downstream public-vs-private classifier.

Annotations are machine-generated. A 4-way independent re-labelling
on a stratified 176-chunk sample yielded F1 ≈ 0.66 against the dataset's
labels; the gap is dominated by policy-interpretation differences (the
re-labellers tended to under-flag publicly-listed institutional PII)
rather than random error, so this number is best read as an upper bound
on label noise, not as a quality claim. Full discussion is in
[`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
§2.5 and §5.

The `secret` cell of validation and test is populated **entirely** by
synthetic gap-fill chunks; real Swiss text yielded fewer than ten
secret-bearing examples. For a real-only secret evaluation, filter the
source dataset to `synthetic == false`.

---

## Related datasets

For a structured comparison against `ai4privacy/openpii-1m`,
`ai4privacy/pii-masking-300k`, CoNLL-2003 and `ZurichNLP/swissner`
(size, language coverage, source text, PII categories, account-number
coverage, annotation method, held-out integrity, license), see
[`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
§6.1 and §6.3.

---

## Licensing

This dataset is published under [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/).

The source corpora (`swiss-ai/apertus-pretrain-swiss`,
`swiss-ai/apertus-pretrain-romansh`) are open-licensed; the most
restrictive upstream licence is CC BY 4.0, which we therefore inherit.
PII span annotations were produced by automated tooling owned and
operated by the dataset curator (Joel Barmettler) and are released
under the same CC BY 4.0.

When using this dataset, please attribute:

> "gheim-ch-pii-171k" by Joel Barmettler, 2026.
> Built from `swiss-ai/apertus-pretrain-swiss` and `-romansh` plus
> Gemma-generated synthetic gap-fill.

---

## Reproducibility

The complete curation pipeline (sourcing, labelling, balancing,
splitting, synthesis, gap-fill placement) is available at
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim)
under `training/src/gheim_training/data/`. All stages use fixed random
seeds and produce deterministic outputs. The regex catalogue with
checksum validators (ISO 13616 mod-97 for IBAN, EAN-13 mod-10 for AHV,
ISO 7064 mod-11 for VAT-CHE, Luhn for credit cards) lives at
[`packages/gheim-py/src/gheim/detectors/composite.py`](https://github.com/joelbarmettlerUZH/gheim/blob/main/packages/gheim-py/src/gheim/detectors/composite.py).

---

## Versioning

Semantic versioning, dataset edition: **MAJOR** for schema or
label-space changes; **MINOR** for new languages, source corpora, or
≥10% growth; **PATCH** for label corrections, span-offset fixes, or
documentation. Latest version: **1.0**.

---

## Changelog

### v1.0 (2026-05-09, initial release)

- 171,336 chunks across 5 languages (de_ch, fr_ch, it_ch, rm, en).
- 8 PII categories, 33-class BIOES label space.
- Real-Swiss source: Apertus court rulings (`entscheidsuche`),
  parliament (`curia_vista`), Swiss-filtered web (`fineweb`), Romansh
  corpus (`romansh`).
- Synthetic gap-fill: 24,900 Gemma-slot-fill chunks (developer chat,
  customer-record notes, incident reports) targeting `secret` × all
  languages and `en × {email/address/account/phone}`.
- Document-level isolated train/val/test split (real chunks);
  synthetic chunks placed deterministically (seed=17).

---

## Contact

Joel Barmettler · [jbarmettler@proton.me](mailto:jbarmettler@proton.me) · [joelbarmettler.xyz](https://joelbarmettler.xyz) · [github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim) · [GitHub Issues](https://github.com/joelbarmettlerUZH/gheim/issues)

If you use this dataset in published work, please cite:

```bibtex
@misc{barmettler2026gheim_ch_pii,
  title  = {gheim-ch-pii-171k: A Swiss-grounded PII NER dataset with synthetic gap-fill},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k},
  note   = {Built on swiss-ai/apertus-pretrain-swiss + apertus-pretrain-romansh, with Gemma-generated synthetic chunks for sparse cells}
}
```
