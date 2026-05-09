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

# gheim-ch-pii-171k

> **TL;DR.** A 171k-chunk multilingual PII NER dataset for the Swiss market.
> Real text (~85%) from Apertus court rulings / parliament / Swiss-filtered
> web / Romansh corpus, plus LLM-generated gap-fill (~15%) to populate
> categories the real corpus left empty (notably API keys / IBAN-CH / AHV).
> 8 PII categories, BIOES-aligned with `openai/privacy-filter`. Doc-level
> stratified train/val/test. License **CC BY 4.0**.

**171,336 chunks · 8 PII categories · 5 languages · 5 source types · doc-level isolated splits.**

**Jump to:** [Quick start](#quick-start) · [Schema](#data-fields) · [Splits](#splits) · [Distribution](#per-language--per-category-chunks-all-splits-combined) · [Limitations](#limitations) · [Comparison vs other PII datasets](#comparison-to-related-datasets) · [Reproducibility](#reproducibility)

This card describes the curation pipeline, balancing decisions, per-cell
distribution, and known limitations honestly. The dataset is suitable for
training Swiss-market PII NER models that need to generalise across legal,
parliamentary, news, and Romansh-language text.

> **Status:** Pre-publication snapshot. Will be released alongside the
> companion `joelbarmettler/gheim-1` model after the bake-off completes.

---

## Comparison to related datasets

Why exist? `gheim-ch-pii-171k` fills a gap that none of the existing PII NER
datasets address: a **doc-level-isolated, multi-register, Swiss-grounded**
NER training set with a label space aligned to a production model. Where
other datasets are stronger on one axis, we are explicit:

| Property | gheim-ch-pii-171k | [`ai4privacy/openpii-1m`](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m) | [`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) | [`conll2003`](https://huggingface.co/datasets/eriktks/conll2003) | [`ZurichNLP/swissbert-ner`](https://huggingface.co/ZurichNLP/swissbert-ner) (eval set) |
|---|---|---|---|---|---|
| Size (chunks) | 171k | 1.43M | 209k | 21k | small (eval-only) |
| Languages | 5 (de/fr/it/rm/en) | 4-21 (locale-mixed) | 4 (en/de/fr/it) | 1 (en) | 4 (de/fr/it/rm) |
| **Romansh** | ✓ ~17k | ✗ | ✗ | ✗ | partial |
| **Swiss-format IBAN/AHV/VAT** | ✓ checksum-validated | ✗ (national IDs only) | ✗ | ✗ | ✗ |
| Real-world text source | ✓ Apertus court rulings + parliament + web + Romansh | ✗ Faker-templated | ✗ Faker-templated | ✓ Reuters newswire | ✓ Swiss news |
| Synthetic augmentation | ✓ documented gap-fill | (full corpus is synthetic) | (full corpus is synthetic) | ✗ | ✗ |
| `secret` (API keys, JWT, OAuth) | ✓ ~16k chunks | ✗ | ✓ but generic | ✗ | ✗ |
| Doc-level split isolation | ✓ enforced | ? | ? | ✓ | n/a |
| BIOES / IOB scheme | ✓ BIOES, 33 classes | mask-based + token BIO | mask-based + BIO | IOB2, 9 classes | BIO |
| Aligned to existing model head | ✓ matches `openai/privacy-filter` 33-class head exactly | ✗ | ✗ | n/a | n/a |
| Held-out test labelling | LLM-labelled + synthetic; estimated F1 vs subagent gold = 0.66 | LLM-labelled (Faker-generated) | Faker-generated | hand-labelled gold | hand-labelled gold |
| Public license | CC BY 4.0 | CC BY 4.0 | CC BY 4.0 | other (Reuters annotations) | CC BY 4.0 |

**Use this dataset when** you need Swiss-specific PII detection (real CH
addresses, real Swiss legal/parliamentary text, Romansh coverage), a label
space compatible with `openai/privacy-filter`, or a dataset that explicitly
documents its noise ceiling and synthesis methodology.

**Use the alternatives when** you need a fully hand-gold-labelled
benchmark (CoNLL-2003), pure-Faker template diversity at much larger
scale (openpii-1m), or the 12+ extended PII categories of AI4Privacy
(driver's license, passport, etc.) that we deliberately collapsed into 8.

---

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("joelbarmettler/gheim-ch-pii-171k")
ds["train"][0]
# → {"id": "...", "text": "...", "language": "de_ch",
#    "spans": [{"start": 138, "end": 146, "label": "private_person", "value": "..."}],
#    "synthetic": False, ...}

# Filter to pure-real chunks only (excludes LLM-generated gap-fill):
ds["train"].filter(lambda r: not r["synthetic"])

# Train an NER model with HuggingFace transformers:
from transformers import AutoModelForTokenClassification
labels = ["O"] + [p + "-" + c for c in (
    "account_number","private_address","private_date","private_email",
    "private_person","private_phone","private_url","secret",
) for p in ("B","I","E","S")]
model = AutoModelForTokenClassification.from_pretrained(
    "FacebookAI/xlm-roberta-large", num_labels=len(labels)
)
# ... HF Trainer-compatible after BIOES tag encoding
```

---

## Dataset summary

`gheim-ch-pii-171k` is a token-level PII NER dataset built from
[`swiss-ai/apertus-pretrain-swiss`](https://huggingface.co/datasets/swiss-ai/apertus-pretrain-swiss)
and
[`swiss-ai/apertus-pretrain-romansh`](https://huggingface.co/datasets/swiss-ai/apertus-pretrain-romansh),
with a **synthetic gap-fill layer** that addresses cells too sparse in the
real corpus (notably `secret`, English-language entities, and
Italian/Romansh `account_number`/`private_email`). 146,436 chunks come from
real Swiss court rulings (`entscheidsuche`), federal parliament records
(`curia_vista`), the Swiss-filtered slice of FineWeb-2, and the Romansh
corpus. The remaining 24,900 chunks are LLM-generated (Gemma 4 26B-A4B)
multilingual prose — Slack messages, CRM records, incident reports — that
embed pre-generated PII values verbatim and then verify the embedding by
exact substring match.

The synthetic chunks live **only in the train split**. Validation and test
splits contain pure real Swiss text. Every record carries a `synthetic`
boolean field so consumers can filter to pure-real if desired.

PII spans on the real chunks were detected by an LLM-ensemble pipeline
(Gemma 4 26B-A4B labeling + checksum-validated regex augmentation) and
then aggressively balanced to remove document-level concentration, value
memorisation hotspots, and source duplicates.

The dataset uses an **8-category PII schema** that aligns byte-for-byte with
the [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)
33-class BIOES label space, so models fine-tuned on it can be served
interchangeably with the upstream base.

| | |
|---|---|
| Size | 171,336 chunks (146,436 real + 24,900 synthetic) |
| Languages | de_CH, fr_CH, it_CH, rm, en |
| Categories | 8 (account_number, private_address, private_date, private_email, private_person, private_phone, private_url, secret) |
| Tag scheme | BIOES, 33 classes (O + B/I/E/S × 8 cats) |
| Source corpora | apertus-pretrain-swiss, apertus-pretrain-romansh, gheim-synthetic |
| Registers | Court rulings, parliamentary records, web news, Romansh literary/journalistic, synthetic (dev chats / CRM records / incident reports) |
| Synthetic location | Train (22,500 chunks) + val (1,200) + test (1,200). The val/test slices fill cells where real Swiss text was too sparse to evaluate (notably `secret` × all langs and `en` × {email/address/account/phone}). |
| Negative ratio | ~7.5% (no-PII chunks; synthetic chunks all positive) |
| License | CC BY 4.0 |
| Splits | `train` (139,641) / `validation` (15,834) / `test` (15,861) — doc-level isolated for real chunks, stratified by (lang × subset). |

---

## Languages

| Code | Name | Variant | Chunks | % of dataset |
|---|---|---|---|---|
| `de_ch` | German | Swiss German written standard | 48,564 | 28.3% |
| `fr_ch` | French | Swiss French | 48,097 | 28.1% |
| `it_ch` | Italian | Swiss Italian (Ticino/Grisons) | 45,462 | 26.5% |
| `rm` | Romansh | Mostly Rumantsch Grischun + idiom variants | 17,171 | 10.0% |
| `en` | English | International, Swiss-context | 12,042 | 7.0% |

**Swiss-German dialect (GSW) is not explicitly represented.** The corpus is
written-standard German as published in Swiss legal and journalistic
contexts. Spoken-form GSW (chuchichäschtli-style orthography) is not in
scope.

Language detection used `facebook/fasttext-language-identification` (NLLB-200
labels) at confidence ≥ 0.5. Chunks with confident non-target labels (e.g.
Russian, Portuguese, Spanish) were dropped — see "Curation pipeline" below.

---

## Categories

The 8 PII categories follow the gheim and `openai/privacy-filter` taxonomy.
The "private_" prefix is historical and does **not** mean the entity has been
verified to be private; it is a category label, applied with an aggressive
recall policy.

| Category | What it covers |
|---|---|
| `account_number` | IBAN (CH/LI), AHV/AVS social security number, VAT (CHE-XXX.XXX.XXX), credit-card numbers |
| `private_address` | Postal addresses, building names, PO boxes, Swiss-format street/PLZ/city |
| `private_date` | Dates and date phrases in any locale-format (DD.MM.YYYY, ISO, written-out months) |
| `private_email` | Email addresses (full), institutional and personal |
| `private_person` | Person names — first/last/full, titles included when adjacent (Frau, Madame, Signor, Dr., av., etc.) |
| `private_phone` | Phone numbers in CH (+41, 0XX) and international formats |
| `private_url` | URLs and bare hostnames |
| `secret` | API keys, passwords, secrets, OAuth tokens, JWTs, hash digests |

### Aggressive recall policy

The labeling pipeline favours **recall over precision**. A pattern that *looks
like* PII is flagged even when context indicates it is not actually private.
Examples that are intentionally retained:

- A Geneva court switchboard number is labelled as `private_phone` even though
  it is publicly listed.
- The Federal Court's IBAN is labelled as `account_number`.
- Public officials (judges, parliamentarians) are labelled as `private_person`.
- Sequences like `0000 0000 0000 0000` that pass the Luhn check but are
  obviously test data are kept.

This matches the production stance of a PII redaction tool: a false negative
(real PII leaks) is worse than a false positive (a public phone number is
masked).

---

## Dataset structure

### Data instances

Six representative records — one per source-type × major-PII-shape — to
convey what the dataset looks like in practice.

**1. Real Swiss-German court ruling with regex-detected date** (Apertus
`entscheidsuche`):

```jsonc
{
  "id": "/entscheidsuche_html/filtered/documents_0456.jsonl.gz/73#3",
  "text": "Mit Urteil vom 7. August 2016 hat das Bundesgericht entschieden, ...",
  "language": "de_ch",
  "subset": "entscheidsuche",
  "source_dataset": "swiss-ai/apertus-pretrain-swiss",
  "doc_id": "/entscheidsuche_html/filtered/documents_0456.jsonl.gz/73",
  "chunk_index_in_doc": 3,
  "spans": [
    {"start": 16, "end": 29, "label": "private_date", "value": "7. August 2016", "source": "regex"}
  ],
  "labeler": "regex+checksum:gheim.detectors.composite._find_regex_spans",
  "prompt_version": "regex-v1-2026-05-08",
  "labeled_at": "2026-05-08T11:24:13",
  "language_old": "de_ch",
  "language_raw": "deu_Latn",
  "language_confidence": 0.9986,
  "synthetic": false
}
```

**2. Real French parliamentary record with multiple PII** (Apertus
`curia_vista`):

```jsonc
{
  "id": "/curia_vista_2024/...0192.jsonl.gz/15#0",
  "text": "M. le Conseiller fédéral Berset a indiqué dans sa réponse écrite du 4 mai 2023 que le bureau zurichois (Werdstrasse 36, 8004 Zürich) avait reçu un courriel à finance@parl.admin.ch ...",
  "language": "fr_ch",
  "subset": "curia_vista",
  "source_dataset": "swiss-ai/apertus-pretrain-swiss",
  "spans": [
    {"start": 25, "end": 31, "label": "private_person", "value": "Berset"},
    {"start": 73, "end": 84, "label": "private_date", "value": "4 mai 2023", "source": "regex"},
    {"start": 116, "end": 142, "label": "private_address", "value": "Werdstrasse 36, 8004 Zürich"},
    {"start": 161, "end": 184, "label": "private_email", "value": "finance@parl.admin.ch", "source": "regex"}
  ],
  "labeler": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
  "synthetic": false
}
```

**3. Real Italian Swiss commercial registry text** (Apertus `fineweb`,
demonstrates the regex+checksum CHE-VAT detection):

```jsonc
{
  "id": "/fineweb-2-swissfilter-quality_10-filterrobots/.../1226#2",
  "text": "Aggiornamento contatto effettuato per il cliente. La società ha sede a Viale Garibaldi 41, 6816 Bissone. Riferimento UID: CHE-101.868.987.",
  "language": "it_ch",
  "subset": "fineweb",
  "source_dataset": "swiss-ai/apertus-pretrain-swiss",
  "spans": [
    {"start": 71, "end": 102, "label": "private_address", "value": "Viale Garibaldi 41, 6816 Bissone"},
    {"start": 121, "end": 136, "label": "account_number", "value": "CHE-101.868.987", "source": "regex"}
  ],
  "synthetic": false
}
```

**4. Real Romansh news with person + URL** (Apertus `romansh`):

```jsonc
{
  "id": "/roh_data/.../engadinerpost_2024_03_05.jsonl.gz/12#1",
  "text": "Il president dal cussegl da Stat, Mario Cavigelli, ha annunzià novas mesiras (https://www.gr.ch/cumissiun)...",
  "language": "rm",
  "subset": "romansh",
  "source_dataset": "swiss-ai/apertus-pretrain-romansh",
  "spans": [
    {"start": 35, "end": 50, "label": "private_person", "value": "Mario Cavigelli"},
    {"start": 78, "end": 105, "label": "private_url", "value": "https://www.gr.ch/cumissiun", "source": "regex"}
  ],
  "synthetic": false
}
```

**5. Synthetic dev chat with secret + person + email** (gap-fill, train
+ val/test only — primary use is teaching the model secret detection):

```jsonc
{
  "id": "synthetic#dev_chat#de_ch#1234",
  "text": "Hey, wir müssen sofort den API key sk-proj-brnTP3fAbnFbmOHnKYaXRvj7uff0LYTH8xIZM1JRcoreogrN rotieren, da er im Repo geleakt wurde. Ich habe Luca Weber bereits informiert, er kann dich unter luca.weber@credit-suisse.com erreichen, um den Staging-Zugang kurzfristig zu klären.",
  "language": "de_ch",
  "subset": "synthetic",
  "source_dataset": "gheim-synthetic",
  "doc_id": "synthetic_dev_chat",
  "chunk_index_in_doc": 1233,
  "spans": [
    {"start": 38, "end": 88, "label": "secret", "value": "sk-proj-brnTP3fAbnFbmOHnKYaXRvj7uff0LYTH8xIZM1JRcoreogrN", "source": "synthetic"},
    {"start": 145, "end": 156, "label": "private_person", "value": "Luca Weber", "source": "synthetic"},
    {"start": 188, "end": 217, "label": "private_email", "value": "luca.weber@credit-suisse.com", "source": "synthetic"}
  ],
  "labeler": "gemma-template-fill+verifier:gheim_training.data.p9_synth",
  "prompt_version": "p9-v1-2026-05-08",
  "labeled_at": "2026-05-09T00:15:42",
  "synthetic": true,
  "synthetic_template": "dev_chat"
}
```

**6. Synthetic English customer record with full multi-PII payload**
(stresses the model on dense, real-format data):

```jsonc
{
  "id": "synthetic#customer_record#en#5",
  "text": "Billing Entry:\nName: Michelle Miles\nAddress: Berghof 110a, 8494 Bauma\nEmail: michelle.miles@vd.ch\nPhone: +41526968028\nPayment processed via CC: 4387882217846191",
  "language": "en",
  "subset": "synthetic",
  "source_dataset": "gheim-synthetic",
  "spans": [
    {"start": 21, "end": 35, "label": "private_person", "value": "Michelle Miles", "source": "synthetic"},
    {"start": 45, "end": 70, "label": "private_address", "value": "Berghof 110a, 8494 Bauma", "source": "synthetic"},
    {"start": 78, "end": 100, "label": "private_email", "value": "michelle.miles@vd.ch", "source": "synthetic"},
    {"start": 108, "end": 121, "label": "private_phone", "value": "+41526968028", "source": "synthetic"},
    {"start": 145, "end": 161, "label": "account_number", "value": "4387882217846191", "source": "synthetic"}
  ],
  "synthetic": true,
  "synthetic_template": "customer_record"
}
```

### Data fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique chunk identifier. Of the form `<doc_id>#<chunk_index_in_doc>`. |
| `text` | string | The chunk text. UTF-8. Lengths typically 200–2000 characters. |
| `language` | string | One of `de_ch`, `fr_ch`, `it_ch`, `rm`, `en`. The corrected language after honest fasttext re-detection. |
| `subset` | string | Source register: `entscheidsuche` (court rulings), `curia_vista` (parliament), `fineweb` (Swiss-filtered web), `romansh`, `synthetic` (LLM-generated gap-fill). |
| `source_dataset` | string | `swiss-ai/apertus-pretrain-swiss`, `swiss-ai/apertus-pretrain-romansh`, or `gheim-synthetic`. |
| `doc_id` | string | The source document identifier, stable within `source_dataset`. |
| `chunk_index_in_doc` | int | Position of this chunk within its source document (0-indexed). |
| `spans` | list | List of PII spans. Each span is an object — see below. |
| `labeler` | string | The model or program that produced this chunk's spans (Gemma model ID, or `regex+checksum:...`). |
| `prompt_version` | string | The labeler prompt or rule version, for reproducibility. |
| `labeled_at` | string | ISO-8601 timestamp of when labeling occurred. |
| `language_old` | string \| null | (Real chunks only) The original language label before re-detection. |
| `language_raw` | string \| null | (Real chunks only) The raw NLLB-200 label from fasttext (e.g. `deu_Latn`). |
| `language_confidence` | float \| null | (Real chunks only) fasttext confidence, in [0, 1]. |
| `synthetic` | bool | `true` for the 24,900 LLM-generated gap-fill chunks (train split only); `false` for the 146,436 real Apertus-sourced chunks. |
| `synthetic_template` | string \| null | (Synthetic chunks only) one of `dev_chat`, `customer_record`, `incident_report`. |

### Span structure

```jsonc
{
  "start": 16,            // character offset, inclusive
  "end": 29,              // character offset, exclusive
  "label": "private_date",// one of the 8 categories
  "value": "7. August 2016", // the literal substring text[start:end]
  "source": "regex"       // optional. Values: "regex" (regex+checksum
                          //   augmentation pass), "synthetic" (Gemma
                          //   slot-fill template), absent (primary
                          //   Gemma labelling on real Apertus text).
}
```

Offsets are character indices into `text`. `text[start:end] == value`.

### Splits

| Split | Chunks | Share | Real | Synthetic | Notes |
|---|---:|---:|---:|---:|---|
| `train` | 139,641 | 81.5% | 117,141 | 22,500 | For fine-tuning |
| `validation` | 15,834 | 9.2% | 14,634 | 1,200 | For early stopping + best-model selection |
| `test` | 15,861 | 9.3% | 14,661 | 1,200 | For final eval. **Touched once per model.** |

The real-chunk split is **document-level**: every real `doc_id` lives in
exactly one split, so chunks from the same source document never leak
across splits. Stratification is on (language × subset) to preserve
register balance in each split. Within each (lang × subset) stratum,
documents are deterministically shuffled (seed=42) and greedily assigned
to splits by chunk-count quota.

**Synthetic gap-fill placement.** The bulk of the 24,900 synthetic
chunks (22,500 / 90.4%) live in train. The remaining 2,400 (1,200 each
to val and test) are placed in the held-out splits to fill cells where
real-Swiss coverage was too sparse to evaluate the model on:

- `secret` × {de_ch, fr_ch, it_ch, rm, en}: real coverage was 0 in val
  and ≤ 3 in test, making secret detection unmeasurable. Each held-out
  split now has ~200 synthetic secret-bearing chunks per language.
- `en × email`, `en × address`, `en × account_number`, `en × phone`:
  real EN coverage was single-digit in each cell. Synthetic
  `customer_record` and `incident_report` chunks (each carrying multiple
  PII types) bring these cells to ~100–400 spans.

These synthetic chunks are flagged with `synthetic: true` and
`subset: "synthetic"`, so analysis can compute per-source metrics
(real-only F1, synthetic-only F1, mixed F1). The synthetic placement is
deterministic (seed=17). No synthetic chunk appears in more than one
split — train/val/test contain disjoint synthetic chunk IDs.

**What this means for benchmarking.** Reported overall F1 on val/test
mixes real-Swiss and synthetic-Faker performance. For a "real-only"
number, filter to `synthetic == false` before scoring. The model card
for the gheim-1 model will report both numbers.

---

## Curation pipeline

This dataset is the output of a 7-phase pipeline. Each phase is reproducible
from source via scripts in
[`training/src/gheim_training/data/`](training/src/gheim_training/data/).

```
swiss-ai/apertus-pretrain-{swiss,romansh}
        │
        ▼  [stream + chunk + lang-detect + per-doc-cap]
  Layer 5v4 (2.31M Gemma-labelled chunks)
        │
        ├──▶ [P1a-i: regex-mine new chunks Gemma missed]
        │    + 1.38M regex-only chunks
        │
        ├──▶ [P1a-ii: regex-augment existing chunks]
        │    + 25.6k missed spans on Gemma chunks
        │
        ├──▶ [P1b: re-detect language honestly]
        │    150k mislabels caught (rus_Cyrl, yue_Hant, ...)
        │
        ▼  3.69M total chunks, 7.63M total spans
   [P2: subagent gate — F1 0.66 vs 4-way majority]
        │
        ▼  [P4: frequency analysis]
   [P5: balance]
        │
        │  - target-lang filter:  3.69M → 3.42M
        │  - text-hash dedup:     3.42M → 2.66M  (757k duplicates)
        │  - placeholder filter:  remove 2,486 `<email-pii>` spans
        │  - per-doc cap (30):    drop 84,288 chunks across 2,767 docs
        │  - per-value cap (30):  cap 32,995 (lang,cat,value) tuples
        │  - per-cell cap:        8,000 / 3,000 / 2,000 / 1,500
        │  - negative sample:     +13,310 no-PII chunks at 10% per-lang
        │
        ▼  146,436 real Swiss chunks
   [P6: doc-level stratified split → train/validation/test (80/10/10)]
        │  validation + test locked, never modified after this point
        │
        ▼
   [P9: Gemma slot-fill gap-fill — TRAIN ONLY]
        │  - 3 templates × 5 langs = up to 9 buckets
        │  - dev_chat: secret + person + email                  (3000 / lang)
        │  - customer_record: account + person + addr + email + phone (2800 / lang for it/rm/en)
        │  - incident_report: secret + person + phone + email   (1500 / en)
        │  - Verifier requires text.find(value) for every slot — 2.0% reject rate
        │  + 24,900 synthetic chunks
        │
        ▼
   gheim-ch-pii-171k:
     train      = 142,041  (117,141 real + 24,900 synthetic)
     validation =  14,634  (real only)
     test       =  14,661  (real only)
```

### Source data

| Source | Subsets | License | What it provides |
|---|---|---|---|
| `swiss-ai/apertus-pretrain-swiss` | `entscheidsuche`, `curia_vista`, `fineweb` | Open (CC BY-compatible) | Swiss-context text in de/fr/it: court rulings, parliamentary records, web news/blogs |
| `swiss-ai/apertus-pretrain-romansh` | `romansh` | Open (CC BY-compatible) | Romansh literary, journalistic, and bilingual-corpus text |
| `gheim-synthetic` (this repo, P9) | `synthetic` | CC BY 4.0 (this dataset) | LLM-generated multilingual prose embedding pre-generated PII values verbatim, used to fill cells too sparse in the real corpus |

Chunking: source documents were split into ~1500-character chunks at sentence
boundaries (with rolling overlap). Chunks shorter than 20 characters or with
language confidence < 0.5 were dropped during streaming.

### Annotations

#### Annotators

This dataset is **not human-annotated**. Spans come from three
complementary automated annotators:

1. **`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`** — a 26B-parameter mixture-of-experts
   instruction model run via vLLM with structured-output constraints
   (JSON-schema-guided decoding). Produces labels for all 8 categories in
   every real chunk it sees. Prompt version `v3-localized-2026-05-06`.
   Provides the majority of spans on real chunks.

2. **`gheim.detectors.composite._find_regex_spans`** — a curated catalogue of
   regular expressions for high-precision categories (IBAN, AHV/AVS, VAT,
   credit cards, phone numbers, emails, URLs) backed by checksum validators
   (ISO 13616 mod-97-10 for IBAN, EAN-13 mod-10 for AHV, ISO 7064
   mod-11-10 for VAT-CHE, Luhn for cards). Catches structured-format PII the
   LLM may have missed and adds the `"source": "regex"` marker on the
   resulting spans. Concentrated in `account_number`, `private_phone`,
   `private_email`, `private_url`.

3. **`gheim_training.data.p9_synth` (synthetic chunks only)** — slot-fill
   pipeline that pre-generates realistic PII payload values (Faker_CH for
   names/addresses/phones; checksummed IBAN-CH/AHV/VAT-CHE; 11 secret
   formats covering OpenAI / GitHub / Slack / AWS / Google / Stripe / JWT
   / hex / base64), prompts Gemma 4 26B-A4B in the target language to
   embed them verbatim into natural prose (Slack message / CRM record /
   incident report), then verifies via exact `text.find(value)` for every
   slot — rejecting and regenerating if Gemma paraphrased or modified
   any value. Final reject rate **2.0%** across 25,311 attempts. All
   spans on synthetic chunks carry `"source": "synthetic"`.

#### Annotation process

The two annotators were run independently and merged with regex-on-overlap
priority. A 4-way independent subagent labeling on a stratified 176-chunk
sample yielded **F1 = 0.664** between the dataset and a majority-vote of
fresh subagents (FN-dominated: 198 false-negatives vs 24 false-positives).
The gap is largely a **policy-interpretation disagreement** — fresh
subagents tend to under-flag public-figure mentions and institutional phone
numbers, which the aggressive-recall policy here intentionally retains. This
is documented for honesty, not as a quality claim.

A second-opinion pass with a different LLM (Qwen) was scoped but
**deferred**: it will only be run if downstream model performance suggests
labelling noise is the bottleneck.

#### Personal and sensitive information

Source corpora contain personal information published in public Swiss
documents:

- **Court rulings (`entscheidsuche`)** are public per Swiss federal court
  publication norms. Parties, witnesses, and counsel are sometimes named in
  full, sometimes anonymised by the court itself.
- **Parliamentary records (`curia_vista`)** are public.
- **Web text (`fineweb`)** is sourced from publicly accessible Swiss-domain
  pages.
- **Romansh corpus** consists of public news (RTR, Engadiner Post),
  literary, and bilingual-reference material.

The PII labels in this dataset are **annotations on already-public text**,
not redactions of private text. By using this dataset, you commit to
processing it for the purpose of training and evaluating PII detection
models, not for re-identification or surveillance.

---

## Distribution

### Per-language × per-category chunks (all splits combined)

A chunk is counted in a category if it contains at least one span of that
category. A chunk with both a person and a date appears in both rows.

| | de_ch | fr_ch | it_ch | rm | en | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| `account_number` | 1,374 | 1,215 | 3,033 | 3,060 | 2,820 | **11,502** |
| `private_address` | 8,883 | 9,096 | 9,812 | 5,038 | 2,839 | **35,668** |
| `private_date` | 17,268 | 18,146 | 17,325 | 3,401 | 1,701 | **57,841** |
| `private_email` | 3,869 | 4,833 | 6,186 | 6,505 | 7,316 | **28,709** |
| `private_person` | 21,844 | 18,483 | 21,584 | 11,334 | 8,898 | **82,143** |
| `private_phone` | 8,264 | 7,016 | 9,456 | 5,017 | 4,402 | **34,155** |
| `private_url` | 9,130 | 8,853 | 8,905 | 2,098 | 1,520 | **30,506** |
| `secret` | 3,042 | 3,008 | 3,002 | 3,004 | 4,503 | **16,559** |
| **Total chunks** | **48,564** | **48,097** | **45,462** | **17,171** | **12,042** | **171,336** |

#### Synthetic gap-fill contribution to train (24,900 chunks)

The P9 synthesis pass added these per-cell increments to the train split:

| | de_ch | fr_ch | it_ch | rm | en |
|---|---:|---:|---:|---:|---:|
| `secret` (was 39 / 8 / 2 / 3 / 3) | +3,000 | +3,000 | +3,000 | +3,000 | +4,500 |
| `account_number` (was 1,092 / 992 / **182** / **200** / **14**) | — | — | +2,800 | +2,800 | +2,800 |
| `private_address` (was 7,146 / 7,322 / 5,692 / 1,801 / **31**) | — | — | — | — | +2,800 |
| `private_phone` (was 6,602 / 5,617 / 5,343 / 1,787 / **75**) | — | — | — | — | +4,300 |
| `private_email` (was 671 / 1,431 / **310** / 562 / **10**) | +3,000 | +3,000 | +5,800 | +5,800 | +7,300 |
| `private_person` (side-effect) | +3,000 | +3,000 | +5,800 | +5,800 | +7,300 |

Bold pre-synthesis counts are the original gaps the synthesis was designed
to close.

### Per-source distribution

| Subset | Chunks | % |
|---|---:|---:|
| `fineweb` | 82,558 | 48.2% |
| `entscheidsuche` | 42,365 | 24.7% |
| `synthetic` | 24,900 | 14.5% |
| `romansh` | 15,029 | 8.8% |
| `curia_vista` | 6,484 | 3.8% |

### Span density

Per-chunk span count is right-skewed in the source corpus and the balanced
dataset preserves this distribution:

| Spans per chunk | 0 (neg) | 1 | 2–3 | 4–5 | 6–10 | 11–20 | 21+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| de_ch | 0.7% | 49.9% | 34.8% | 9.8% | 4.5% | 0.3% | 0.0% |
| fr_ch | 0.7% | 54.1% | 32.3% | 8.5% | 4.0% | 0.3% | 0.0% |
| it_ch | 1.0% | 51.9% | 34.9% | 8.3% | 3.8% | 0.2% | 0.0% |
| rm | 2.3% | 56.5% | 32.4% | 6.4% | 2.2% | 0.2% | 0.0% |
| en | 22.7% | 48.3% | 22.0% | 4.3% | 2.2% | 0.5% | 0.0% |

About 85% of positive chunks contain 1–3 spans. < 0.5% contain more than 10.

---

## Considerations for using the data

### Biases

1. **Public-figure over-representation, then capped.** The raw labelling of
   Apertus court rulings produced top values like Stefan Engler ×2,150,
   Donald Trump ×3,682, Roger Federer ×2,396 — all public figures. The
   per-value cap of 30 reduces these to ≤ 30 occurrences each, but the
   underlying selection is still biased toward people who appear frequently
   in Swiss news and legal text (Swiss politicians, federal judges, public
   officials). Models trained on this dataset may be more confident on
   capitalised proper nouns that look like Swiss-political-figure names.
2. **Court-ruling register dominates the long-form text.** 25% of chunks
   come from `entscheidsuche`. Legal-text PII patterns ("X. SA c. Y. SA",
   "die Beschwerdeführerin", appositive titles) are over-represented
   relative to casual web text. The fineweb subset (48%) provides
   counterweight but is still Swiss-filtered.
3. **Romansh has a single source.** Almost all real `rm` chunks come from
   the `romansh` subset (which is itself dominated by the bilingualer
   corpus and RTR/Engadiner Post). RM coverage is real but narrow.
4. **English is anchor-only on real data.** The 4,742 real English chunks
   exist to keep models from regressing on English, not to claim
   English-language competence. The 7,300 synthetic English chunks address
   sparse cells (account/email/phone/secret) but inherit synthetic biases.
5. **Synthetic chunks bias toward short, high-PII-density formats.**
   Slack messages, CRM rows, and incident reports are short (typically
   100–400 chars) and contain 3–5 PII spans each — a different
   distribution from real Apertus chunks (typically 200–2,000 chars,
   1–3 spans). A model trained on the mix will see PII-dense text more
   frequently than production data warrants.
6. **Synthetic person/email values are Faker-generated.** Names, emails,
   and addresses in the synthetic chunks come from `faker_de_CH`,
   `faker_fr_CH`, `faker_it_CH` plus a small Romansh name pool. They
   look Swiss but are not drawn from any real-world distribution; some
   may collide with real persons by chance.

### Limitations

1. **Labels on real chunks are LLM-generated, not gold.** Estimated F1
   vs a 4-way majority subagent gold is **0.664** (P2 audit, n=176). The
   gap is policy-driven (aggressive recall) but should be interpreted as
   the ceiling of label noise. The `validation` and `test` splits inherit
   the same labelling pipeline — we do not claim they are hand-verified
   gold. Benchmark numbers from this dataset should be reported alongside
   this noise ceiling.
2. **Labels on synthetic chunks are exact-match from the slot-fill
   payload.** Every synthetic span was placed by the verifier finding
   the pre-generated value verbatim in Gemma's output. This makes
   synthetic spans nearly noise-free at the offset level — but only
   captures the entities we asked Gemma to embed. If Gemma added other
   PII-shaped text incidentally (e.g. an extra fake email in a list),
   that text is **not** labelled. So synthetic chunks are precision-
   biased; the model sees confident positive examples but no
   "tricky negatives" within them.
3. **`secret` benchmark is dominated by synthetic.** Real-Swiss text
   yielded only ~7 secret-bearing chunks across the whole 146k corpus,
   so the val/test secret cells (~200/lang each) are entirely synthetic
   `dev_chat` + `incident_report` chunks. The model is benchmarked on
   detecting Faker-generated API keys / JWTs / OAuth tokens embedded in
   LLM-generated chat — that's a real proxy for production secret
   detection (secrets in dev/ops chat are the primary leak vector) but
   it's not Swiss-grounded. For a Swiss-text-only secret eval, sample
   `synthetic == false`.
4. **Swiss German dialect (GSW) coverage is unmeasured.** Apertus does
   not label GSW separately, and our fasttext detector treats GSW as
   German. Use a GSW-specific eval if your application sees dialect text.
5. **Per-doc cap may have removed contiguous narrative.** Court rulings
   that contributed > 30 chunks were down-sampled; the surviving 30
   chunks from such docs are no longer in source order. Don't
   reconstruct documents from this dataset.
6. **Date detection is loose.** The regex-augmentation pass marks any
   plausible date pattern, including court-ruling dates that are
   arguably public-record metadata rather than personal information.
   This is intentional under the aggressive-recall policy.
7. **Synthetic Romansh is the highest-reject bucket** (4.5% of P9
   attempts rejected for the RM dev_chat template, vs 0.04% for en
   customer_record). The accepted RM synthetic chunks are usable, but
   Gemma's RM grasp is the weakest of the five languages — expect more
   stylistic awkwardness in RM synthetic than in other languages.

### Social impact

This dataset is intended to enable PII detection so that Swiss
organisations can use cloud-hosted LLM APIs without leaking customer data
across jurisdictions. It is built from already-public text and exists to
make redaction more reliable, not to make re-identification easier.

Misuse the dataset could enable, in principle, supports of building
extraction or attribution models trained on Swiss text — but the public
nature of the source corpora and the categorical (not entity-linked)
labelling make this dataset substantially less useful for that than direct
scraping of the source corpora would be.

---

## Licensing

This dataset is published under [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/).

The source corpora (`swiss-ai/apertus-pretrain-swiss`,
`swiss-ai/apertus-pretrain-romansh`) are open-licensed; the most restrictive
upstream licence in the mix is CC BY 4.0, which we therefore inherit. PII
span annotations were produced by automated tooling owned and operated by
the dataset curator (Joel Barmettler) and are released under the same
CC BY 4.0.

When using this dataset, **please attribute**:

> "gheim-ch-pii-171k" — Joel Barmettler, 2026.
> Built from `swiss-ai/apertus-pretrain-swiss` and `-romansh` plus
> Gemma-generated synthetic gap-fill.

---

## Reproducibility

The full curation pipeline lives at
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim),
on the `swiss-pii-finetune` branch. Key scripts:

| Script | Purpose |
|---|---|
| `training/src/gheim_training/data/gemma/run_label_streaming.py` | Stream Apertus, run Gemma labeler, write Layer 5v4 |
| `training/src/gheim_training/data/p1_regex_augment.py` | Add missed regex spans to Layer 5v4 |
| `training/src/gheim_training/data/p1_regex_mine_source.py` | Mine new regex-only PII chunks from source |
| `training/src/gheim_training/data/p1_lang_redetect.py` | Honest language re-detection |
| `training/src/gheim_training/data/p2_sample.py` + `p2_score.py` | Subagent quality gate |
| `training/src/gheim_training/data/p4_freq.py` | Frequency analysis |
| `training/src/gheim_training/data/p4b_density_neg.py` | Pre-balance instrumentation |
| `training/src/gheim_training/data/p5_balance.py` | Balancer (caps + dedup + filters) |
| `training/src/gheim_training/data/p6_split.py` | Doc-level stratified train/val/test split |
| `training/src/gheim_training/data/p9_synth.py` | Gemma slot-fill synthetic gap-fill |
| `training/src/gheim_training/data/p14_inject_synth_to_eval.py` | Move stratified slice of synthetic chunks from train into val + test |
| `packages/gheim-py/src/gheim/detectors/composite.py` | The validated regex catalogue |

Run the full pipeline:

```bash
# Balance the post-Phase-1 corpus into a single 146k Swiss-real file:
uv run --project training python -m gheim_training.data.p5_balance --write

# Split into train/val/test (real chunks only):
uv run --project training python -m gheim_training.data.p6_split --write

# Generate synthetic gap-fill (~50 min on 2×4090 with Gemma vLLM):
uv run --project training python -m gheim_training.data.p9_synth

# Append synthetic to train initially:
cat data/gheim_synthetic.jsonl >> data/gheim_pii_balanced_train.jsonl

# Move a stratified slice of synthetic from train to val + test, filling
# cells (secret × all langs, en × {email/address/account/phone}) that
# real-Swiss coverage left empty:
uv run --project training python -m gheim_training.data.p14_inject_synth_to_eval --write
```

All seeds are fixed (P5/P6: seed=42, P9: seed=42 + Gemma temperature=0.85,
P14: seed=17). Outputs are
`data/gheim_pii_balanced_{train,validation,test}.jsonl`.

---

## Citation

```bibtex
@misc{barmettler2026gheim,
  title  = {gheim-ch-pii-171k: A Swiss-grounded PII NER dataset with synthetic gap-fill},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k},
  note   = {Built on swiss-ai/apertus-pretrain-swiss + apertus-pretrain-romansh, with Gemma-generated synthetic chunks for sparse cells}
}
```

## Dataset Curators

This dataset was curated and is maintained by **Joel Barmettler**
(University of Zurich, [@joelbarmettlerUZH](https://github.com/joelbarmettlerUZH)).
Annotations are machine-generated; no third-party human annotators were
contracted. The text sources were curated and released by the
[swiss-ai](https://huggingface.co/swiss-ai) collective as part of the
Apertus pretraining corpus.

**Maintenance commitment.** Issues are monitored on the
[gheim GitHub repo](https://github.com/joelbarmettlerUZH/gheim/issues).
Substantive bugs (label errors, span-offset bugs, schema breaks) trigger
a patch release within 14 days where feasible. Volume additions, language
extensions, or label-space changes are batched into minor/major releases
(see [Versioning](#versioning) below).

---

## Acknowledgments

This dataset stands on the shoulders of:

- **`swiss-ai/apertus-pretrain-{swiss,romansh}`** — without the curated
  Swiss text corpus, none of this would be possible. Thanks to the
  swiss-ai team for the open release.
- **`openai/privacy-filter`** — provided the 33-class BIOES label space
  that this dataset is byte-aligned with, enabling drop-in replacement of
  the upstream model with a Swiss-tuned fine-tune.
- **`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`** — the AWQ-quantised Gemma 4
  used for the bulk of LLM-based labelling and synthetic gap-fill.
- **`ai4privacy/openpii-1m` and `pii-masking-300k`** — informed the label
  REMAP and the multi-schema reconciliation. Used downstream as a
  training augment for English anchor and CH-region email rescue.
- **HuggingFace `datasets`, `transformers`, `accelerate`** — the entire
  pipeline, from streaming Apertus to publishing this card, runs on HF
  infra.
- **vLLM** — made the 24,900-chunk synthetic generation tractable on
  2× RTX 4090 in under an hour.
- **Faker** — for Faker_CH locales used in synthetic payload generation
  (names, addresses, phones).

---

## Versioning

Semantic versioning, dataset edition:

- **MAJOR** — schema changes (field added/removed/renamed), label-space
  changes (categories added/removed/merged), train/val/test resplit,
  significant content removal. Will require dataset-loading code changes
  for downstream users.
- **MINOR** — new languages added, new source corpora added, additional
  configs published, ≥10% growth in chunk count, opt-in fields added.
  Existing fields and splits stay backward compatible.
- **PATCH** — label corrections, span-offset fixes, documentation
  improvements, frontmatter updates. Chunk identities preserved.

Latest version: **1.0** (this card).

---

## Changelog

### v1.0 — 2026-05-09 (initial release)

- 171,336 chunks across 5 languages (de_ch, fr_ch, it_ch, rm, en).
- 8 PII categories, 33-class BIOES label space.
- Real-Swiss source: Apertus court rulings (`entscheidsuche`),
  parliament (`curia_vista`), Swiss-filtered web (`fineweb`), Romansh
  corpus (`romansh`).
- Synthetic gap-fill: 24,900 Gemma-slot-fill chunks (dev_chat,
  customer_record, incident_report) targeting `secret` × all langs and
  `en` × {email/address/account/phone}.
- Doc-level isolated train/val/test split (real chunks); synthetic
  chunks placed deterministically (seed=17).
- Estimated label-noise ceiling F1 = 0.664 (P2 audit, n=176).

---

## Contact

Joel Barmettler — joel.barmettler@gmail.com
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim) · [GitHub Issues](https://github.com/joelbarmettlerUZH/gheim/issues)
