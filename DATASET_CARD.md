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

**Jump to:** [Quick start](#quick-start) · [Schema](#data-fields) · [Splits](#splits) · [Distribution](#per-language--per-category-chunks-all-splits-combined) · [Limitations](#limitations) · [Related datasets](#related-datasets) · [Reproducibility](#reproducibility)

This card documents the curation pipeline, balancing decisions, per-cell
distribution, and known limitations of the dataset. It is intended to
support both reproduction of the curation process and informed
downstream use.

> **Status:** Pre-publication snapshot. To be released alongside the
> companion `joelbarmettler/gheim-1` model after the bake-off completes.

---

## Related datasets

The PII NER landscape includes several public datasets with overlapping
but distinct goals. The table below describes each dataset's design
without ranking. The right choice depends on the user's task. All
metrics are taken from each dataset's published documentation as of
mid-2026.

| Property | gheim-ch-pii-171k | [`ai4privacy/openpii-1m`](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m) | [`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) | [`conll2003`](https://huggingface.co/datasets/eriktks/conll2003) | [`ZurichNLP/swissbert-ner`](https://huggingface.co/ZurichNLP/swissbert-ner) |
|---|---|---|---|---|---|
| Size | 171,336 chunks | 1.43M chunks | 209k chunks | 21k tokenised sentences | small held-out eval set |
| Languages | de_ch, fr_ch, it_ch, rm, en | English plus 17+ locales (de, fr, it, es, nl, ...) | English, de, fr, it | English | de_ch, fr_ch, it_ch, rm |
| Romansh coverage | approximately 17,000 chunks | not present | not present | not present | partial |
| Source text | Apertus pretrain corpus (Swiss court rulings, federal parliament records, Swiss-filtered web, Romansh corpus) plus LLM-generated synthetic | Faker-templated synthetic | Faker-templated synthetic | Reuters newswire | Swiss news (eval) |
| PII categories | 8, collapsed into BIOES (33 classes) | 12+ extended (incl. USERNAME, JOBAREA, PHONEIMEI, EYECOLOR, NEARBYGPSCOORDINATE) | Same 12+ extended categories | 4 (PER, ORG, LOC, MISC) | 4 (PER, ORG, LOC, MISC) |
| Account-number coverage | CH IBAN, AHV/AVS, VAT-CHE, credit cards (checksum-validated by `gheim.detectors.composite`) | National IDs, passports, driver's licenses, credit cards (random alphanumeric, no checksum) | Same | not in scope | not in scope |
| Annotation method | LLM (Gemma 4 26B-A4B) + checksum-validated regex augmentation + LLM-generated synthetic gap-fill (verified by exact substring match) | LLM-generated during synthesis | LLM-generated during synthesis | Hand-annotated by human linguists | Hand-annotated by human linguists |
| Held-out integrity | Real chunks: doc-level split isolation enforced. Synthetic chunks: deterministic seeded placement, disjoint chunk IDs across splits. | Not documented | Not documented | Yes (CoNLL convention) | n/a (eval-only) |
| Reported label-noise ceiling | F1 ≈ 0.66 (estimated by 4-way independent labelling on a 176-chunk sample) | Not documented | Not documented | Considered gold | Considered gold |
| License | CC BY 4.0 | CC BY 4.0 | CC BY 4.0 | Annotations under Reuters Corpus terms | CC BY 4.0 |

`gheim-ch-pii-171k` is differentiated by its inclusion of Romansh, the
checksum-validated Swiss-format account-number subset (CH-IBAN, AHV,
VAT-CHE), and a label space byte-aligned with the
`openai/privacy-filter` 33-class head. Limitations relative to the
alternatives include a smaller absolute size than `openpii-1m`, an
LLM-only annotation pipeline rather than the hand-gold annotations of
CoNLL-2003 and `swissbert-ner`, and a deliberately collapsed 8-category
schema rather than the 12+ extended categories of the AI4Privacy
family. Users with hand-gold needs, very-large-scale needs, or the
extended AI4Privacy categories may find those datasets a better fit.

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
multilingual prose (Slack messages, CRM records, incident reports) that
embed pre-generated PII values verbatim and then verify the embedding by
exact substring match.

Synthetic chunks are placed primarily in the train split (22,500 of
24,900); the remaining 2,400 are split between validation and test
(1,200 each) to provide measurable coverage in cells where real-Swiss
text was too sparse for evaluation, in particular `secret` across all
five languages and English × {email, address, account_number, phone}.
Every record carries a `synthetic` boolean field so consumers can filter
to real-only or synthetic-only as needed.

PII spans on the real chunks were detected by an LLM-ensemble pipeline
(Gemma 4 26B-A4B labelling + checksum-validated regex augmentation),
then balanced via per-document, per-value, and per-cell caps to limit
document-level concentration and value memorisation. See
[Curation pipeline](#curation-pipeline) for the full procedure.

The dataset uses an **8-category PII schema** in BIOES (33 classes:
O + B/I/E/S × 8). The label order and naming match
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)'s
`id2label` exactly, so fine-tunes started from that base preserve the
pretrained classifier-head weights without label remapping.

| | |
|---|---|
| Size | 171,336 chunks (146,436 real + 24,900 synthetic) |
| Languages | de_CH, fr_CH, it_CH, rm, en |
| Categories | 8 (account_number, private_address, private_date, private_email, private_person, private_phone, private_url, secret) |
| Tag scheme | BIOES, 33 classes (O + B/I/E/S × 8 cats) |
| Source corpora | apertus-pretrain-swiss, apertus-pretrain-romansh, gheim-synthetic |
| Registers | Court rulings, parliamentary records, web news, Romansh literary/journalistic, synthetic (dev chats / CRM records / incident reports) |
| Synthetic location | Train (22,500 chunks) + val (1,200) + test (1,200). The val/test slices fill cells where real Swiss text was too sparse to evaluate (in particular `secret` across all languages and `en` × {email/address/account/phone}). |
| Negative ratio | ~7.5% (no-PII chunks; synthetic chunks all positive) |
| License | CC BY 4.0 |
| Splits | `train` (139,641) / `validation` (15,834) / `test` (15,861); document-level isolation for real chunks, stratified by (lang × subset). |

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
Russian, Portuguese, Spanish) were dropped. See "Curation pipeline" below.

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
| `private_person` | Person names (first, last, or full); titles included when adjacent (Frau, Madame, Signor, Dr., av., etc.) |
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

Six representative records (one per source-type, covering the major PII
shapes) showing the on-disk schema.

**1. Real Swiss-German court ruling with regex-detected date** (Apertus
`entscheidsuche`):

```jsonc
{
  "id": "/entscheidsuche_html/filtered/documents_0456.jsonl.gz/73#3",
  "text": "Mit Urteil vom 7. August 2016 hat das Bundesgericht entschieden, ...",
  "language": "de_ch",
  "language_confidence": 0.9986,
  "subset": "entscheidsuche",
  "source_dataset": "swiss-ai/apertus-pretrain-swiss",
  "doc_id": "/entscheidsuche_html/filtered/documents_0456.jsonl.gz/73",
  "chunk_index_in_doc": 3,
  "spans": [
    {"start": 16, "end": 29, "label": "private_date", "value": "7. August 2016", "source": "regex"}
  ],
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
    {"start": 25, "end": 31, "label": "private_person", "value": "Berset", "source": "gemma"},
    {"start": 73, "end": 84, "label": "private_date", "value": "4 mai 2023", "source": "regex"},
    {"start": 116, "end": 142, "label": "private_address", "value": "Werdstrasse 36, 8004 Zürich", "source": "gemma"},
    {"start": 161, "end": 184, "label": "private_email", "value": "finance@parl.admin.ch", "source": "regex"}
  ],
  "synthetic": false
}
```

**3. Real Italian Swiss commercial registry text** (Apertus `fineweb`,
demonstrates checksummed CHE-VAT detection by regex):

```jsonc
{
  "id": "/fineweb-2-swissfilter-quality_10-filterrobots/.../1226#2",
  "text": "Aggiornamento contatto effettuato per il cliente. La società ha sede a Viale Garibaldi 41, 6816 Bissone. Riferimento UID: CHE-101.868.987.",
  "language": "it_ch",
  "subset": "fineweb",
  "source_dataset": "swiss-ai/apertus-pretrain-swiss",
  "spans": [
    {"start": 71, "end": 102, "label": "private_address", "value": "Viale Garibaldi 41, 6816 Bissone", "source": "gemma"},
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
    {"start": 35, "end": 50, "label": "private_person", "value": "Mario Cavigelli", "source": "gemma"},
    {"start": 78, "end": 105, "label": "private_url", "value": "https://www.gr.ch/cumissiun", "source": "regex"}
  ],
  "synthetic": false
}
```

**5. Synthetic dev chat with secret + person + email**
(LLM-generated; primary purpose is to teach the model secret detection):

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
  "synthetic": true,
  "synthetic_template": "dev_chat"
}
```

**6. Synthetic English customer record with full multi-PII payload**:

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
| `language` | string | One of `de_ch`, `fr_ch`, `it_ch`, `rm`, `en`. Language label after fasttext re-detection (raw NLLB output, no fallback defaults). |
| `subset` | string | Source register: `entscheidsuche` (court rulings), `curia_vista` (parliament), `fineweb` (Swiss-filtered web), `romansh`, `synthetic` (LLM-generated gap-fill). |
| `source_dataset` | string | `swiss-ai/apertus-pretrain-swiss`, `swiss-ai/apertus-pretrain-romansh`, or `gheim-synthetic`. |
| `doc_id` | string | The source document identifier, stable within `source_dataset`. |
| `chunk_index_in_doc` | int | Position of this chunk within its source document (0-indexed). |
| `spans` | list | List of PII spans. Each span is an object (see "Span structure" below). |
| `labeler` | string | The model or program that produced this chunk's spans (Gemma model ID, or `regex+checksum:...`). |
| `prompt_version` | string | The labeler prompt or rule version, for reproducibility. |
| `labeled_at` | string | ISO-8601 timestamp of when labeling occurred. |
| `language_confidence` | float \| null | (Real chunks only) fasttext language-detection confidence in [0, 1]. Useful for filtering to high-confidence chunks. Absent on synthetic chunks. |
| `synthetic` | bool | `true` for LLM-generated gap-fill chunks; `false` for real Apertus-sourced chunks. |
| `synthetic_template` | string \| null | (Synthetic chunks only) one of `dev_chat`, `customer_record`, `incident_report`. |
| `ai4p_region`, `ai4p_lang_code` | string \| null | (Optional, training-mix only) For chunks pulled from `ai4privacy/openpii-1m` as a training augment, records the originating region (`CH`, `DE`, ...) and ISO-639 language code. |

### Span structure

```jsonc
{
  "start": 16,             // character offset, inclusive
  "end": 29,               // character offset, exclusive
  "label": "private_date", // one of the 8 categories
  "value": "7. August 2016", // the literal substring text[start:end]
  "source": "regex"        // provenance, one of:
                           //   "gemma":      labelled by Gemma 4 26B-A4B
                           //   "regex":      added by the checksum-validated regex catalogue
                           //   "synthetic":  placed by the slot-fill verifier
                           //   "ai4privacy": sourced from the AI4Privacy training augment
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
split: train, validation, and test contain disjoint synthetic chunk IDs.

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
       Apertus pretrain corpora
              │
              ▼
       chunk → language-detect → annotate
              │     (Gemma 4 26B-A4B + checksum-validated regex)
              │
              ▼
       balance (per-doc, per-value, per-cell caps; dedup; negatives)
              │
              ▼
       doc-level stratified split → train / validation / test
              │
              ▼
       synthetic gap-fill (Gemma slot-fill)
              │
              ▼
       gheim-ch-pii-171k
```

### Source data

| Source | Subsets | License |
|---|---|---|
| `swiss-ai/apertus-pretrain-swiss` | `entscheidsuche`, `curia_vista`, `fineweb` | Open (CC BY-compatible) |
| `swiss-ai/apertus-pretrain-romansh` | `romansh` | Open (CC BY-compatible) |
| `gheim-synthetic` | `synthetic` | CC BY 4.0 (this release) |

Source documents are chunked at sentence boundaries (~1,500 characters
each) and language-tagged with `facebook/fasttext-language-identification`
at confidence ≥ 0.5; chunks below threshold are dropped.

### Annotations

This dataset is not human-annotated. Spans are produced by three
automated annotators:

1. **Gemma 4 26B-A4B (cyankiwi AWQ-4bit, run via vLLM).** Produces the
   primary span labels on real chunks via structured-output JSON-schema
   decoding. Contributes the majority of spans on real chunks.
2. **`gheim.detectors.composite`.** A regex catalogue with checksum
   validators (ISO 13616 mod-97 for IBAN, EAN-13 mod-10 for AHV,
   ISO 7064 mod-11 for VAT-CHE, Luhn for credit cards) for
   structured-format PII (IBAN, AHV, VAT, credit card, phone, email,
   URL). Spans added by this stage are marked `"source": "regex"`.
3. **Slot-fill verifier on synthetic chunks.** A script that
   pre-generates payload values (Faker_CH names, addresses, phones;
   checksummed CH-IBAN, AHV, VAT-CHE; common secret formats) and
   prompts Gemma to embed them verbatim into natural prose. The
   verifier accepts a chunk only if every payload value appears
   exactly in the model output. Rejection rate was approximately 2%
   of generations. Spans added by this stage are marked
   `"source": "synthetic"`.

The first two annotators were run independently on each real chunk
and merged with regex-on-overlap priority. Label noise was measured by
comparing the dataset to a 4-way independent subagent labelling on a
stratified 176-chunk sample, yielding F1 = 0.664 as an upper-bound
estimate of label noise. The gap is largely policy-interpretation
rather than random error (see [Limitations](#limitations)).

#### Personal and sensitive information

The source corpora contain personal information published in public
Swiss documents (court rulings, parliamentary records, public-domain
news, web text). The PII labels in this dataset are *annotations on
already-public text*, not redactions of private text. Use is intended
for training and evaluating PII detection models, not for
re-identification or surveillance.

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

#### Synthetic gap-fill contribution

Synthetic chunks contribute the following per-cell increments. Bold
pre-synthesis counts mark cells where real-text coverage was sparse
enough to motivate the gap-fill.

| | de_ch | fr_ch | it_ch | rm | en |
|---|---:|---:|---:|---:|---:|
| `secret` (real: 39 / 8 / 2 / 3 / 3) | +3,000 | +3,000 | +3,000 | +3,000 | +4,500 |
| `account_number` (real: 1,092 / 992 / **182** / **200** / **14**) | 0 | 0 | +2,800 | +2,800 | +2,800 |
| `private_address` (real: 7,146 / 7,322 / 5,692 / 1,801 / **31**) | 0 | 0 | 0 | 0 | +2,800 |
| `private_phone` (real: 6,602 / 5,617 / 5,343 / 1,787 / **75**) | 0 | 0 | 0 | 0 | +4,300 |
| `private_email` (real: 671 / 1,431 / **310** / 562 / **10**) | +3,000 | +3,000 | +5,800 | +5,800 | +7,300 |
| `private_person` (side-effect of multi-PII templates) | +3,000 | +3,000 | +5,800 | +5,800 | +7,300 |

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

- **Public-figure over-representation.** The real-text labelling of
  Swiss legal and news content over-samples Swiss politicians, federal
  judges, and other frequently-mentioned public officials. The per-value
  cap (30 occurrences) limits any single name's contribution but does
  not change the underlying selection bias.
- **Court-ruling register dominates the long-form text.** Roughly 25% of
  chunks come from `entscheidsuche`; legal-text PII patterns are
  over-represented relative to casual web text. The Swiss-filtered
  FineWeb subset (~48%) provides partial counterweight.
- **Romansh has a single register.** Real RM chunks come almost entirely
  from the Romansh subset (literary corpus, RTR, Engadiner Post). RM
  coverage is genuine but narrow.
- **English is a regression anchor, not full coverage.** Real English
  chunks (~4.7k) exist to prevent fine-tunes from forgetting English;
  they do not constitute representative English-PII coverage.
- **Synthetic chunks are short and PII-dense.** The synthetic templates
  (Slack/Teams messages, CRM records, incident reports) typically span
  100-400 characters with 3-5 PII spans each, denser than the real
  Apertus chunks (200–2,000 characters, 1–3 spans). Models trained on
  the mix may over-expect PII density.
- **Synthetic person/email values are Faker-derived.** They look Swiss
  but are not drawn from a real-world distribution; collisions with real
  persons are possible by chance.

### Limitations

- **Annotations are machine-generated.** Real chunks are labelled by
  Gemma + checksum-validated regex; synthetic chunks by the slot-fill
  verifier. A 4-way independent labelling experiment on a 176-chunk
  sample yielded F1 ≈ 0.66 against a fresh-subagent majority. The gap
  is largely policy-interpretation (the aggressive-recall policy here
  flags publicly-listed institutional PII; fresh subagents tend to
  under-flag it), so this number is best read as an upper bound on
  label noise, not as a quality claim.
- **Synthetic spans are precision-biased.** The verifier labels only
  the values it asked the LLM to embed. If Gemma incidentally generated
  other PII-shaped text (an extra fake email in a list, etc.), that
  text is unlabelled. Models see confident positives but no within-chunk
  tricky negatives on synthetic data.
- **The `secret` cell in val/test is entirely synthetic.** Real-Swiss
  text yielded too few `secret` examples to populate the held-out
  splits (~7 chunks total before synthesis). Reported `secret` F1
  reflects detection of Faker-generated API keys, JWTs, and OAuth
  tokens embedded in LLM-generated developer prose. This is a reasonable
  proxy for the production "secret in dev/ops chat" leak path, but
  not a Swiss-grounded benchmark. Filter `synthetic == false` for a
  real-only `secret` evaluation.
- **Swiss German dialect (GSW) is not measured.** The fasttext language
  detector treats GSW as standard German; downstream applications
  encountering dialect text should run a GSW-specific evaluation.
- **Date detection is intentionally loose.** Following the
  aggressive-recall policy, the regex augmentation flags any plausible
  date pattern, including court-ruling dates that are arguably public
  metadata rather than personal information.

### Social impact and intended use

This dataset is intended to support training and evaluation of PII
detection models, particularly for Swiss-market applications where
data must be anonymised before being sent to cloud-hosted services. The
underlying source corpora are publicly published Swiss documents
(court rulings, parliament, web, Romansh literary/news), so the dataset
does not introduce new privacy exposure beyond what already exists in
the source. The categorical labelling (not entity-linked) makes the
dataset less useful for re-identification or attribution than direct
access to the source corpora would be.

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

> "gheim-ch-pii-171k" by Joel Barmettler, 2026.
> Built from `swiss-ai/apertus-pretrain-swiss` and `-romansh` plus
> Gemma-generated synthetic gap-fill.

---

## Reproducibility

The complete curation pipeline (sourcing, labelling, balancing,
splitting, synthesis, gap-fill placement) is available at
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim)
under `training/src/gheim_training/data/`. All stages use fixed
random seeds and produce deterministic outputs. The regex catalogue
with checksum validators lives at
`packages/gheim-py/src/gheim/detectors/composite.py`.

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

- **`swiss-ai/apertus-pretrain-{swiss,romansh}`.** without the curated
  Swiss text corpus, none of this would be possible. Thanks to the
  swiss-ai team for the open release.
- **`openai/privacy-filter`.** the 33-class BIOES label space used here
  was adopted from this model, so models trained on this dataset share
  the same output schema.
- **`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`.** the AWQ-quantised Gemma 4
  used for the bulk of LLM-based labelling and synthetic gap-fill.
- **`ai4privacy/openpii-1m` and `pii-masking-300k`.** informed the label
  REMAP and the multi-schema reconciliation. Used downstream as a
  training augment for English anchor and CH-region email rescue.
- **HuggingFace `datasets`, `transformers`, `accelerate`.** the entire
  pipeline, from streaming Apertus to publishing this card, runs on HF
  infra.
- **vLLM.** made the 24,900-chunk synthetic generation tractable on
  2× RTX 4090 in under an hour.
- **Faker.** for Faker_CH locales used in synthetic payload generation
  (names, addresses, phones).

---

## Versioning

Semantic versioning, dataset edition:

- **MAJOR.** schema changes (field added/removed/renamed), label-space
  changes (categories added/removed/merged), train/val/test resplit,
  significant content removal. Will require dataset-loading code changes
  for downstream users.
- **MINOR.** new languages added, new source corpora added, additional
  configs published, ≥10% growth in chunk count, opt-in fields added.
  Existing fields and splits stay backward compatible.
- **PATCH.** label corrections, span-offset fixes, documentation
  improvements, frontmatter updates. Chunk identities preserved.

Latest version: **1.0** (this card).

---

## Changelog

### v1.0 (2026-05-09, initial release)

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
- Estimated label-noise upper bound F1 ≈ 0.66 (4-way independent labelling, n=176).

---

## Contact

Joel Barmettler. joel.barmettler@gmail.com
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim) · [GitHub Issues](https://github.com/joelbarmettlerUZH/gheim/issues)
