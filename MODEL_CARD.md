---
language:
  - de
  - fr
  - it
  - rm
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
tags:
  - pii
  - ner
  - swiss
  - de-CH
  - fr-CH
  - it-CH
  - rm
  - privacy
base_model: TBD  # one of ZurichNLP/swissbert · FacebookAI/xlm-roberta-large · openai/privacy-filter
datasets:
  - joelbarmettler/gheim-ch-pii-171k
metrics:
  - f1
  - precision
  - recall
model-index:
  - name: gheim-ch-XXXm
    results:
      - task:
          type: token-classification
          name: PII NER (held-out test)
        dataset:
          type: joelbarmettler/gheim-ch-pii-171k
          name: gheim-ch-pii-171k (test split)
        metrics:
          - type: f1
            value: TBD       # filled in after final test eval
          - type: precision
            value: TBD
          - type: recall
            value: TBD
---

# gheim-ch-XXXm

> **Status: provisional.** This card describes the bake-off candidate for
> the eventual `joelbarmettler/gheim-ch-XXXm` release. Three base models
> (`ZurichNLP/swissbert` 270M, `FacebookAI/xlm-roberta-large` 550M,
> `openai/privacy-filter` 1.4B-MoE) are being evaluated on
> [`joelbarmettler/gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k);
> the leader at the end will be released. The numbers below are the
> currently-leading candidate (Stage 2 swissbert-270M, **val F1 = 0.9099**)
> and will be refreshed as the larger models complete training.

A multilingual PII detector for the four official Swiss languages
(de_CH, fr_CH, it_CH, rm) plus English. Token-classification model
fine-tuned on `gheim-ch-pii-171k` and aligned to the 33-class BIOES
output schema of [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter).

---

## Intended use

The model takes natural-language text and returns character spans
classified into 8 PII categories (`account_number`, `private_address`,
`private_date`, `private_email`, `private_person`, `private_phone`,
`private_url`, `secret`). Primary use case: redacting Swiss-market
text before sending it to an external LLM API (anonymising customer
support chats, legal documents, internal communications) so that
downstream cloud services do not receive personal data subject to
Swiss / EU data-protection requirements.

The model is intended for token-level redaction, not for entity linking
or re-identification. The recall-aggressive labelling policy of the
training data (institutional phone numbers and public-figure names
are flagged as PII) carries through to model behaviour; users
requiring stricter precision should pair the model with downstream
filtering, or consult the per-category trade-offs in the
[Performance](#performance) section.

---

## Quick start

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

repo = "joelbarmettler/gheim-ch-XXXm"
tok = AutoTokenizer.from_pretrained(repo)
mdl = AutoModelForTokenClassification.from_pretrained(repo)

ner = pipeline("token-classification", model=mdl, tokenizer=tok,
               aggregation_strategy="simple")

text = ("Bitte überweisen Sie an Müller AG, IBAN CH93 0076 2011 6238 5295 7, "
        "Werdstrasse 36, 8004 Zürich.")

for span in ner(text):
    print(f"  {span['entity_group']:<18}  {span['score']:.2f}  {span['word']!r}")
# Output:
#   private_person      0.99  'Müller AG'
#   account_number      1.00  'CH93 0076 2011 6238 5295 7'
#   private_address     0.97  'Werdstrasse 36, 8004 Zürich'
```

For production redaction with sentinel substitution and round-trip
de-anonymisation, see the [`gheim`](https://pypi.org/project/gheim/)
Python package, which wraps this model.

---

## Performance

All numbers are **strict-span F1** (seqeval) on the held-out test split
of `gheim-ch-pii-171k` (15,861 chunks). The test split is document-isolated
from train and was scored exactly once per model.

> Numbers below are from the **provisional candidate (swissbert-270M,
> Stage 2)** evaluated on **validation** (final-test eval to follow).
> They will be replaced with the released model's test numbers once the
> bake-off is complete.

### Overall

| Metric | Value |
|---|---:|
| **Overall F1** | **0.910** *(provisional, validation)* |
| Overall precision | 0.898 |
| Overall recall | 0.920 |

### Per-category F1 (provisional, validation)

| Category | F1 | Comment |
|---|---:|---|
| `secret` | 0.999 | Synthetic-only val cell — strong on Faker-generated API keys / JWTs / OAuth tokens; real-world secret performance not yet measured. |
| `private_email` | 0.982 | Strong, regex-shaped. |
| `private_phone` | 0.972 | Strong, regex-shaped. |
| `private_url` | 0.941 | Strong, regex-shaped. |
| `private_date` | 0.922 | Solid; date-format diversity learned. |
| `private_person` | 0.905 | Strong on Swiss-text names. |
| `private_address` | 0.770 | Moderate. Multi-token boundary ambiguity; addresses are intrinsically harder. |
| `account_number` | 0.663 | Weak as a pure ML signal. The composite-detector regex front-end (IBAN/AHV/VAT-CHE checksum validators) is recommended in production; it pushes effective recall on this category to near-1.0 with high precision. |

### Per-language F1 (provisional, validation)

*Will be filled in after final-test eval.*

### Comparison to baselines (provisional)

| Detector | Overall F1 | Notes |
|---|---:|---|
| **gheim-ch-XXXm** (this model) | **0.910** | Provisional; final number after test eval |
| Zero-shot `openai/privacy-filter` | TBD | To be re-measured on the new test split |
| `ZurichNLP/swissbert-ner` + regex | TBD | To be re-measured on the new test split |

---

## Training procedure

### Bake-off + sweep

Three base models were fine-tuned with an identical pipeline on
`gheim-ch-pii-171k`, then a 5 LR × 3 LLRD grid sweep (15 cells × 1 epoch
on the train-mix) selected the best (learning_rate, layer-wise LR decay)
per model. The leader is then trained for 3 full epochs at its winning
hyperparameters; final test evaluation runs once.

For the provisional swissbert candidate, the winning sweep cell was
**LR = 5e-5, LLRD = 1.0 (off)** (peak at sweep cell `lr5e-5_llrd100`,
F1 = 0.880 at 1 epoch). Stage 2 trained 3 full epochs and selected the
best checkpoint by validation F1 (epoch 2.50, step 3000, F1 = 0.9099).

### Hyperparameters (Stage 2, swissbert candidate)

| Hyperparameter | Value |
|---|---|
| Base model | `ZurichNLP/swissbert` (270M, XMOD architecture) |
| Train data | `data/built_v2/train` — 153,641 chunks (gheim-ch-pii train + AI4Privacy en/email augment) |
| Eval data | `data/built_v2/validation` — 15,834 chunks |
| Learning rate | 5e-5 |
| LR scheduler | Cosine, 5% warmup |
| Layer-wise LR decay | Off (LLRD = 1.0) |
| Optimizer | AdamW |
| Effective batch size | 128 (per-device 64 × 2 GPUs DDP) |
| Epochs | 3 |
| Max sequence length | 512 |
| Precision | fp32 (XMOD adapter requires non-bf16) |
| Best checkpoint | by validation overall_f1 |
| Hardware | 2× RTX 4090 (DDP) |
| Wall time | ~38 min train + 4 min eval |

### Training-data augmentation

The published training set (`gheim-ch-pii-171k` train split, 139,641
chunks) is supplemented at training time with:

- ~8,000 English chunks from `ai4privacy/openpii-1m` as an English
  regression anchor.
- ~6,000 CH-region de/fr/it chunks from `ai4privacy/openpii-1m`
  filtered to those containing email spans (sparse-cell rescue).

These augmentation chunks are **not part of the published dataset** —
they are training-only. The validation and test splits used for
evaluation contain no AI4Privacy data.

---

## Dataset

Trained on
[`joelbarmettler/gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k).
See that dataset's card for the full curation pipeline, label noise
characterisation, per-cell distribution, biases, and limitations.

Key facts inherited from the dataset:

- Annotations are machine-generated (Gemma 4 26B-A4B + checksum-validated
  regex + slot-fill verifier); the upper bound on label noise is
  F1 ≈ 0.66 (4-way independent labelling, n=176).
- The `secret` validation/test cells are populated entirely by synthetic
  gap-fill (LLM-generated developer chat / incident reports embedding
  Faker-generated API keys / JWTs).
- Real-Swiss-text validation/test contains zero `secret` examples; for
  a Swiss-text-only secret evaluation, filter the source dataset to
  `synthetic == false`.

---

## Limitations

- **Aggressive recall.** Following the dataset's labelling policy, the
  model flags publicly-listed institutional PII (court switchboard
  numbers, parliament email addresses, public-official names). For
  precision-critical applications, downstream filtering is needed.
- **Address F1 is moderate (~0.77).** Address spans have variable
  token boundaries; the model may under-extract long structured
  addresses.
- **Account-number F1 is weak (~0.66) as a pure ML signal.** Use the
  `gheim-ch-composite` variant (regex front-end with checksum
  validation) for production IBAN-CH / AHV / VAT-CHE / credit-card
  detection.
- **Swiss German dialect (GSW) is not measured.** The model treats GSW
  as standard German; users encountering dialect text should run a
  GSW-specific evaluation.
- **English is anchor-only on real data.** Real English coverage in
  training is ~4.7k chunks; the model is intended to not regress on
  English, not to claim English-language competence.
- **Romansh data is single-register.** Real Romansh chunks come almost
  entirely from the Romansh literary/news corpus.

---

## License

This model is released under the **Apache 2.0** license. The base model
weights are inherited from the upstream model's license:

- If based on `ZurichNLP/swissbert` — verify upstream license terms.
- If based on `FacebookAI/xlm-roberta-large` — Apache 2.0.
- If based on `openai/privacy-filter` — verify upstream license terms.

The training data (`gheim-ch-pii-171k`) is released under CC BY 4.0;
attribution to the source corpora (Apertus pretrain) is required.

---

## Citation

```bibtex
@misc{barmettler2026gheim_ch_model,
  title  = {gheim-ch-XXXm: A Swiss-market PII detection model},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/joelbarmettler/gheim-ch-XXXm},
  note   = {Trained on joelbarmettler/gheim-ch-pii-171k}
}
```

If you use this model, please also cite the underlying dataset:

```bibtex
@misc{barmettler2026gheim_ch_pii,
  title  = {gheim-ch-pii-171k: A Swiss-grounded PII NER dataset with synthetic gap-fill},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k}
}
```

---

## Maintenance and contact

Maintainer: Joel Barmettler — joel.barmettler@gmail.com ·
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim) ·
[Issues](https://github.com/joelbarmettlerUZH/gheim/issues)

For the broader gheim ecosystem (Python/Node libraries, server,
composite detector with regex front-end), see the GitHub repository.
