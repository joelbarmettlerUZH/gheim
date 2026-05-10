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
base_model: FacebookAI/xlm-roberta-large
datasets:
  - joelbarmettler/gheim-ch-pii-171k
metrics:
  - f1
  - precision
  - recall
model-index:
  - name: gheim-ch-559m
    results:
      - task:
          type: token-classification
          name: PII NER (held-out test)
        dataset:
          type: joelbarmettler/gheim-ch-pii-171k
          name: gheim-ch-pii-171k (test split)
        metrics:
          - type: f1
            value: 0.9161
          - type: precision
            value: 0.9067
          - type: recall
            value: 0.9258
---

# gheim-ch-559m

A multilingual token-classification model for personally-identifiable information
(PII) detection across the four official Swiss languages (de_CH, fr_CH, it_CH, rm)
and English. The model is a fine-tune of
[`FacebookAI/xlm-roberta-large`](https://huggingface.co/FacebookAI/xlm-roberta-large)
on [`joelbarmettler/gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k).
Output schema is a 33-class BIOES tag set (8 PII categories plus the outside class)
aligned with the categorical naming used by `openai/privacy-filter`.

| | |
|---|---|
| Parameters | 559M |
| Languages | de_CH, fr_CH, it_CH, rm, en |
| Categories | account_number, private_address, private_date, private_email, private_person, private_phone, private_url, secret |
| Tag scheme | BIOES (33 classes) |
| Max sequence length | 512 |
| License | Apache 2.0 |

## Intended use

The model classifies character-level spans of PII so that text can be redacted
prior to transmission to systems where personal data should not appear (for
example, third-party LLM APIs hosted outside the data subject's jurisdiction).
Output spans are intended for substitution or masking, not for entity linking
or re-identification. The training data follows a recall-oriented labelling
policy under which publicly-listed institutional information (e.g. court
switchboard numbers, parliament email addresses, public-official names) is
flagged as PII. Applications requiring stricter precision should pair model
output with downstream filtering.

## Usage

### Python (transformers)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

repo = "joelbarmettler/gheim-ch-559m"
tok = AutoTokenizer.from_pretrained(repo)
mdl = AutoModelForTokenClassification.from_pretrained(repo)
ner = pipeline("token-classification", model=mdl, tokenizer=tok,
               aggregation_strategy="simple")

text = ("Bitte überweisen Sie an Müller AG, IBAN CH9300762011623852957, "
        "Werdstrasse 36, 8004 Zürich.")
for span in ner(text):
    print(f"{span['entity_group']:<18} {span['score']:.2f} {span['word']!r}")
```

### JavaScript (transformers.js)

The repository ships an ONNX export under `onnx/` so the model loads in
[`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers)
on Node, Bun, and supported browsers.

```ts
import { pipeline } from "@huggingface/transformers";

const ner = await pipeline("token-classification",
  "joelbarmettler/gheim-ch-559m",
  { aggregation_strategy: "simple" });
const out = await ner("Email me at alice@example.ch, phone +41 44 268 12 34.");
```

### gheim round-trip wrapper

For sentinel-based round-trip anonymisation (text out, text back through an
external LLM, restored output) the [`gheim`](https://pypi.org/project/gheim/)
Python package and the [`gheim`](https://www.npmjs.com/package/gheim)
JavaScript package wrap this model under their `LocalDetector` and expose a
streaming-aware decode path for the substitution sentinels.

## Performance

All numbers are strict-span F1 (`seqeval`) on the held-out test split of
`gheim-ch-pii-171k` (15,861 chunks, document-isolated from the training
split for the real-text portion). The test set was scored once.

### Overall

| Metric | Test | Validation |
|---|---:|---:|
| F1 | 0.916 | 0.918 |
| Precision | 0.907 | 0.908 |
| Recall | 0.926 | 0.926 |

The validation/test gap is 0.002, consistent with no observable overfitting
to the validation set.

### Per category

| Category | F1 | Precision | Recall | n_gold |
|---|---:|---:|---:|---:|
| `secret` | 0.996 | 0.994 | 0.999 | 1,104 |
| `private_email` | 0.987 | 0.987 | 0.987 | 1,712 |
| `private_phone` | 0.986 | 0.984 | 0.988 | 3,329 |
| `private_url` | 0.947 | 0.936 | 0.957 | 3,991 |
| `private_date` | 0.922 | 0.902 | 0.944 | 9,678 |
| `private_person` | 0.915 | 0.913 | 0.917 | 14,726 |
| `private_address` | 0.782 | 0.759 | 0.806 | 3,507 |
| `account_number` | 0.669 | 0.694 | 0.645 | 502 |

### Per language

| Language | F1 | Precision | Recall | n_chunks |
|---|---:|---:|---:|---:|
| en | 0.954 | 0.942 | 0.966 | 893 |
| it_ch | 0.942 | 0.940 | 0.945 | 4,169 |
| de_ch | 0.911 | 0.908 | 0.915 | 4,753 |
| fr_ch | 0.906 | 0.890 | 0.923 | 4,705 |
| rm | 0.853 | 0.828 | 0.880 | 1,341 |

### Model selection

Three base models were evaluated under an identical fine-tuning pipeline.
Each model received a 5 × 3 sweep over (learning rate, layer-wise LR decay)
at 1 epoch on the training mix; the winning configuration per model was
then trained for 3 full epochs and selected by best validation F1.

| Base model | Stage 2 val F1 | Test F1 | Note |
|---|---:|---:|---|
| `ZurichNLP/swissbert` (270M) | 0.910 | not run | Trained but not selected for the leader test eval. |
| `FacebookAI/xlm-roberta-large` (559M) | 0.918 | 0.916 | Released as `gheim-ch-559m`. |
| `openai/privacy-filter` (1.4B MoE) | not run | not run | Sweep deferred. Olmo-MoE ONNX export is not currently stable in `optimum`, which constrains JavaScript portability for this base. |

In the (LR, LLRD) sweep, learning rate was the dominant factor: F1 increased
monotonically with LR over the tested range (5e-6 to 5e-5) for all three
base models. Layer-wise LR decay did not improve any cell at any tested
value (1.0, 0.95, 0.9). The selected configuration is therefore LR = 5e-5
without LLRD.

## Training procedure

### Configuration (Stage 2)

| Hyperparameter | Value |
|---|---|
| Base model | `FacebookAI/xlm-roberta-large` |
| Train data | `data/built_v2/train`, 153,641 chunks (gheim-ch-pii-171k train split plus AI4Privacy en/email augmentation) |
| Validation data | `data/built_v2/validation`, 15,834 chunks |
| Optimiser | AdamW |
| Learning rate | 5e-5 |
| LR scheduler | Cosine, 5% warmup |
| Layer-wise LR decay | 1.0 (off) |
| Effective batch size | 128 (per-device 64 × 2 GPUs DDP) |
| Epochs | 3; best checkpoint at epoch 2.50 |
| Max sequence length | 512 |
| Precision | bf16 |
| Hardware | 2 × RTX 4090 (DDP) |
| Wall time | ≈ 52 minutes train, ≈ 4 minutes evaluation |

The best checkpoint was selected by validation `overall_f1`.

### Training-data augmentation

The published training split of `gheim-ch-pii-171k` (139,641 chunks) was
supplemented at training time with two slices drawn from
`ai4privacy/pii-masking-openpii-1m`:

- approximately 8,000 English chunks as an English regression anchor;
- approximately 6,000 chunks (de, fr, it; CH region only) containing at
  least one `private_email` span, addressing a sparse-cell coverage gap.

These supplementary chunks are training-only. The validation and test
splits used for the metrics above contain no AI4Privacy data.

## Dataset

Trained on
[`joelbarmettler/gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k).
That dataset's card documents the curation pipeline, per-cell distribution,
label-noise characterisation, and limitations. Two facts are load-bearing
here:

1. Annotations are machine-generated (LLM labelling plus checksum-validated
   regex augmentation, with slot-fill synthesis for sparse cells). A 4-way
   independent labelling on a 176-chunk sample yielded F1 ≈ 0.66 against a
   majority vote, which serves as an upper bound on label noise.
2. The `secret` cell of the validation and test splits is populated
   entirely by synthetic gap-fill chunks. Real Swiss text contributed
   fewer than ten secret-bearing examples in total; for a real-only
   evaluation of `secret`, filter the source dataset to `synthetic == false`.

## Limitations

1. The model inherits the recall-oriented policy of the training data and
   therefore flags publicly-listed institutional contact information.
   Applications needing stricter precision should apply downstream
   filtering or use a private-vs-public-entity post-classifier.
2. `private_address` test F1 is 0.78. Boundary placement on multi-token
   addresses is the dominant error mode.
3. `account_number` test F1 is 0.67. For production use, pair the model
   with the regex front-end documented in the `gheim` library, which
   applies checksum validation (ISO 13616 mod-97 for IBAN, EAN-13 mod-10
   for AHV, ISO 7064 mod-11 for VAT-CHE, Luhn for credit cards) and
   handles structured-format PII deterministically.
4. Romansh test F1 is 0.85, the weakest among the five languages. Romansh
   training material is dominated by a single literary/journalistic
   register; performance on dialectal or technical RM text is unmeasured.
5. Swiss German dialect (GSW) is not measured. The fasttext detector used
   in data preparation labels GSW as standard German.
6. The model is intended for redaction. It is not designed to identify or
   link entities to real persons, and it does not return entity-linked
   identifiers.

## License

Released under Apache 2.0, inherited from the base model
`FacebookAI/xlm-roberta-large`. The training data
(`joelbarmettler/gheim-ch-pii-171k`) is released under CC BY 4.0;
attribution to its upstream corpora (the `swiss-ai/apertus-pretrain-*`
datasets) is required when reusing the data.

## Citation

```bibtex
@misc{barmettler2026gheim_ch_559m,
  title  = {gheim-ch-559m: A multilingual PII detection model for the Swiss market},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/joelbarmettler/gheim-ch-559m}
}
```

If the model is used in published work, please also cite the dataset:

```bibtex
@misc{barmettler2026gheim_ch_pii,
  title  = {gheim-ch-pii-171k: A Swiss-grounded PII NER dataset with synthetic gap-fill},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k}
}
```

## Maintainer

Joel Barmettler. Source code, issue tracker, and the wider gheim ecosystem
(Python and Node libraries, redaction server, composite detector) are at
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim).
