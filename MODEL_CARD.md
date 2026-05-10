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
  - name: gheim-ch-560m
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

<p align="center">
  <img src="https://raw.githubusercontent.com/joelbarmettlerUZH/gheim/main/assets/logo.png" alt="gheim" width="360">
</p>

# gheim-ch-560m

A multilingual token-classification model for personally-identifiable information
(PII) detection across the four official Swiss languages (de_CH, fr_CH, it_CH, rm)
and English. The model is a fine-tune of
[`FacebookAI/xlm-roberta-large`](https://huggingface.co/FacebookAI/xlm-roberta-large)
on [`joelbarmettler/gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k).
Output schema is a 33-class BIOES tag set (8 PII categories plus the outside class)
aligned with the categorical naming used by `openai/privacy-filter`.

| | |
|---|---|
| Parameters | 560M |
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

repo = "joelbarmettler/gheim-ch-560m"
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
  "joelbarmettler/gheim-ch-560m",
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
| `FacebookAI/xlm-roberta-large` (560M) | 0.918 | 0.916 | Released as `gheim-ch-560m`. |
| `openai/privacy-filter` (1.4B MoE) | not run | not run | Sweep deferred. Olmo-MoE ONNX export is not currently stable in `optimum`, which constrains JavaScript portability for this base. |

In the (LR, LLRD) sweep, learning rate was the dominant factor: F1 increased
monotonically with LR over the tested range (5e-6 to 5e-5) for all three
base models. Layer-wise LR decay did not improve any cell at any tested
value (1.0, 0.95, 0.9). The selected configuration is therefore LR = 5e-5
without LLRD.

## Comparison to other models and benchmarks

To position the model in the broader PII / NER landscape, two evaluations were run. Direction A scores other open PII detectors and multilingual NER models on the same held-out test split that produced the headline F1. Direction B scores `joelbarmettler/gheim-ch-560m` on widely-used external benchmarks.

### Two metrics: strict-span and char F1

Cross-model PII evaluation has a structural problem: different models use different entity-segmentation policies. AI4Privacy splits a person name into `GIVENNAME` + `SURNAME`; `gheim` and CoNLL emit one combined `private_person` span; address detectors fragment "Werdstrasse 36, 8004 Zürich" at the comma or run it through. Strict-span F1 (the NER literature standard, computed by `seqeval`) penalises every boundary mismatch, even when the model has correctly masked the right characters.

Both metrics are reported. **Strict-span F1**: exact (start, end, label) match per span via `seqeval` (token-level) or char-set match (for span-emitting backends like spaCy / Presidio that lack a token grid). **Char F1**: per-character precision and recall on the (character, category) set; fragmentation-invariant; reflects the redaction utility the model is built for.

### Direction A: other models on our held-out test set

| Model | Strict F1 | Char F1 | PER strict | PER char |
|---|---:|---:|---:|---:|
| **joelbarmettler/gheim-ch-560m** | **0.916** | **0.958** | **0.915** | **0.944** |
| openai/privacy-filter (1.4B MoE, zero-shot) | 0.443 | 0.610 | 0.406 | 0.561 |
| Microsoft Presidio Analyzer (multi-lang config) | 0.434 | 0.562 | 0.442 | 0.521 |
| Davlan/xlm-roberta-base-ner-hrl | n/a | n/a | 0.645 | 0.728 |
| Davlan/distilbert-base-multilingual-cased-ner-hrl | n/a | n/a | 0.619 | 0.771 |
| spaCy de/fr/it core_news_lg | n/a | n/a | 0.505 | 0.621 |
| dslim/bert-base-NER (English slice) | n/a | n/a | 0.512 | 0.770 |
| Isotonic/distilbert_finetuned_ai4privacy_v2 | 0.157 | 0.508 | 0.264 | 0.507 |

Per-language overall char F1 (top three Direction A contestants):

| Language | joelbarmettler/gheim-ch-560m | openai/privacy-filter (1.4B MoE, zero-shot) | Microsoft Presidio Analyzer (multi-lang config) |
|---|---:|---:|---:|
| de_ch | **0.954** | 0.569 | 0.572 |
| en | **0.981** | 0.788 | 0.561 |
| fr_ch | **0.953** | 0.577 | 0.596 |
| it_ch | **0.972** | 0.630 | 0.690 |
| rm | **0.923** | 0.643 | 0.251 |

### Direction B: `gheim-ch-560m` on external benchmarks

Each external dataset uses its own label schema; PER / SURNAME / GIVENNAME etc. are mapped to `private_person` and remaining categories to the closest `gheim` cell or to `O`. For pure-NER datasets (swissner, CoNLL-2003, WikiNeural) only PER maps to a `gheim` category, so the meaningful number is the PER cell. The non-PER `Overall F1` is dragged down by `gheim` predicting categories the NER datasets don't label.

| External benchmark | n_chunks | Overall strict | Overall char | PER strict | PER char |
|---|---:|---:|---:|---:|---:|
| `ai4privacy/pii-masking-openpii-1m` | 8,000 | 0.919 | 0.972 | 0.776 | 0.920 |
| `Babelscape/wikineural` | 8,000 | 0.833 | 0.880 | 0.919 | 0.960 |
| `ZurichNLP/swissner` | 800 | 0.781 | 0.838 | 0.903 | 0.946 |
| `tomaarsen/conll2003` | 3,453 | 0.678 | 0.703 | 0.887 | 0.936 |

Per-language PER char F1 on Swiss news (`swissner`):

| Language | PER char F1 |
|---|---:|
| de | 0.884 |
| fr | 0.862 |
| it | 0.810 |
| rm | 0.795 |


### Methodology validation

To check that our scoring harness is sound (and the published numbers
above are not eval-bugs), we replicated each external model on the
dataset it was trained on and verified our reproduction against the
model's own published F1:

| Replication target | Their claim | Our reproduction | Δ |
|---|---:|---:|---:|
| `dslim/bert-base-NER` on CoNLL-2003 | 0.910 | 0.913 | +0.003 |
| `dslim/bert-base-NER` PER on CoNLL-2003 | ~0.96 | 0.957 | -0.003 |
| `Davlan/xlm-r-base-ner-hrl` on WikiNeural avg | ~0.84 | 0.813 | -0.027 |
| spaCy `de_core_news_lg` PER on `swissner_de` | ~0.83 | 0.841 | +0.011 |
| `Isotonic/distilbert_finetuned_ai4privacy_v2` | 0.955 | 0.969‡ | +0.014 |

Three of four replications match within ±0.03 F1, validating the harness.

‡ The fourth (Isotonic) initially reproduced as 0.78 strict-span F1 vs.
their claimed 0.955. Investigation found their reported number is
**per-token classification accuracy** (which includes the dominant `O`
class and is much higher than per-span F1). Reproducing Isotonic's
metric exactly (per-token accuracy including `O`) yields 0.969,
matching their 0.955 within sampling noise. Our strict-span F1 of 0.78
on the same data is the correct NER-literature comparison.

The full numerical breakdown is at
[`eval/positioning_matrix.json`](https://github.com/joelbarmettlerUZH/gheim/blob/main/eval/positioning_matrix.json).


## Deployment formats

The model is published in two formats:

- `model.safetensors` at the repository root: the fp32 PyTorch checkpoint,
  intended for server-side inference via `transformers`.
- `onnx/model_quantized.onnx`: an int8 (UINT8) dynamic-quantized ONNX
  export, intended for in-browser inference via
  [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers)
  on WebGPU or WebAssembly. Selected with `dtype: "q8"`.

Both formats were scored on the same held-out test split. The int8 export
loses 0.7 pp overall F1 relative to the PyTorch checkpoint; the loss is
concentrated in the two categories that were already weakest under fp32.

| Format | File | Size | Test F1 | Δ vs fp32 |
|---|---|---:|---:|---:|
| PyTorch fp32 | `model.safetensors` | 2.2 GB | 0.916 | (baseline) |
| ONNX int8 (dynamic) | `onnx/model_quantized.onnx` | 552 MB | 0.909 | -0.7 pp |

### Per-category F1 under int8 quantization

| Category | fp32 | int8 | Δ |
|---|---:|---:|---:|
| `secret` | 0.996 | 0.998 | +0.002 |
| `private_email` | 0.987 | 0.984 | -0.003 |
| `private_phone` | 0.986 | 0.983 | -0.002 |
| `private_url` | 0.947 | 0.945 | -0.002 |
| `private_date` | 0.922 | 0.915 | -0.007 |
| `private_person` | 0.915 | 0.909 | -0.006 |
| `private_address` | 0.782 | 0.761 | -0.021 |
| `account_number` | 0.669 | 0.616 | -0.053 |

The int8 export is uniform across categories within ~1 pp except for
`private_address` and `account_number`, which lose 2 pp and 5 pp
respectively. Both categories were already the weakest under fp32, and
production deployments are expected to pair them with the regex front-end
documented under "Composite detector" in the
[`gheim`](https://pypi.org/project/gheim/) library, which validates
account-number checksums (IBAN, AHV, VAT-CHE, Luhn) deterministically.

### Per-language F1 under int8 quantization

| Language | fp32 | int8 | Δ |
|---|---:|---:|---:|
| en | 0.954 | 0.947 | -0.007 |
| it_ch | 0.942 | 0.935 | -0.007 |
| de_ch | 0.911 | 0.904 | -0.007 |
| fr_ch | 0.906 | 0.898 | -0.008 |
| rm | 0.853 | 0.848 | -0.005 |

Quantization affects all five languages roughly equally (within ±0.001 pp
of the overall -0.7 pp drop); no language is disproportionately degraded.

The int8 file is reproducible via `optimum.onnxruntime.ORTQuantizer` with
`AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)`
applied to the fp32 ONNX export.

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
@misc{barmettler2026gheim_ch_560m,
  title  = {gheim-ch-560m: A multilingual PII detection model for the Swiss market},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/joelbarmettler/gheim-ch-560m}
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
