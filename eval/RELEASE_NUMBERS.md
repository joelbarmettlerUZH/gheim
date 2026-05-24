# Canonical gheim release numbers

**Single source of truth for every F1 that ships in the model cards, dataset card, and tech report.** Re-generate via `uv run python -m gheim_training.eval.compile_release_numbers`. All numbers below come from the JSON files in `eval/`; any discrepancy between this report and a published artefact is a bug in the artefact, not in this report.

## In-domain test split (data/built, test, 21,246 chunks)

Headline strict-span F1 and char F1 on the held-out test split.

| Checkpoint | License | Strict F1 | Char F1 | Precision (strict) | Recall (strict) |
|---|---|---:|---:|---:|---:|
| **gheim-ch-560m (Apache 2.0, this release)** | Apache 2.0 | 0.910 | 0.946 | 0.890 | 0.932 |
| **gheim-ch-560m-research (CC BY-NC-SA, this release)** | CC BY-NC-SA 4.0 | 0.912 | 0.946 | 0.894 | 0.929 |

### Per-(language × category) char F1 — `gheim-ch-560m (Apache 2.0)` on the in-domain test split

| Category | de_ch | fr_ch | it_ch | rm | en | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| `account_number` | 0.994 | 0.998 | 0.990 | 0.971 | 1.000 | 0.994 |
| `private_address` | 0.933 | 0.911 | 0.916 | 0.853 | 0.955 | 0.917 |
| `private_date` | 0.951 | 0.908 | 0.952 | 0.919 | 0.883 | 0.933 |
| `private_email` | 0.996 | 1.000 | 0.997 | 0.989 | 1.000 | 0.997 |
| `private_person` | 0.913 | 0.939 | 0.951 | 0.909 | 0.955 | 0.930 |
| `private_phone` | 0.995 | 0.996 | 0.996 | 0.985 | 1.000 | 0.995 |
| `private_url` | 0.990 | 0.993 | 0.993 | 0.994 | 0.970 | 0.991 |
| `secret` | 0.994 | 0.999 | 1.000 | 0.999 | n/a | 0.997 |
| **Avg.** | **0.944** | **0.931** | **0.956** | **0.918** | **0.954** | **0.940** |

### Per-(language × category) char F1 — `gheim-ch-560m-research` on the in-domain test split

| Category | de_ch | fr_ch | it_ch | rm | en | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| `account_number` | 0.994 | 0.998 | 0.992 | 0.971 | 1.000 | 0.995 |
| `private_address` | 0.932 | 0.912 | 0.920 | 0.851 | 0.962 | 0.917 |
| `private_date` | 0.949 | 0.909 | 0.951 | 0.919 | 0.899 | 0.933 |
| `private_email` | 0.996 | 0.998 | 0.999 | 0.991 | 1.000 | 0.996 |
| `private_person` | 0.913 | 0.939 | 0.952 | 0.907 | 0.959 | 0.930 |
| `private_phone` | 0.995 | 0.998 | 0.997 | 0.987 | 1.000 | 0.995 |
| `private_url` | 0.991 | 0.994 | 0.993 | 0.993 | 0.967 | 0.991 |
| `secret` | 0.993 | 0.999 | 1.000 | 1.000 | n/a | 0.997 |
| **Avg.** | **0.943** | **0.931** | **0.956** | **0.916** | **0.959** | **0.940** |

## Direction A — other detectors on our test split (21,246 chunks)

Metrics: strict-span F1 (seqeval) and char-level label-aware F1, plus the `private_person`-cell numbers in the right columns. PER-only models emit only person mentions (no email / phone / IBAN / etc.); their overall 8-category F1 is reported as n/a.

| Model | License | Strict F1 | Char F1 | PER strict | PER char |
|---|---|---:|---:|---:|---:|
| gheim-ch-560m (Apache 2.0, this release) | Apache 2.0 | **0.910** | **0.946** | **0.896** | **0.929** |
| gheim-ch-560m-research (CC BY-NC-SA, this release) | CC BY-NC-SA 4.0 | **0.912** | **0.946** | **0.898** | **0.929** |
| dslim/bert-base-NER (English slice) | MIT | n/a | n/a | 0.561 | 0.834 |
| ZurichNLP/swissbert-ner + regex hybrid | CC BY 4.0 | 0.346 | 0.612 | 0.464 | 0.770 |
| Davlan/distilbert-base-multilingual-cased-ner-hrl | Apache 2.0 | n/a | n/a | 0.569 | 0.758 |
| Davlan/xlm-roberta-base-ner-hrl | Apache 2.0 | n/a | n/a | 0.610 | 0.733 |
| openai/privacy-filter (1.4B MoE, ONNX, zero-shot) | Apache 2.0 | 0.449 | 0.619 | 0.436 | 0.582 |
| spaCy de/fr/it_core_news_lg | MIT | n/a | n/a | 0.473 | 0.574 |
| Microsoft Presidio Analyzer (multi-lang config) | MIT | 0.398 | 0.501 | 0.433 | 0.519 |
| Isotonic/distilbert_finetuned_ai4privacy_v2 | Apache 2.0 | 0.140 | 0.441 | 0.243 | 0.513 |

## Direction B — gheim variants on external benchmarks

Metric: `private_person`-cell character-level F1 on each external benchmark. Three of the six benchmarks (openpii-1m, WikiNeural, CoNLL-2003) are present in the research variant's training mix, so its numbers on those rows reflect in-distribution generalisation rather than zero-shot transfer; the other three (swissner, open-pii-500k, gretel_finance) are zero-shot for every variant.

| Benchmark | License | Baseline (Apache 2.0) | Commercial (experimental) | Research (CC BY-NC-SA) | Note |
|---|---|---:|---:|---:|---|
| ZurichNLP/swissner (Swiss-news NER) | CC BY 4.0 | 0.702 | 0.844 | 0.903 | swissner has no train split; all variants are zero-shot |
| ai4privacy/pii-masking-openpii-1m | Apache 2.0 / CC BY 4.0 | 0.938 | 0.995 | 0.995 | research/commercial trained on openpii train split |
| ai4privacy/open-pii-masking-500k | CC BY 4.0 | 0.933 | 0.973 | 0.982 | zero-shot for all variants |
| gretelai/synthetic_pii_finance_multilingual | Apache 2.0 | 0.624 | 0.619 | 0.627 | zero-shot for all variants |
| Babelscape/WikiNeural | CC BY-NC-SA 4.0 | 0.808 | 0.681 | 0.795 | research trained on WikiNeural train split; commercial trained on WikiAnn (substitute) |
| tomaarsen/conll2003 (PER only) | research-only (Reuters) | 0.911 | 0.548 | 0.765 | research trained on CoNLL-2003 train split |

## Per-language swissner — Swiss-news NER cross-domain (zero-shot for all variants)

Per-language char F1 on ZurichNLP/swissner's de / fr / it / rm test splits (200 chunks each). `swissner` has no train split, so every variant is zero-shot on this benchmark.

| Language | Baseline | Commercial | Research |
|---|---:|---:|---:|
| de | 0.539 | 0.805 | 0.931 |
| fr | 0.761 | 0.845 | 0.913 |
| it | 0.643 | 0.745 | 0.856 |
| rm | 0.409 | 0.552 | 0.873 |

## ONNX deployment-format deltas (gheim-ch-560m, Apache 2.0)

| Format | Size | Test strict F1 | Test char F1 | Δ strict | Δ char |
|---|---:|---:|---:|---:|---:|
| PyTorch fp32 | 2.2 GB | 0.910 | 0.946 | (baseline) | (baseline) |
| ONNX fp32    | 2.2 GB | 0.910 | 0.946 | 0.000 | 0.000 |
| ONNX int8 (dynamic) | 557 MB | 0.904 | 0.945 | -0.0061 | -0.0013 |
