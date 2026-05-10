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

<!-- Methodology footnote: strict_span is computed via seqeval-token-level for HF/ONNX backends and via char-set-strict for spaCy/Presidio (which emit char spans directly). char F1 is char-label-aware and uses an identical formulation across all backends. The full numerical breakdown is in [`eval/positioning_matrix.json`](eval/positioning_matrix.json). -->
