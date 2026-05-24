## Comparison to other models and benchmarks

To position the model in the broader PII / NER landscape, two evaluations were run. **Direction A** scores other open PII detectors and multilingual NER models on the same held-out test split that produced the headline F1. **Direction B** scores both released `gheim` checkpoints on widely-used external benchmarks.

### Two metrics: strict-span and char F1

Cross-model PII evaluation has a structural problem: different models use different entity-segmentation policies. AI4Privacy splits a person name into `GIVENNAME` + `SURNAME`; `gheim` and CoNLL emit one combined `private_person` span; address detectors fragment "Werdstrasse 36, 8004 Zürich" at the comma or run it through. Strict-span F1 (the NER literature standard, computed by `seqeval`) penalises every boundary mismatch, even when the model has correctly masked the right characters.

Both metrics are reported. **Strict-span F1**: exact (start, end, label) match per span via `seqeval` (token-level) or char-set match (for span-emitting backends like spaCy / Presidio that lack a token grid). **Char F1**: per-character precision and recall on the (character, category) set; fragmentation-invariant; reflects the redaction utility the model is built for.

### Direction A: other models on our held-out test set

`joelbarmettler/gheim-ch-pii-212k` test split, 21,246 chunks, document-level isolated from train for the real-text portion. The two `gheim` checkpoints have essentially identical in-distribution numbers; the cross-domain differences are reported in Direction B.

| Model | License | Strict F1 | Char F1 | PER strict | PER char |
|---|---|---:|---:|---:|---:|
| **`joelbarmettler/gheim-ch-560m`** (this release) | Apache 2.0 | **0.910** | **0.946** | **0.896** | **0.929** |
| **`joelbarmettler/gheim-ch-560m-research`** | CC BY-NC-SA 4.0 (research-only) | **0.912** | **0.946** | **0.898** | **0.929** |
| `dslim/bert-base-NER` (English slice) | MIT | n/a | n/a | 0.561 | 0.834 |
| `ZurichNLP/swissbert-ner` + regex hybrid | CC BY 4.0 | 0.346 | 0.613 | 0.464 | 0.770 |
| `Davlan/distilbert-base-multilingual-cased-ner-hrl` | Apache 2.0 | n/a | n/a | 0.569 | 0.758 |
| `Davlan/xlm-roberta-base-ner-hrl` | Apache 2.0 | n/a | n/a | 0.610 | 0.733 |
| `openai/privacy-filter` (ONNX, 1.4B MoE, zero-shot) | Apache 2.0 | 0.449 | 0.619 | 0.436 | 0.582 |
| spaCy `de/fr/it_core_news_lg` | MIT | n/a | n/a | 0.473 | 0.574 |
| Microsoft Presidio Analyzer (multi-lang config) | MIT | 0.398 | 0.501 | 0.433 | 0.519 |
| `Isotonic/distilbert_finetuned_ai4privacy_v2` | Apache 2.0 | 0.140 | 0.441 | 0.243 | 0.513 |

PER-only models (Davlan, dslim, spaCy) emit only person mentions, so their overall 8-category F1 is reported as `n/a`. `openai/privacy-filter` is scored via its shipped ONNX export (`onnx/model.onnx`) because its custom `model_type=openai_privacy_filter` is not registered in `transformers` 4.57; the eval harness applies a config patch (`model_type → xlm-roberta`) so AutoConfig can load the id-to-label mapping, then runs inference through the ONNX graph unmodified. `ZurichNLP/swissbert-ner` is run via the XMOD-adapter-aware baseline runner with regex fallbacks for non-PER categories (IBAN, AHV, VAT-CHE, phone, email, URL, secret).

### Direction B: `gheim` checkpoints on external benchmarks (zero-shot)

PER character-level F1 on each benchmark. Both `gheim` checkpoints are scored zero-shot: neither has seen the test material at training time. The research variant has been trained on the *train* split of `ai4privacy/openpii-1m`, `Babelscape/wikineural`, and CoNLL-2003 (in addition to the in-domain Swiss data); the Apache-2.0 baseline has not. Numbers are character-level F1, the metric that survives schema-fragmentation noise.

| External benchmark | n | License | Apache-2.0 baseline | Research |
|---|---:|---|---:|---:|
| `ZurichNLP/swissner` (Swiss-news NER, **zero-shot for both**, all 4 langs incl. RM) | 800 | CC BY 4.0 | 0.702 | **0.903** |
| `ai4privacy/pii-masking-openpii-1m` (research trained on train split) | 8,000 | Apache 2.0 | 0.938 | **0.995**† |
| `ai4privacy/open-pii-masking-500k-ai4privacy` (**zero-shot for both**) | 8,000 | CC BY 4.0 | 0.933 | **0.982** |
| `gretelai/synthetic_pii_finance_multilingual` (**zero-shot for both**) | 4,800 | Apache 2.0 | 0.624 | 0.627 |
| `Babelscape/wikineural` (research trained on train split) | 8,000 | CC BY-NC-SA 4.0 | **0.808** | 0.795† |
| `tomaarsen/conll2003` (research trained on train split, PER only) | 3,453 | research-only | **0.911** | 0.765† |

†Research-variant numbers on these three rows reflect in-distribution generalisation, not zero-shot cross-domain transfer (the research training mix includes the train splits of these three datasets). On the two news-style NER benchmarks (`WikiNeural`, `CoNLL-2003`) the research variant actually regresses against the in-domain-only Apache baseline because its broader 8-category output schema produces non-PER false positives on news text that the Apache baseline does not.

#### Per-language `swissner` PER char F1 — the headline cross-domain test

`swissner` is the only Swiss-news NER benchmark that includes Romansh. The Apache-2.0 baseline's per-language drop on `swissner` is the main reason the research variant exists.

| Language | Apache-2.0 baseline | Research | Δ |
|---|---:|---:|---:|
| de | 0.539 | **0.931** | +39 pp |
| fr | 0.761 | **0.913** | +15 pp |
| it | 0.643 | **0.856** | +21 pp |
| rm | 0.409 | **0.873** | **+46 pp** |

The Romansh gain is the most surprising: none of the external training corpora include Romansh, but the cross-lingual transfer in XLM-R's shared encoder body from the additional de/fr/it/en supervision lifts RM by 46 pp.

#### Choosing a variant

| Use case | Variant |
|---|---|
| Production deployment, customer data, commercial product | `gheim-ch-560m` (Apache 2.0) |
| Swiss court / parliament / web text redaction (the original target distribution) | either — equivalent in-distribution |
| Swiss-news article redaction (any of de/fr/it/rm) | `gheim-ch-560m-research` |
| Cross-distribution PII / NER experiments, papers, evaluations | `gheim-ch-560m-research` |

### Methodology validation

To check that the harness is sound, each external baseline was replicated on its own training-distribution dataset and the reproduction compared to the baseline's own published number; three of four match within ±0.03. See `paper/paper.pdf` Appendix A4 (Table replication) for the full breakdown.

<!-- Methodology footnote: strict_span is computed via seqeval-token-level for HF/ONNX backends and via char-set-strict for spaCy/Presidio (which emit char spans directly). char F1 is char-label-aware and uses an identical formulation across all backends. The full numerical breakdown is in [`eval/positioning_matrix.json`](eval/positioning_matrix.json). -->
