---
language:
  - de
  - fr
  - it
  - rm
  - en
license: cc-by-nc-sa-4.0
license_name: cc-by-nc-sa-4.0-with-reuters-research-only-restriction
license_link: https://creativecommons.org/licenses/by-nc-sa/4.0/
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
  - research-only
  - non-commercial
base_model: FacebookAI/xlm-roberta-large
datasets:
  - joelbarmettler/gheim-ch-pii-212k
  - ai4privacy/pii-masking-openpii-1m
  - Babelscape/wikineural
  - tomaarsen/conll2003
metrics:
  - f1
  - precision
  - recall
model-index:
  - name: gheim-ch-560m-research
    results:
      - task:
          type: token-classification
          name: PII NER (in-domain held-out test)
        dataset:
          type: joelbarmettler/gheim-ch-pii-212k
          name: gheim-ch-pii-212k (test split)
        metrics:
          - type: f1
            value: 0.9115
          - type: precision
            value: 0.8944
          - type: recall
            value: 0.9293
      - task:
          type: token-classification
          name: Swiss-news PER NER (zero-shot cross-domain)
        dataset:
          type: ZurichNLP/swissner
          name: swissner (de+fr+it+rm test)
        metrics:
          - type: f1
            value: 0.903
            name: PER char F1 (overall)
---

<p align="center">
  <img src="https://raw.githubusercontent.com/joelbarmettlerUZH/gheim/main/assets/logo.png" alt="gheim" width="360">
</p>

# gheim-ch-560m-research

> ⚠️ **Research-only / non-commercial.** This variant of `gheim-ch-560m`
> was fine-tuned on a multi-source training mix that includes
> Babelscape WikiNeural (CC BY-NC-SA 4.0) and CoNLL-2003 (Reuters
> Corpus, research-only). The most restrictive upstream licence binds
> the released model: **the checkpoint may not be used for commercial
> purposes**. For commercial deployments use the sister checkpoint
> [`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m),
> which is trained only on the Apache-2.0-compatible
> [`gheim-ch-pii-212k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k)
> dataset.

A multilingual token-classification model for personally-identifiable
information (PII) detection across the four official Swiss languages
(de_CH, fr_CH, it_CH, rm) and English. Architecture and parameter
count are identical to the flagship `joelbarmettler/gheim-ch-560m`;
the difference is the **training mix**, which adds public external
NER / PII training data on top of the in-domain
`gheim-ch-pii-212k` corpus to lift cross-domain transfer (especially
on Swiss-news text and external person-NER benchmarks).

| | |
|---|---|
| Parameters | 560M |
| Languages | de_CH, fr_CH, it_CH, rm, en |
| Categories | account_number, private_address, private_date, private_email, private_person, private_phone, private_url, secret |
| Tag scheme | BIOES (33 classes) |
| Max sequence length | 512 |
| License | **CC BY-NC-SA 4.0** (with Reuters research-only restriction inherited from CoNLL-2003) |

## When to choose this variant

| If your use case is… | Use |
|---|---|
| Production deployment, customer data, commercial product | [`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m) (Apache 2.0) |
| Academic research, benchmarks, papers, evaluations | **this checkpoint** |
| Cross-domain redaction (Swiss-news articles, English news, multilingual Wikipedia) | **this checkpoint** (substantially stronger on those distributions) |
| In-domain redaction (Swiss court / parliament / web text) | either — numbers are essentially identical (see below) |

## Performance vs the commercial flagship

In-distribution (held-out test split of `gheim-ch-pii-212k`,
21,246 chunks) — the two checkpoints are within 0.2 pp on every
headline metric:

| Metric | `gheim-ch-560m` (Apache 2.0) | `gheim-ch-560m-research` (this) | Δ |
|---|---:|---:|---:|
| Test strict F1 | 0.910 | 0.912 | +0.002 |
| Test char F1 | 0.946 | 0.946 | 0.000 |
| Per-(lang × cat) char F1 | 0.940 | 0.940 | 0.000 |

On six external benchmarks (PER char F1). Three of the six —
`openpii-1m`, `WikiNeural`, `CoNLL-2003` — are present in this
variant's training mix, so those numbers are **in-distribution
generalisation**, not zero-shot cross-domain transfer. The
`swissner`, `open-pii-500k`, and `gretel_finance` rows are
zero-shot for both checkpoints and are the honest cross-domain
comparison.

| Benchmark | n | gheim-ch-560m (Apache 2.0) | gheim-ch-560m-research (this) | Δ |
|---|---:|---:|---:|---:|
| `ZurichNLP/swissner` (Swiss-news NER, **zero-shot for both**) | 800 | 0.702 | **0.903** | **+20.1 pp** |
| `ai4privacy/pii-masking-openpii-1m` (in-distribution for this variant) | 8,000 | 0.938 | **0.995** | +5.7 pp |
| `ai4privacy/open-pii-masking-500k` (**zero-shot for both**) | 8,000 | 0.933 | **0.982** | +4.9 pp |
| `gretelai/synthetic_pii_finance_multilingual` (**zero-shot for both**) | 4,800 | 0.624 | 0.627 | +0.3 pp |
| `Babelscape/wikineural` (in-distribution for this variant) | 8,000 | **0.808** | 0.795 | −1.3 pp |
| `tomaarsen/conll2003` (in-distribution for this variant) | 3,453 | **0.911** | 0.765 | −14.6 pp |

Headline finding: **on the zero-shot Swiss-news test (`swissner`),
this variant attains 0.903 PER char F1 vs the Apache-2.0 baseline's
0.702 — a 20 pp gain from broader training data even though
swissner itself is not seen at training time** (no swissner train
split exists; the gain comes from openpii + WikiNeural + CoNLL-2003
generalisation).

Two caveats worth understanding before using this variant:

1. **On `Babelscape/WikiNeural` and `tomaarsen/conll2003` — both
   present in this variant's training mix — the research variant
   actually regresses against the Apache-2.0 baseline.** The
   broader 8-category output schema produces non-PER false
   positives on news / Wikipedia text (where any IBAN-shaped or
   email-shaped string would be flagged) that the in-domain-only
   Apache baseline does not produce. PII-detector recall comes at
   the cost of NER-benchmark precision.
2. **On `gretel_finance` the two variants are statistically
   tied** (~0.62). None of the training mixes contain
   financial-document text, so this benchmark measures the
   architecture's structural-PII generalisation. Adding
   financial-domain data is a planned future iteration.

Per-language `swissner` PER char F1 (the headline cross-domain
finding):

| Language | Apache 2.0 | Research | Δ |
|---|---:|---:|---:|
| de | 0.539 | **0.931** | +39 pp |
| fr | 0.761 | **0.913** | +15 pp |
| it | 0.643 | **0.856** | +21 pp |
| rm | 0.409 | **0.873** | **+46 pp** |

The Romansh result is the most notable: no Romansh training data
is added (none of the external corpora include RM), but the model
gains 46 pp on Romansh swissner PER char F1 via cross-lingual
transfer in XLM-R's shared encoder body from the additional
de/fr/it/en supervision.

### Per-(language × category) char-level F1 on the in-domain test

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

## Usage

The serving surface is identical to the Apache-2.0 variant; the SDKs
and `transformers` / `transformers.js` pipelines accept either repo
ID. Swap the model ID:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

repo = "joelbarmettler/gheim-ch-560m-research"   # research variant
tok = AutoTokenizer.from_pretrained(repo)
mdl = AutoModelForTokenClassification.from_pretrained(repo)
ner = pipeline("token-classification", model=mdl, tokenizer=tok,
               aggregation_strategy="simple")
```

For the gheim SDKs:

```python
from gheim import Anonymizer
a = Anonymizer(detector_model="joelbarmettler/gheim-ch-560m-research")
```

## Training procedure

Same architecture (`FacebookAI/xlm-roberta-large`, 560M params),
same hyperparameters as the flagship checkpoint (AdamW, LR 5e-5,
cosine with 5% warmup, no LLRD, effective batch 128 on 2× RTX 4090
DDP, bf16, 3 epochs, max sequence length 512). Best checkpoint at
step 5,500 of 5,946 (epoch 2.77) by validation `overall_f1` 0.911.
Wall time ≈ 94 min train + 5 min eval.

The training data is the union of:

| Source | Train chunks | License | Role |
|---|---:|---|---|
| [`joelbarmettler/gheim-ch-pii-212k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k) (train split) | 170,001 | CC BY 4.0 | In-domain Swiss PII (court, parliament, web, Romansh) |
| `ai4privacy/pii-masking-openpii-1m` (train split, de/fr/it/en) | 40,000 | Apache 2.0 / CC-BY 4.0 | Multi-category PII supervision (chat / forms) |
| `Babelscape/wikineural` (train splits, de/fr/it/en) | 39,223 | **CC BY-NC-SA 4.0** | News / Wikipedia PER supervision |
| `tomaarsen/conll2003` (train split, en, PER-bearing) | 4,373 | **Research-only (Reuters)** | English news PER supervision |
| **Total** | **253,597** | (most restrictive binds) | |

Only chunks containing at least one in-schema PII span are kept from
the external sources, so the model receives positive supervision
without "implicit O" negatives on text the source dataset did not
annotate exhaustively.

The full builder is at
[`training/src/gheim_training/data/external_train.py`](https://github.com/joelbarmettlerUZH/gheim/blob/main/training/src/gheim_training/data/external_train.py)
and the train-mix HF DatasetDict builder is at
[`training/src/gheim_training/data/build_hf_multisource.py`](https://github.com/joelbarmettlerUZH/gheim/blob/main/training/src/gheim_training/data/build_hf_multisource.py).

## Limitations

1. **Non-commercial use only.** Inherits the most restrictive of its
   training-data licenses (CC BY-NC-SA 4.0 from WikiNeural; Reuters
   research-only from CoNLL-2003). If you need a commercial-friendly
   checkpoint, use [`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m)
   (Apache 2.0, trained only on in-domain Swiss text; cross-domain
   numbers are weaker — see the comparison table above).
2. **`private_address` test strict F1 is 0.84** (char F1 0.92).
   Boundary placement on multi-token addresses is the dominant
   error mode.
3. **Swiss German dialect (GSW) is not measured.** The fasttext
   detector used in data preparation labels GSW as standard German.
4. **`account_number` test strict F1 is 0.99** in the headline, but
   regex-shaped non-PII in numeric tables can still slip through.
   For production use, pair with the regex front-end documented in
   the `gheim` library (checksum validation: IBAN, AHV, VAT-CHE,
   Luhn).
5. **Re-identification is not in scope.** The model is intended for
   redaction; it does not return entity-linked identifiers.

## License

Composite, bounded by the most restrictive upstream:

- Model weights: **CC BY-NC-SA 4.0** with a Reuters research-only
  rider inherited from CoNLL-2003.
- Base architecture (`FacebookAI/xlm-roberta-large`): Apache 2.0
  (does not bind the fine-tune since training data overrides).
- In-domain training data
  ([`gheim-ch-pii-212k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k)):
  CC BY 4.0.
- `ai4privacy/openpii-1m` training data: Apache 2.0 / CC BY 4.0.
- `Babelscape/wikineural` training data: **CC BY-NC-SA 4.0**.
- `tomaarsen/conll2003` training data: **Reuters research-only**.

If your use case cannot accept research-only / non-commercial terms,
use the sibling checkpoint
[`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m).

## Citation

```bibtex
@misc{barmettler2026gheim_ch_560m_research,
  title  = {gheim-ch-560m-research: A multi-source-trained Swiss-PII detector with strong cross-domain transfer},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/joelbarmettler/gheim-ch-560m-research},
  note   = {Research-only variant of gheim-ch-560m; trained on gheim-ch-pii-212k + ai4privacy/openpii-1m + Babelscape/wikineural + tomaarsen/conll2003}
}
```

## Maintainer

Joel Barmettler · [jbarmettler@proton.me](mailto:jbarmettler@proton.me) · [joelbarmettler.xyz](https://joelbarmettler.xyz) · [github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim)
