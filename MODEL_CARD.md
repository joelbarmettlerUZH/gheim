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
  - pii-detection
  - ner
  - named-entity-recognition
  - swiss
  - swiss-german
  - de-CH
  - fr-CH
  - it-CH
  - rm
  - privacy
  - gdpr
  - onnx
  - quantized
base_model: FacebookAI/xlm-roberta-large
base_model_relation: finetune
datasets:
  - joelbarmettler/gheim-ch-pii-212k
metrics:
  - f1
  - precision
  - recall
model-index:
  - name: gheim-ch-560m
    results:
      # ============================================================
      # In-distribution headline (held-out test split)
      # ============================================================
      - task:
          type: token-classification
          name: PII NER (in-distribution, held-out test)
        dataset:
          type: joelbarmettler/gheim-ch-pii-212k
          name: gheim-ch-pii-212k
          split: test
        metrics:
          - type: f1
            value: 0.9105
            name: Strict-span F1 (seqeval)
          - type: f1
            value: 0.9461
            name: Char-level F1 (label-aware)
          - type: precision
            value: 0.8904
            name: Strict-span precision
          - type: recall
            value: 0.9315
            name: Strict-span recall
      # ============================================================
      # Cross-domain (zero-shot) — six external benchmarks
      # ============================================================
      - task:
          type: token-classification
          name: PII NER (zero-shot, Swiss-news)
        dataset:
          type: ZurichNLP/swissner
          name: ZurichNLP/swissner
          split: test
          args:
            evaluation: zero_shot
        metrics:
          - type: f1
            value: 0.702
            name: PER char F1 (overall, zero-shot)
          - type: f1
            value: 0.539
            name: PER char F1 (de, zero-shot)
          - type: f1
            value: 0.761
            name: PER char F1 (fr, zero-shot)
          - type: f1
            value: 0.643
            name: PER char F1 (it, zero-shot)
          - type: f1
            value: 0.409
            name: PER char F1 (rm, zero-shot)
      - task:
          type: token-classification
          name: PII NER (zero-shot)
        dataset:
          type: ai4privacy/pii-masking-openpii-1m
          name: ai4privacy/pii-masking-openpii-1m
          split: validation
          args:
            evaluation: zero_shot
            sample_per_lang: 2000
        metrics:
          - type: f1
            value: 0.938
            name: PER char F1 (zero-shot)
      - task:
          type: token-classification
          name: PII NER (zero-shot)
        dataset:
          type: ai4privacy/open-pii-masking-500k-ai4privacy
          name: ai4privacy/open-pii-masking-500k-ai4privacy
          split: validation
          args:
            evaluation: zero_shot
            sample_per_lang: 2000
        metrics:
          - type: f1
            value: 0.933
            name: PER char F1 (zero-shot)
      - task:
          type: token-classification
          name: PII NER (zero-shot, financial documents)
        dataset:
          type: gretelai/synthetic_pii_finance_multilingual
          name: gretelai/synthetic_pii_finance_multilingual
          split: test
          args:
            evaluation: zero_shot
            sample_per_lang: 1000
        metrics:
          - type: f1
            value: 0.624
            name: PER char F1 (zero-shot)
      - task:
          type: token-classification
          name: NER PER cell (zero-shot)
        dataset:
          type: Babelscape/wikineural
          name: Babelscape/wikineural
          split: test
          args:
            evaluation: zero_shot
            sample_per_lang: 2000
        metrics:
          - type: f1
            value: 0.808
            name: PER char F1 (zero-shot)
      - task:
          type: token-classification
          name: NER PER cell (zero-shot)
        dataset:
          type: tomaarsen/conll2003
          name: tomaarsen/conll2003
          split: test
          args:
            evaluation: zero_shot
        metrics:
          - type: f1
            value: 0.911
            name: PER char F1 (zero-shot)
      # ============================================================
      # ONNX int8 deployment delta (browser default)
      # ============================================================
      - task:
          type: token-classification
          name: PII NER (ONNX int8 dynamic quantisation)
        dataset:
          type: joelbarmettler/gheim-ch-pii-212k
          name: gheim-ch-pii-212k
          split: test
          args:
            format: onnx_int8_dynamic
            file_name: onnx/model_quantized.onnx
        metrics:
          - type: f1
            value: 0.9044
            name: Strict-span F1 (ONNX int8; delta vs fp32 -0.0061)
          - type: f1
            value: 0.9448
            name: Char-level F1 (ONNX int8; delta vs fp32 -0.0013)
---

<p align="center">
  <img src="https://raw.githubusercontent.com/joelbarmettlerUZH/gheim/main/assets/logo.png" alt="gheim" width="360">
</p>

# gheim-ch-560m

A multilingual token-classification model for personally-identifiable information
(PII) detection across the four official Swiss languages (de_CH, fr_CH, it_CH, rm)
and English. The model is a fine-tune of
[`FacebookAI/xlm-roberta-large`](https://huggingface.co/FacebookAI/xlm-roberta-large)
on [`joelbarmettler/gheim-ch-pii-212k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k).
Output schema is a 33-class BIOES tag set (8 PII categories plus the outside class)
aligned with the categorical naming used by `openai/privacy-filter`.

> **Two-variant release.** This checkpoint is the **Apache-2.0
> commercial flagship**, trained only on the in-domain
> `gheim-ch-pii-212k` so that the entire pipeline is end-to-end
> reproducible from the gheim repository alone. A sibling checkpoint
> [`joelbarmettler/gheim-ch-560m-research`](https://huggingface.co/joelbarmettler/gheim-ch-560m-research)
> is trained on the same architecture + the same in-domain data
> **plus** external public NER / PII corpora (ai4privacy/openpii-1m,
> Babelscape/wikineural, tomaarsen/conll2003). It attains the same
> in-distribution numbers but substantially stronger cross-domain
> transfer on Swiss news (`swissner` PER char F1: 0.70 → 0.90) and
> external person-NER benchmarks. **Non-commercial / research-only
> licence** (CC BY-NC-SA 4.0 with Reuters research-only restriction).
> See the [research card](https://huggingface.co/joelbarmettler/gheim-ch-560m-research)
> for the full comparison table.

| | |
|---|---|
| Parameters | 560M |
| Languages | de_CH, fr_CH, it_CH, rm, en |
| Categories | account_number, private_address, private_date, private_email, private_person, private_phone, private_url, secret |
| Tag scheme | BIOES (33 classes) |
| Max sequence length | 512 |
| License | Apache 2.0 |

> **Full report:** the curation pipeline, training procedure, comparison
> against seven other PII / NER systems, cross-domain results on four
> external benchmarks, methodology validation, and an extended related-work
> section are documented in
> [`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf).
> This card is the deployment-facing summary.

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

### Recommended: gheim SDKs (round-trip with sentinel restoration)

For the typical use case — anonymise text, send to an LLM, restore the
originals on the way back — install the [`gheim`](https://pypi.org/project/gheim/)
Python or [`gheim`](https://www.npmjs.com/package/gheim) npm package.
This model is the default detector in both, and the wrappers handle
sentinel allocation, streaming-aware decode, multi-turn coherent
sessions, and a drop-in `OpenAI` client.

```bash
pip install "gheim[local,openai]"        # Python
npm install gheim openai @huggingface/transformers   # JS / TS
```

```python
# Python — drop-in OpenAI client. Defaults to gheim-ch-560m.
from gheim.openai import OpenAI

client = OpenAI()  # accepts the same kwargs as openai.OpenAI
r = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user",
               "content": "Hi, my name is Joel. My phone is +41 44 268 12 34."}],
)
# r.choices[0].message.content has the original PII restored.
# OpenAI only ever saw "<PERSON_1>" and "<PHONE_1>".
```

```ts
// JS / TS — same idea.
import { OpenAI } from "gheim/openai";

const client = new OpenAI();  // accepts the same opts as openai's OpenAI
const r = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user",
               content: "Hi, my name is Joel. My phone is +41 44 268 12 34." }],
});
```

Streaming, async, tool calls, and 9 other text-carrying endpoints
(`responses`, `embeddings`, `moderations`, `audio.*`, `images.*`) are
wrapped automatically. Full surface in the package READMEs:
[Python](https://github.com/joelbarmettlerUZH/gheim/blob/main/packages/gheim-py/README.md)
·
[JS](https://github.com/joelbarmettlerUZH/gheim/blob/main/packages/gheim-js/README.md).

### Alternative: raw transformers / transformers.js

If you only need a token classifier (no sentinel round-trip), use the
HuggingFace pipelines directly.

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

```ts
// Node / Bun / browser via @huggingface/transformers (transformers.js).
// Recommended: dtype: "fp16" when WebGPU is available (1.1 GB, byte-
// equivalent to fp32 on the forensic probe), "q8" for low-RAM fallback
// (557 MB, but see the JS-runtime caveat under Deployment formats below
// — int8 degrades sharply on Swiss edge cases via onnxruntime-web/WASM).
// The `gheim` SDK picks this automatically via its `dtype: "auto"` default.
import { pipeline } from "@huggingface/transformers";

const ner = await pipeline("token-classification",
  "joelbarmettler/gheim-ch-560m",
  { aggregation_strategy: "simple", dtype: "fp16", device: "webgpu" });
const out = await ner("Email me at alice@example.ch, phone +41 44 268 12 34.");
```

## Performance

Strict-span F1 (`seqeval`) on the held-out test split of
[`joelbarmettler/gheim-ch-pii-212k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k)
(21,246 chunks, document-isolated from the training split for the
real-text portion). The test set was scored once.

| Metric | Test | Validation |
|---|---:|---:|
| F1 | 0.910 | 0.910 |
| Precision | 0.890 | 0.891 |
| Recall | 0.932 | 0.930 |

Per-language × per-category char-level F1 on the same test split. Body
cells are char-level F1 for each (language, category) pair; the
right-most column gives the per-category average over languages
(gold-weighted), and the bottom row gives the per-language average
over categories. The bottom-right cell is the overall char F1.

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

### Cross-domain (six external benchmarks)

PER character-level F1 on the six external benchmarks below. This
`gheim-ch-560m` checkpoint is run zero-shot on every row; the
research variant is included for context but is zero-shot only on
the gretel + swissner + open-pii-500k rows — three of the six
(`ai4privacy/openpii-1m`, `Babelscape/WikiNeural`,
`tomaarsen/conll2003`) are present in its training mix (see the
[research model card](https://huggingface.co/joelbarmettler/gheim-ch-560m-research)
for the full discussion). Numbers on those three rows for the
research variant are therefore in-distribution, not transfer.

| Benchmark | n | License | gheim-ch-560m (zero-shot) | gheim-ch-560m-research |
|---|---:|---|---:|---:|
| `ZurichNLP/swissner` (Swiss-news NER, zero-shot for both) | 800 | CC BY 4.0 | 0.702 | **0.903** |
| `ai4privacy/pii-masking-openpii-1m` (research trained on train split) | 8,000 | Apache 2.0 / CC BY 4.0 | 0.938 | **0.995**† |
| `ai4privacy/open-pii-masking-500k` (zero-shot for both) | 8,000 | CC BY 4.0 | 0.933 | **0.982** |
| `gretelai/synthetic_pii_finance_multilingual` (zero-shot for both) | 4,800 | Apache 2.0 | 0.624 | 0.627 |
| Babelscape `WikiNeural` (research trained on train split) | 8,000 | CC BY-NC-SA 4.0 | **0.808** | 0.795† |
| `tomaarsen/conll2003` (research trained on train split, PER only) | 3,453 | research-only | **0.911** | 0.765† |

†Research-variant numbers on these three rows reflect in-distribution generalisation, not zero-shot cross-domain transfer (research training mix includes the train splits of these three datasets). Note that on `Babelscape/WikiNeural` and `tomaarsen/conll2003` the research variant actually **regresses** relative to the Apache-2.0 baseline despite training on their train splits — the broader 8-category output schema produces non-PER false positives on news / Wikipedia text that the in-domain-only baseline doesn't make.

Per-language `ZurichNLP/swissner` PER char F1 (the headline
cross-domain test for Swiss-market deployment, zero-shot for both
checkpoints):

| Language | gheim-ch-560m | gheim-ch-560m-research |
|---|---:|---:|
| de | 0.539 | **0.931** |
| fr | 0.761 | **0.913** |
| it | 0.643 | **0.856** |
| rm | 0.409 | **0.873** |

For Swiss-news redaction at scale, the research variant is
substantially stronger — especially on Romansh (+46 pp). For
in-domain Swiss court / parliament / web text and structured PII
(IBAN/AHV/email/phone), the two variants are essentially identical.

For the full comparison against eight other open PII / NER systems on
the same Swiss test set, the methodology-validation reproductions of
each baseline's published numbers, and the full three-variant
training-mix experiment that motivates the two-checkpoint release,
see [`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
and the machine-readable matrix at
[`eval/positioning_matrix.json`](https://github.com/joelbarmettlerUZH/gheim/blob/main/eval/positioning_matrix.json).

## Deployment formats

The model is published in four formats:

- `model.safetensors` (root): fp32 PyTorch checkpoint, 2.2 GB, intended
  for server-side inference via `transformers`.
- `onnx/model.onnx` (+ `onnx/model.onnx_data`): fp32 ONNX export, 2.2 GB,
  intended for server-side ONNX Runtime / GPU deployment.
- `onnx/model_fp16.onnx` (+ `onnx/model_fp16.onnx_data`): fp16 ONNX
  export, 1.1 GB, recommended for browser/Node consumers via
  [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers)
  when WebGPU is available. Byte-equivalent to fp32 on the forensic
  probe. Selected with `dtype: "fp16"` (or `dtype: "auto"` in the
  `gheim` SDK, which picks this on WebGPU).
- `onnx/model_quantized.onnx`: int8 dynamic-quantised ONNX export, 557 MB,
  intended for low-RAM mobile fallback. Selected with `dtype: "q8"`
  (or `dtype: "auto"` when WebGPU is not available). **JS-runtime
  caveat:** this file is essentially fp32-equivalent under Python
  `onnxruntime` (91.0% forensic-probe perfect-rate vs 91.5% for fp32 /
  fp16) but degrades to 73.1% perfect-rate under `transformers.js` /
  `onnxruntime-web` — a JS runtime divergence, not a quantisation
  issue. Common-word surnames like "Bach" become false positives and
  some commercial-register person names go undetected. The published
  Python SDK is unaffected. See
  [`eval/q8_quality_report.md`](https://github.com/joelbarmettlerUZH/gheim/blob/main/eval/q8_quality_report.md)
  for the diagnostic and per-language / per-category breakdown.

| Format | Size | Test strict F1 | Test char F1 | Δ strict vs fp32 | Δ char vs fp32 |
|---|---:|---:|---:|---:|---:|
| PyTorch fp32 | 2.2 GB | 0.9105 | 0.9461 | (baseline) | (baseline) |
| ONNX fp32 | 2.2 GB | 0.9105 | 0.9461 | 0.000 | 0.000 |
| ONNX fp16 | 1.1 GB | ≈0.9105 | ≈0.9461 | 0.000* | 0.000* |
| ONNX int8 (dynamic) | 557 MB | 0.9044 | 0.9448 | −0.0061 | −0.0013 |

*fp16 measured equivalent to fp32 on the 212-case forensic probe and on a
~150k-token logit-divergence sample (mean abs logit diff 0.0011, 100%
per-token argmax match). Full test-set numbers not separately tabulated.

Per-category int8 vs fp32 char F1 deltas (Python / `onnxruntime` CPU):
`account_number` 0.00, `private_address` −0.003, `private_date` −0.002,
`private_email` 0.00, `private_person` −0.001, `private_phone` 0.00,
`private_url` 0.00, `secret` 0.00. The int8 quantisation cost is
concentrated almost entirely on the `private_address` cell; structured-PII
categories are unaffected.

The fp16 export above is produced by `training/eval/quantize_onnx.py::quantize_fp16`
with a post-conversion fixup that inserts `fp16->fp32` promotion Casts at
type-match-op boundaries (Div/Mul/MatMul/LayerNorm) and strips the
redundant trailing classifier Cast. The naked `onnxconverter_common`
output was unloadable in onnxruntime because XLM-R's attention block
casts to fp32 around its `sqrt(d_k)` for numerical stability and the
converter doesn't propagate that mix.

## Training procedure

Selected from a controlled bake-off against `ZurichNLP/swissbert` (270M
dense), each model receiving an identical 5 × 3 sweep over (learning
rate, layer-wise LR decay) at 1 epoch. The winning configuration per
base model was trained for 3 full epochs and selected by best validation
F1. `xlm-roberta-large` won the bake-off (val F1 0.918 vs swissbert's
0.910). Selected configuration: AdamW, LR 5e-5 cosine with 5% warmup,
no LLRD, effective batch 128 (per-device 64 × 2 GPUs DDP), bf16, 3
epochs, max sequence length 512. Best checkpoint at step
3,500 of 3,987 (epoch 2.63) by validation `overall_f1` 0.910.
Wall time ≈ 66 min train + 5 min eval on 2 × RTX 4090. Full
procedure including the hyperparameter sweep results is in
[`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
§3.

The training data is the train split of
[`joelbarmettler/gheim-ch-pii-212k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k)
(170,001 chunks), used end-to-end with no external augmentation. The
validation and test splits are held out from the same dataset.

## Limitations

1. **Recall-oriented labelling policy.** The model inherits the dataset's
   policy of flagging publicly-listed institutional contact information.
   Applications needing stricter precision should apply downstream
   filtering or a private-vs-public-entity post-classifier.
2. **`private_address` test strict F1 is 0.84** (char F1 0.92).
   Boundary placement on multi-token addresses is the dominant error
   mode.
3. **`account_number` test strict F1 is 0.99** in the headline,
   but a small fraction of regex-shaped non-PII (numeric tables in
   court documents and parliamentary statistics) still slips
   through. For production use, pair the model with the regex
   front-end documented in the `gheim` library, which applies
   checksum validation (IBAN, AHV, VAT-CHE, Luhn).
4. **Romansh test strict F1 is 0.89** (char F1 0.93), the weakest
   of the five languages. The RM training material is dominated by
   a single literary/journalistic register; performance on
   dialectal or technical RM text is unmeasured.
5. **Cross-domain Swiss-news transfer is weaker than the
   in-distribution headline.** On `ZurichNLP/swissner` the model
   scores 0.70 PER char F1 overall (per-lang: de 0.54 / fr 0.76 /
   it 0.64 / rm 0.41). The released checkpoint trains only on the
   in-domain `gheim-ch-pii-212k` corpus so the published pipeline
   is end-to-end reproducible from this repository alone; a
   multi-source training mix that includes external public NER
   corpora is being investigated as the next iteration.
6. **Swiss German dialect (GSW) is not measured.** The fasttext
   detector used in data preparation labels GSW as standard German.
7. **Lone first-names in greetings can be missed.** Bare first names
   in greeting positions (e.g. "Hallo Marius,") are a known coverage
   gap; pair with a deterministic greeting-pattern regex for
   chat-style inputs where this matters.
8. **Re-identification is not in scope.** The model is intended for
   redaction; it does not return entity-linked identifiers.

## Note on the predecessor release

An earlier checkpoint of the same architecture was previously
published at this URL, trained against an earlier dataset revision.
That release surfaced several pipeline issues after publication —
double-labelled overlapping spans, missing Geonames-CH gazetteer
demotion of municipality names mis-tagged as people, and a forked
synthetic generator that disagreed with the templated pipeline on
output schema. The earlier checkpoint and dataset have been retired
and moved to private archive repositories
(`joelbarmettler/gheim-ch-560m-archive` and
`joelbarmettler/gheim-ch-pii-171k-archive`); the current
`joelbarmettler/gheim-ch-560m` checkpoint is a fresh fine-tune on
the reproducible
[`gheim-ch-pii-212k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k).
The architecture and parameter count are unchanged; the training
data and weights are new.

## License

Apache 2.0, inherited from the base model `FacebookAI/xlm-roberta-large`.
The training data
([`joelbarmettler/gheim-ch-pii-212k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k))
is released under CC BY 4.0; attribution to its upstream corpora (the
`swiss-ai/apertus-pretrain-*` datasets) is required when reusing the data.

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
@misc{barmettler2026gheim_ch_pii_212k,
  title  = {gheim-ch-pii-212k: A Swiss-grounded PII NER dataset with multi-LLM consensus labels and synthetic gap-fill},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k}
}
```

## Maintainer

Joel Barmettler · [jbarmettler@proton.me](mailto:jbarmettler@proton.me) · [joelbarmettler.xyz](https://joelbarmettler.xyz) · [github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim)

Source code, issue tracker, and the wider gheim ecosystem (Python and Node
libraries, redaction server, composite detector) are at
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim).
