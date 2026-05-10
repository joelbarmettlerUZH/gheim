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
  <img src="https://github.com/joelbarmettlerUZH/gheim/blob/main/assets/logo.png" alt="gheim" width="360">
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

> **Full report:** the curation pipeline, training procedure, comparison
> against seven other PII / NER systems, cross-domain results on four
> external benchmarks, methodology validation, and an extended related-work
> section are documented in
> [`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
> (arXiv preprint forthcoming). This card is the deployment-facing summary.

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

Strict-span F1 (`seqeval`) on the held-out test split of
[`joelbarmettler/gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k)
(15,861 chunks, document-isolated from the training split for the real-text
portion). The test set was scored once.

| Metric | Test | Validation |
|---|---:|---:|
| F1 | 0.916 | 0.918 |
| Precision | 0.907 | 0.908 |
| Recall | 0.926 | 0.926 |

Per-category and per-language breakdowns:

| Category | F1 | n_gold | | Language | F1 | n_chunks |
|---|---:|---:|---|---|---:|---:|
| `secret` | 0.996 | 1,104 | | en | 0.954 | 893 |
| `private_email` | 0.987 | 1,712 | | it_ch | 0.942 | 4,169 |
| `private_phone` | 0.986 | 3,329 | | de_ch | 0.911 | 4,753 |
| `private_url` | 0.947 | 3,991 | | fr_ch | 0.906 | 4,705 |
| `private_date` | 0.922 | 9,678 | | rm | 0.853 | 1,341 |
| `private_person` | 0.915 | 14,726 | | | | |
| `private_address` | 0.782 | 3,507 | | | | |
| `account_number` | 0.669 | 502 | | | | |

For the full comparison against seven other open PII / NER systems on the
same Swiss test set, cross-domain evaluation on four external benchmarks
(AI4Privacy openpii-1m, ZurichNLP swissner, CoNLL-2003, Babelscape
WikiNeural), and the methodology-validation reproductions of each
baseline's published numbers, see
[`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
§4 and the machine-readable matrix at
[`eval/positioning_matrix.json`](https://github.com/joelbarmettlerUZH/gheim/blob/main/eval/positioning_matrix.json).

## Deployment formats

The model is published in two formats:

- `model.safetensors` (root): fp32 PyTorch checkpoint, 2.2 GB, intended
  for server-side inference via `transformers`.
- `onnx/model_quantized.onnx`: int8 dynamic-quantised ONNX export, 552 MB,
  intended for in-browser inference via
  [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers)
  on WebGPU or WebAssembly. Selected with `dtype: "q8"`.

| Format | Size | Test F1 | Δ vs fp32 |
|---|---:|---:|---:|
| PyTorch fp32 | 2.2 GB | 0.916 | (baseline) |
| ONNX int8 (dynamic) | 552 MB | 0.909 | -0.7 pp |

The int8 export loses 0.7 pp overall F1 relative to the PyTorch checkpoint;
the loss is concentrated in `private_address` (-2.1 pp) and `account_number`
(-5.3 pp), the two categories that were already weakest under fp32. All
five languages are affected uniformly within ±0.001 pp of the overall
drop. Per-category and per-language quantisation deltas are tabulated in
[`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
§3.3.

## Training procedure

Selected from a controlled bake-off against `ZurichNLP/swissbert` (270M
dense), each model receiving an identical 5 × 3 sweep over (learning
rate, layer-wise LR decay) at 1 epoch. The winning configuration per
base model was trained for 3 full epochs and selected by best validation
F1. `xlm-roberta-large` won the bake-off (val F1 0.918 vs swissbert's
0.910). Selected configuration: AdamW, LR 5e-5 cosine with 5% warmup,
no LLRD, effective batch 128 (per-device 64 × 2 GPUs DDP), bf16, 3
epochs, max sequence length 512. Best checkpoint at epoch 2.50 by
validation `overall_f1`. Wall time ≈ 52 min train + 4 min eval on
2 × RTX 4090. Full procedure including the hyperparameter sweep results
is in
[`paper/paper.pdf`](https://github.com/joelbarmettlerUZH/gheim/blob/main/paper/paper.pdf)
§3.

The training data was the train split of `gheim-ch-pii-171k` (139,641
chunks) plus an English-anchor and Swiss-region email rescue slice from
`ai4privacy/pii-masking-openpii-1m` (≈ 14,000 chunks); validation and
test contain no AI4Privacy data.

## Limitations

1. **Recall-oriented labelling policy.** The model inherits the dataset's
   policy of flagging publicly-listed institutional contact information.
   Applications needing stricter precision should apply downstream
   filtering or a private-vs-public-entity post-classifier.
2. **`private_address` test F1 is 0.78.** Boundary placement on
   multi-token addresses is the dominant error mode.
3. **`account_number` test F1 is 0.67.** For production use, pair the
   model with the regex front-end documented in the `gheim` library,
   which applies checksum validation (IBAN, AHV, VAT-CHE, Luhn).
4. **Romansh test F1 is 0.85**, the weakest of the five languages. The RM
   training material is dominated by a single literary/journalistic
   register; performance on dialectal or technical RM text is unmeasured.
5. **Swiss German dialect (GSW) is not measured.** The fasttext detector
   used in data preparation labels GSW as standard German.
6. **Re-identification is not in scope.** The model is intended for
   redaction; it does not return entity-linked identifiers.

## License

Apache 2.0, inherited from the base model `FacebookAI/xlm-roberta-large`.
The training data
([`joelbarmettler/gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k))
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
@misc{barmettler2026gheim_ch_pii,
  title  = {gheim-ch-pii-171k: A Swiss-grounded PII NER dataset with synthetic gap-fill},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k}
}
```

## Maintainer

Joel Barmettler · [jbarmettler@proton.me](mailto:jbarmettler@proton.me) · [joelbarmettler.xyz](https://joelbarmettler.xyz) · [github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim)

Source code, issue tracker, and the wider gheim ecosystem (Python and Node
libraries, redaction server, composite detector) are at
[github.com/joelbarmettlerUZH/gheim](https://github.com/joelbarmettlerUZH/gheim).
