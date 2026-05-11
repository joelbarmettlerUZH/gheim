<div align="center">

<p align="center">
  <img src="https://raw.githubusercontent.com/joelbarmettlerUZH/gheim/main/assets/logo.png" alt="gheim" width="420">
</p>

<p align="center"><strong>PII round-trip for LLM APIs. Anonymise before the request, de-anonymise the stream on the way back.</strong></p>

<p align="center">
  <a href="https://gheim.ch"><img src="https://img.shields.io/badge/demo-gheim.ch-black" alt="gheim.ch"></a>
  <a href="https://pypi.org/project/gheim/"><img src="https://img.shields.io/pypi/v/gheim?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/gheim"><img src="https://img.shields.io/npm/v/gheim?color=blue&label=npm" alt="npm"></a>
  <a href="https://huggingface.co/joelbarmettler/gheim-ch-560m"><img src="https://img.shields.io/badge/HF%20model-gheim--ch--560m-yellow" alt="HF model"></a>
  <a href="https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k"><img src="https://img.shields.io/badge/HF%20dataset-gheim--ch--pii--171k-yellow" alt="HF dataset"></a>
  <a href="https://github.com/joelbarmettlerUZH/gheim/pkgs/container/gheim-server"><img src="https://img.shields.io/badge/ghcr.io-gheim--server-blue" alt="ghcr"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0"></a>
  <a href="https://github.com/joelbarmettlerUZH/gheim"><img src="https://img.shields.io/github/stars/joelbarmettlerUZH/gheim?style=social" alt="GitHub stars"></a>
</p>

</div>

---

Swiss companies want to use GPT, Claude, and Gemini. Their lawyers do not.
Every prompt with a customer name, a Swiss address, or an IBAN is a compliance
problem the moment it crosses the Atlantic.

`gheim` solves it locally. A small on-device model finds the PII, swaps it for
stable sentinels, sends the redacted text to any LLM, and restores the
originals as the response streams back. The user sees `Joel`. OpenAI only ever
sees `<PERSON_1>`.

The round-trip is four steps: (1) detect PII spans on the input, (2) swap
each span for a stable sentinel like `<PERSON_1>` or `<ACCOUNT_1>`, (3)
send the redacted text to any LLM, (4) substitute the originals back into
the response on the way out, including across SSE token boundaries via a
bounded hold-back buffer.

## What ships

| | what | where | how |
|---|---|---|---|
| 🌐 | Live demo | [gheim.ch](https://gheim.ch) | runs the model in your browser |
| 🐍 | Python SDK | [`packages/gheim-py`](packages/gheim-py) · [PyPI](https://pypi.org/project/gheim/) | `uv add "gheim[openai]"` |
| 🟦 | JS / TS SDK | [`packages/gheim-js`](packages/gheim-js) · [npm](https://www.npmjs.com/package/gheim) | `npm install gheim openai` |
| 🐳 | Detection server | [`server`](server) | `docker pull ghcr.io/joelbarmettlerUZH/gheim-server` |
| 🤗 | Model | [`gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m) · [model card](MODEL_CARD.md) | 560M, Apache 2.0, ONNX int8 |
| 📚 | Dataset | [`gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k) · [dataset card](DATASET_CARD.md) | 171k chunks, 5 langs, CC BY 4.0 |

## Quick start

### Python · drop-in OpenAI client

```bash
uv add "gheim[openai]"
```

```python
from gheim.openai import OpenAI    # same constructor as openai.OpenAI

client = OpenAI()
r = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi, my name is Joel."}],
)
# r.choices[0].message.content contains "Joel".
# OpenAI only ever saw "<PERSON_1>".
```

Streaming, async (`AsyncOpenAI`), tool calls, and 9 other text-carrying
endpoints (`responses`, `embeddings`, `moderations`, `audio.*`, `images.*`)
are wrapped automatically. See the
[Python SDK README](packages/gheim-py/README.md) for the full surface.

### JavaScript / TypeScript

```bash
npm install gheim openai
```

```ts
import { OpenAI } from "gheim/openai";

const client = new OpenAI();
const r = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hi, my name is Joel." }],
});
```

Dual ESM + CJS, full `.d.ts`, Node 18+ and Bun 1.1+. Browser-side via
`@huggingface/transformers` (WebGPU). See the
[JS SDK README](packages/gheim-js/README.md).

### Self-host the detection server

```bash
docker run -p 8080:8080 -e GHEIM_API_KEYS=your-key \
  ghcr.io/joelbarmettlerUZH/gheim-server:latest
```

```python
from gheim import RemoteDetector
det = RemoteDetector(base_url="http://your-host:8080", api_key="your-key")
```

Model weights are baked into the image, `HF_HUB_OFFLINE=1` at runtime;
works in air-gapped environments. See the [server README](server/README.md).

## The Swiss-tuned model

`joelbarmettler/gheim-ch-560m` is the default detector when running locally:
a fine-tune of `xlm-roberta-large` on 171k Swiss-domain chunks across the
four official Swiss languages (de_CH, fr_CH, it_CH, rm) plus English.

| | strict-span F1 | char F1 |
|---|---:|---:|
| `gheim-ch-560m` (this) | **0.916** | **0.958** |
| `openai/privacy-filter` (1.4B MoE, zero-shot) | 0.443 | 0.610 |
| Microsoft Presidio Analyzer | 0.434 | 0.562 |
| `Davlan/xlm-roberta-base-ner-hrl` (PER cell) | n/a | 0.728 PER |
| spaCy `de/fr/it_core_news_lg` (PER cell) | n/a | 0.621 PER |

All seven contestants scored on the same 15,861-chunk held-out test split.
Full comparison matrix and methodology in [`MODEL_CARD.md`](MODEL_CARD.md);
raw numbers in [`eval/positioning_matrix.json`](eval/positioning_matrix.json).

The model also ships as `onnx/model_quantized.onnx` (int8, 552 MB) so the
demo at [gheim.ch](https://gheim.ch) runs entirely in the browser via
`@huggingface/transformers`.

## The dataset

`joelbarmettler/gheim-ch-pii-171k` is the training corpus released with the
model. 171,336 chunks across de_CH / fr_CH / it_CH / rm / en, 8 PII
categories, BIOES-tagged. Annotations are machine-generated (LLM labelling
plus checksum-validated regex augmentation, with synthetic gap-fill for
sparse cells); a 4-way subagent labelling on a 176-chunk sample yielded
F1 ≈ 0.66 against majority vote, an upper bound on label noise. See
[`DATASET_CARD.md`](DATASET_CARD.md) for the curation pipeline, per-cell
distribution, and reuse instructions.

## License

Apache 2.0; see [LICENSE](LICENSE). The published model weights and the
training dataset are licensed Apache 2.0 and CC BY 4.0 respectively;
attribution to the upstream `swiss-ai/apertus-pretrain-*` corpora is
required when reusing the dataset.

## Citation

```bibtex
@software{barmettler2026gheim,
  author = {Barmettler, Joel},
  title  = {gheim: PII Round-Trip for LLM APIs},
  year   = {2026},
  url    = {https://github.com/joelbarmettlerUZH/gheim}
}
```
