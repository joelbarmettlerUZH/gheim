<div align="center">

<p align="center">
  <img src="https://raw.githubusercontent.com/joelbarmettlerUZH/gheim/main/assets/logo.png" alt="gheim" width="360">
</p>

<p align="center"><strong>gheim (JavaScript / TypeScript). PII round-trip for LLM APIs.</strong></p>

<p align="center">
  <a href="https://www.npmjs.com/package/gheim"><img src="https://img.shields.io/npm/v/gheim?color=blue&label=npm" alt="npm"></a>
  <a href="https://nodejs.org/"><img src="https://img.shields.io/badge/node-18%2B-blue" alt="Node 18+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0"></a>
</p>

</div>

Detect PII in text, substitute it with stable sentinels (`<PERSON_1>`,
`<EMAIL_2>`, ...), send the redacted text to any LLM, and restore the
originals on the way back, including in streamed responses. The package
is framework-agnostic and ships a drop-in `openai` client wrapper for
zero-effort integration.

See the [monorepo README](https://github.com/joelbarmettlerUZH/gheim) for
the cross-language overview and architecture.

## Install

```bash
npm install gheim                            # or: bun add gheim
npm install gheim openai                     # add the drop-in OpenAI client
npm install gheim @huggingface/transformers  # add on-device detection
```

`openai` and `@huggingface/transformers` are optional peer dependencies.
The core package itself has no runtime dependencies.

## Model choice

`LocalDetector` loads a token-classification model via
[`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers)
(transformers.js). The package's default model is
[`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m)
— a 560M xlm-roberta-large fine-tune optimised for Swiss-market PII
(strict-span F1 0.916 on Swiss text, see
[MODEL_CARD.md](https://github.com/joelbarmettlerUZH/gheim/blob/main/MODEL_CARD.md)).
Any HuggingFace token-classification model that emits the same 33-class
BIOES schema can be substituted via `model:`.

| Model | Best for | Parameters | Notes |
|---|---|---:|---|
| [`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m) **(default)** | Swiss-market text (de_CH, fr_CH, it_CH, rm, en) with CH-format account numbers (IBAN, AHV, VAT-CHE) | 560M | Apache 2.0. Test F1 0.916. Hub repo ships `onnx/model_quantized.onnx` only — load with `dtype: "q8"`. |
| [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) | English-first or general use, long-context (up to 128k tokens) | 1.4B (50M active, MoE) | Apache 2.0. Wider language coverage, larger weights. |

```ts
import { LocalDetector } from "gheim";

// Default — Swiss-tuned, q8 ONNX. `device: "auto"` probes WebGPU in
// the browser and falls back to WASM; in Node it's WASM directly.
const det = new LocalDetector({ dtype: "q8" });

// Alternative for English or general use:
const detEn = new LocalDetector({ model: "openai/privacy-filter" });
```

## Drop-in OpenAI client

```ts
import { OpenAI } from "gheim/openai";

// Same constructor shape as `openai`. `apiKey`, `baseURL`, etc. work
// at the top level (forwarded to the inner client). gheim-specific
// keys: `gheimDetector`, `gheimStrict`, `openaiClient`, `clientOptions`.
const client = new OpenAI();
const r = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hi, my name is Joel" }],
});
// r.choices[0].message.content contains "Joel".
// OpenAI only ever saw "<PERSON_1>".
```

Custom endpoint or key (e.g. OpenRouter, local vLLM):

```ts
import { OpenAI } from "gheim/openai";

const client = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
});
```

Streaming:

```ts
const stream = await client.chat.completions.create({ ..., stream: true });
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta.content ?? "");
}
```

Per-call overrides:

```ts
import { Session } from "gheim";

const session = new Session();  // reuse across calls
const r = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [...],
  gheimSession: session,    // or gheimDetector
});
```

All non-chat OpenAI resources are on `client.raw` once `await client.ready()` resolves.

## Framework-agnostic

```ts
import { Session, LocalDetector, anonymizeText, deanonymizeText } from "gheim";

const session = new Session();
(session as any).detector = new LocalDetector({ dtype: "q8" });
//                                 ^ defaults to joelbarmettler/gheim-ch-560m

const clean = await anonymizeText("Hi, my name is Joel", session);
// ... call any LLM with clean ...
const final = deanonymizeText(responseText, session);
```

Streaming deanonymizer for any text-chunk source:

```ts
import { deanonymizeStream } from "gheim";

for await (const chunk of deanonymizeStream(myChunkIterator, session)) {
  process.stdout.write(chunk);
}
```

OpenAI-typed helpers when you want manual control:

```ts
import { anonymizeOpenAIMessages, deanonymizeOpenAIStream } from "gheim/openai";
```

## Wrapped endpoints

The drop-in `OpenAI` client automatically protects every text-carrying endpoint:
`chat.completions`, `responses`, `completions` (legacy), `embeddings`,
`moderations`, `audio.speech`, `audio.transcriptions`, `audio.translations`,
`images.generate`, `images.edit`. Tool-call arguments and SSE delta chunks are
restored on the way back. See the
[monorepo README](https://github.com/joelbarmettlerUZH/gheim) for the full
coverage matrix and the embeddings caveat.

### Strict mode

`gheimStrict: true` (default) throws `GheimStrictError` if you call an
unwrapped endpoint (`beta`, `batches`, `files`, `uploads`, `fineTuning`,
`vectorStores`). The error names `client.raw.<path>` as the escape hatch.

```ts
const client = new OpenAI({ gheimStrict: false }); // downgrade to one-time warnings
await client.ready();
client.raw.beta.assistants.create(...);  // always works regardless of strict mode
```

## Detectors

```ts
import { LocalDetector, RemoteDetector, defaultDetector } from "gheim";

// Local: via @huggingface/transformers, on Node, Bun, and browsers.
// `device: "auto"` (default) probes navigator.gpu in the browser and
// falls back to WASM if WebGPU isn't usable; in Node it's WASM only.
const local = new LocalDetector({
  device: "auto",
  dtype: "q8",
  onProgress: (e) => {
    if (e.fraction != null) console.log(`loading: ${(e.fraction * 100) | 0}%`);
  },
});
await local.load();
console.log("backend in use:", local.actualDevice);  // "webgpu" or "wasm"

// Remote: against your own gheim-server or api.gheim.ch.
const remote = new RemoteDetector({
  baseUrl: "http://your-host:8080",
  apiKey: "...",
});

// Picks remote when GHEIM_API_KEY is set in the environment, else local.
const auto = defaultDetector();
```

## Runtime support

Tested on Node 18+, Bun 1.1+, and modern browsers (Chrome 113+, Edge,
Safari 17+, Firefox via flag). Browser builds use WebGPU when available
and fall back to WebAssembly automatically.

For Node servers, installing `onnxruntime-node` alongside
`@huggingface/transformers` enables the native ONNX backend, which is
typically 5-10× faster than the default WebAssembly backend.

## Artifacts

Ships as dual ESM and CJS with full `.d.ts` typings:

```
dist/esm/{index,openai}.js
dist/cjs/{index,openai}.cjs
dist/types/**/*.d.ts
```

## Composite detector (recommended for production)

For categories where structure is verifiable by checksum (CH-IBAN, AHV,
VAT-CHE, credit cards, common token formats), the Python sibling
package ships a composite detector that pairs the regex catalogue with
the model. The JavaScript package implements the model side; combine
with your own regex layer or proxy through `gheim-server`, which
applies the composite detector internally.

## License

Apache 2.0. Bundled model weights are inherited from the upstream
license of the model you select.
