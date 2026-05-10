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
(transformers.js). Two models are recommended depending on the
deployment context. Both implement the same 33-class BIOES output
schema and are interchangeable.

| Model | Best for | Parameters | Notes |
|---|---|---:|---|
| [`joelbarmettler/gheim-ch-559m`](https://huggingface.co/joelbarmettler/gheim-ch-559m) | Swiss-market text (de_CH, fr_CH, it_CH, rm, en) with CH-format account numbers (IBAN, AHV, VAT-CHE) | 559M | Apache 2.0. Test F1 = 0.916 on Swiss text. ONNX export ships in the `onnx/` subfolder. |
| [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) | English-first or general use, long-context (up to 128k tokens) | 1.4B (50M active, MoE) | Apache 2.0. Wider language coverage, larger weights. |

```ts
import { LocalDetector } from "gheim";

// Recommended for Swiss-market text:
const det = new LocalDetector({ model: "joelbarmettler/gheim-ch-559m" });

// Alternative for English or general use:
const detEn = new LocalDetector({ model: "openai/privacy-filter" });
```

The package default `model` is `openai/privacy-filter`. Pass `model`
explicitly to opt into the Swiss model.

## Drop-in OpenAI client

```ts
import { OpenAI } from "gheim/openai";

const client = new OpenAI();
const r = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hi, my name is Joel" }],
});
// r.choices[0].message.content contains "Joel".
// OpenAI only ever saw "<PERSON_1>".
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
(session as any).detector = new LocalDetector({
  model: "joelbarmettler/gheim-ch-559m",
});
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

## Detectors

```ts
import { LocalDetector, RemoteDetector, defaultDetector } from "gheim";

// Local: via @huggingface/transformers, on Node and Bun.
const local = new LocalDetector({
  model: "joelbarmettler/gheim-ch-559m",
  device: "auto",
  dtype: "fp32",
});

// Remote: against your own gheim-server or api.gheim.ch.
const remote = new RemoteDetector({
  baseUrl: "http://your-host:8080",
  apiKey: "...",
});

// Picks remote when GHEIM_API_KEY is set in the environment, else local.
const auto = defaultDetector();
```

## Runtime support

Tested on Node 18+ and Bun 1.1+ with `device: "cpu"` (the default). The
underlying transformers.js library also supports `device: "webgpu"` in
modern browsers; the WebGPU path is not exercised in CI and should be
considered best-effort.

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
