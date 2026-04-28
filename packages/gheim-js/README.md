<div align="center">

<h1 align="center" style="font-size: 28px">gheim (JavaScript / TypeScript)</h1>

<p align="center"><strong>PII round-trip for LLM APIs — JavaScript.</strong></p>

<p align="center">
  <a href="https://www.npmjs.com/package/gheim"><img src="https://img.shields.io/npm/v/gheim?color=blue&label=npm" alt="npm"></a>
  <a href="https://nodejs.org/"><img src="https://img.shields.io/badge/node-18%2B-blue" alt="Node 18+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0"></a>
</p>

</div>

See the [monorepo README](https://github.com/joelbarmettlerUZH/gheim) for the
product pitch, diagram, and speed tables.

## Install

```bash
npm install gheim                           # or: bun add gheim
npm install gheim openai                    # drop-in OpenAI client
npm install gheim @huggingface/transformers # on-device detection
```

`openai` and `@huggingface/transformers` are optional peer deps — gheim itself
pulls in nothing extra.

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

All non-chat resources are on `client.raw` once `await client.ready()`
resolves.

## Framework-agnostic

```ts
import { Session, LocalDetector, anonymizeText, deanonymizeText } from "gheim";

const session = new Session();
(session as any).detector = new LocalDetector();
const clean = await anonymizeText("Hi, my name is Joel", session);
// ... call any LLM with clean ...
const final = deanonymizeText(responseText, session);
```

Streaming deanonymizer for any text chunk source:

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

## Artifacts

Ships as dual ESM + CJS with full `.d.ts`:

```
dist/esm/{index,openai}.js
dist/cjs/{index,openai}.cjs
dist/types/**/*.d.ts
```

Works in Node 18+ and the browser (`device: "webgpu"` on `LocalDetector`).

## Detectors

```ts
import { LocalDetector, RemoteDetector, defaultDetector } from "gheim";

// Local — via @huggingface/transformers, Node or browser
const local = new LocalDetector({ device: "auto", dtype: "fp32" });

// Remote — your own gheim-server or api.gheim.ch
const remote = new RemoteDetector({
  baseUrl: "http://your-host:8080",
  apiKey: "...",
});

// Picks remote if GHEIM_API_KEY env is set, else local
const auto = defaultDetector();
```
