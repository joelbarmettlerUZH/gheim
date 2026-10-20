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
(test strict F1 0.910, char F1 0.946 on Swiss text, see
[MODEL_CARD.md](https://github.com/joelbarmettlerUZH/gheim/blob/main/MODEL_CARD.md)).
Any HuggingFace token-classification model that emits the same 33-class
BIOES schema can be substituted via `model:`.

| Model | Best for | Parameters | Notes |
|---|---|---:|---|
| [`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m) **(default)** | Production / commercial. Swiss court / parliament / web text with CH-format account numbers (IBAN, AHV, VAT-CHE) | 560M | Apache 2.0. Test strict F1 0.910, char F1 0.946. Hub repo ships both fp32 and `onnx/model_quantized.onnx` — load with `dtype: "q8"` for browser deployment. |
| [`joelbarmettler/gheim-ch-560m-research`](https://huggingface.co/joelbarmettler/gheim-ch-560m-research) | Research / non-commercial. Stronger cross-domain transfer on Swiss-news text (swissner PER char F1 0.90 vs 0.70 on the default) | 560M | **CC BY-NC-SA 4.0 + Reuters research-only rider.** safetensors only — no browser-targeted ONNX. |
| [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) | English-first or general use, long-context (up to 128k tokens) | 1.4B (50M active, MoE) | Apache 2.0. Wider language coverage, larger weights. |

```ts
import { LocalDetector } from "gheim";

// Default — Swiss-tuned, q8 ONNX. `device: "auto"` probes WebGPU in
// the browser and falls back to WASM; in Node it's WASM directly.
const det = new LocalDetector({ dtype: "q8" });

// Stronger cross-domain transfer (research, non-commercial; safetensors only):
const detResearch = new LocalDetector({ model: "joelbarmettler/gheim-ch-560m-research" });

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

// Picks remote when GHEIM_API_KEY is set in the environment, else the composite
// detector (checksum regex + calibrated local model, see below).
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
VAT-CHE, credit cards, common token formats) the `CompositeDetector` applies a
regex catalogue first, masks the matched spans, then runs the model on the
remainder. `defaultDetector()` returns it (over `CalibratedDetector`) unless
`GHEIM_API_KEY` is set; use `new CalibratedDetector()` for the model alone.

```ts
import { CalibratedDetector, CompositeDetector } from "gheim";

const detector = new CompositeDetector({ model: new CalibratedDetector() });
```

`gheim-server` runs the model only, with no regex layer. To add the regex front-end
and the name handling below to a server, wrap a `RemoteDetector`:
`new CompositeDetector({ model: new RemoteDetector({ baseUrl }) })`.

### Long inputs

Inputs longer than the model's 512-token limit are split into overlapping
windows (`chunkTokens: 510`, `chunkOverlapTokens: 32`) and the spans merged.
Smaller windows catch more on dense documents but produce more false positives:
on a Swiss test set of mixed letters and long email threads, 128-token windows
found noticeably more entities than 510 before repeated-name handling was added,
at about 40 extra false positives; with it, the gain was small and the false
positives remained. The default stays at 510; lower it only after measuring on
your own documents.

### Repeated names

A token classifier often tags a name once and skips a later occurrence of the
same string, so a name caught in a greeting leaks again in a signature. The
composite therefore masks every other occurrence of a string the model found as
a person (`propagateNames`, on by default). A longer propagated occurrence
replaces shorter model fragments it contains (`Käthi` inside `Zbinden Käthi`).
`propagateSurnames: true` also propagates the individual capitalised parts of
each detected name, honorifics such as "Frau" excluded: higher recall, more
false positives. `propagateNames: false` turns it off.

### Whole words

A token classifier sometimes tags only part of a word (`gano` of `Lugano`, `Zbind` +
`en` of `Zbinden`), and a window boundary can cut a word in half. `LocalDetector` and
`CalibratedDetector` widen such a span to the whole word and merge same-label spans
that overlap or touch, so one entity gets one placeholder and nothing readable is
left behind. A span is widened only when the characters on both sides of its edge
are word characters; edges at spaces or punctuation (`Gay` in `Gay-Crosier`) are
left alone.

### Confidence

`CalibratedDetector` sets each span's `score` to the mean token probability of
its tokens, computed from the same logits the label decision used. Clear cases
score near 1.0; ambiguous ones score visibly lower. Regex spans score 1.0.

### Your own patterns

Formats that only your deployment has (insurance card numbers, hospital record
numbers, internal case IDs, ...) are not built in. Pass them as `CustomPattern`
rules; they run in the regex front-end, are masked before the model sees the
text, and win over a model span they overlap.

```ts
import { CalibratedDetector, CompositeDetector, CustomPattern } from "gheim";

const detector = new CompositeDetector({
  model: new CalibratedDetector(),
  patterns: [
    // whole match is the PII
    new CustomPattern({ label: "insurance_card", regex: /\b80756(?:[ .\-]?\d){15}\b/ }),
    // hospital / case numbers after a label word (DE/FR/IT/EN); reported as
    // account_number, only the digits (group 1) are masked
    new CustomPattern({
      label: "account_number",
      regex:
        /\b(?:PID|FID|IPP|MRN|Patienten-?(?:nummer|nr\.?|id)|Patienten ?nr\.?|Fall-?(?:nummer|nr\.?)|Fall|KG-?Nr\.?|Patient ?ID|case|N° ?(?:patient|de séjour|dossier|IPP)|N\. ?(?:paziente|caso)|numero ?(?:paziente|caso))[ ]?[:#]?[ ]?(\d{6,11})\b/i,
      group: 1,
    }),
    // a label word anchors the match; only group 1 (the ID) is masked.
    // The `i` flag covers the label words; the validator keeps the ID letter upper case.
    new CustomPattern({
      label: "id_document",
      regex: /\b(?:pass|ausweis|id-?karte)(?:[ \-]?(?:nr\.?|nummer))?[ ]?[:#]?[ ]?([A-Za-z]\d{7})\b/i,
      group: 1,
      validator: (id) => /^[A-Z]/.test(id),
    }),
  ],
});
```

`label` becomes the placeholder tag (`insurance_card` -> `<INSURANCE_CARD_1>`), so
it must start with a letter and contain only letters, digits and underscores;
reuse a built-in label such as `account_number` to share its tag. A string
`regex` is compiled with `flags` (default `"i"`); a `RegExp` keeps its own.
`validator` receives the PII text and can reject a match, for example on a
checksum.

## Consulting

gheim is built and maintained by
[Souverana](https://souverana.ch), a Swiss AI consulting business helping
companies build secure, privacy-first AI systems — on-premise LLM
deployments, PII/data-protection engineering, and Swiss-market NLP. If you
need help integrating gheim or building compliant AI, get in touch at
[souverana.ch](https://souverana.ch).

## License

Apache 2.0. Bundled model weights are inherited from the upstream
license of the model you select.
