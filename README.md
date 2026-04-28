<div align="center">

<h1 align="center" style="font-size: 32px">gheim</h1>

<p align="center"><strong>PII round-trip for LLM APIs. Anonymize before the request, de-anonymize the stream on the way back.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/gheim/"><img src="https://img.shields.io/pypi/v/gheim?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/gheim"><img src="https://img.shields.io/npm/v/gheim?color=blue&label=npm" alt="npm"></a>
  <a href="https://github.com/joelbarmettlerUZH/gheim/pkgs/container/gheim-server"><img src="https://img.shields.io/badge/ghcr.io-gheim--server-blue" alt="ghcr"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="https://nodejs.org/"><img src="https://img.shields.io/badge/node-18%2B-blue" alt="Node 18+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0"></a>
  <a href="https://github.com/joelbarmettlerUZH/gheim"><img src="https://img.shields.io/github/stars/joelbarmettlerUZH/gheim?style=social" alt="GitHub stars"></a>
</p>

</div>

---

## The problem

Swiss companies want to use GPT-4, Claude, and Gemini. Their lawyers don't.
Every chat message with a customer name, an address, or an account number is a
compliance problem the moment it crosses the Atlantic.

**gheim** solves it locally. Detect PII with a small on-device model, swap it
for stable sentinels, send the redacted text to any LLM, restore originals in
the streamed response. Your users never see a placeholder. OpenAI never sees a
name.

```
┌──────────────────────────────────────────────────────────────────────┐
│   user writes:   "Hi, my name is Joel, my IBAN is CH93 0076 ..."     │
│                                      │                                │
│                                      ▼                                │
│                    ┌─────────────────────────────────┐                │
│                    │  gheim (local or your endpoint) │                │
│                    │  detect → allocate sentinels    │                │
│                    └────────────────┬────────────────┘                │
│                                      ▼                                │
│           "Hi, my name is <PERSON_1>, my IBAN is <ACCOUNT_1>."        │
│                                      │                                │
│                                      ▼                                │
│                          ┌────────────────────────┐                   │
│                          │  OpenAI / Claude / etc │                   │
│                          └───────────┬────────────┘                   │
│                                      ▼                                │
│             stream:  "Hi ", "<PER", "SON_1>", ", I can help"          │
│                                      │                                │
│                                      ▼                                │
│                    ┌─────────────────────────────────┐                │
│                    │  gheim streaming deanonymizer   │                │
│                    │  hold-back + substitute + emit  │                │
│                    └────────────────┬────────────────┘                │
│                                      ▼                                │
│   user sees:  "Hi Joel, I can help..."  (never sees the sentinel)     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Packages

| what | where | ships as |
|---|---|---|
| Python library | [`packages/gheim-py`](packages/gheim-py) | `pip install gheim` |
| JavaScript library | [`packages/gheim-js`](packages/gheim-js) | `npm install gheim` |
| Detection server | [`server`](server) | `docker pull ghcr.io/joelbarmettlerUZH/gheim-server` |

All three are Apache 2.0.

---

## Quick start

### Python — drop-in OpenAI client

```bash
uv add "gheim[openai]" "gheim[local]"
```

```python
from gheim.openai import OpenAI   # same signature as openai.OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi, my name is Joel"}],
)
# response.choices[0].message.content contains "Joel".
# OpenAI only ever saw "<PERSON_1>".
```

Streaming works the same way — `stream=True` returns an iterator whose deltas
are restored as they arrive. Async support via `gheim.openai.AsyncOpenAI`.

### JavaScript / TypeScript

```bash
npm install gheim openai
```

```ts
import { OpenAI } from "gheim/openai";

const client = new OpenAI();
const r = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hi, my name is Joel" }],
});
// r.choices[0].message.content contains "Joel".
```

Ships as dual ESM + CJS with full `.d.ts` types. Works in Node 18+ and in the
browser via `@huggingface/transformers` (WebGPU).

### Framework-agnostic (any LLM)

```python
from gheim import Session, LocalDetector, anonymize_text, deanonymize_text

session = Session(detector=LocalDetector())
redacted = anonymize_text("Hi, my name is Joel", session)
# ... call any LLM with `redacted` ...
final = deanonymize_text(response_text, session)
```

For streaming LLMs:

```python
from gheim import deanonymize_stream

for chunk in deanonymize_stream(my_chunk_iterator, session):
    print(chunk, end="", flush=True)
```

### Self-host the detection server

```bash
docker run -p 8080:8080 \
  -e GHEIM_API_KEYS=your-key \
  ghcr.io/joelbarmettlerUZH/gheim-server:latest
```

Point the library at it:

```python
from gheim import RemoteDetector
detector = RemoteDetector(base_url="http://your-host:8080", api_key="your-key")
```

The server image bakes the model weights at build time. No HuggingFace download
at runtime. `HF_HUB_OFFLINE=1` in the runtime image — works in air-gapped
environments.

---

## How it works

1. **Detection.** A token-classification model (default:
   [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter), 1.5B
   params with 50M active via MoE) labels character-offset spans of PII.
2. **Allocation.** Each `(label, surface)` pair maps to a stable sentinel:
   `<PERSON_1>`, `<PERSON_2>`, `<EMAIL_1>`. Repeats coalesce — same "Joel"
   always gets the same sentinel so the LLM can refer back coherently.
3. **Send.** The redacted text (no PII) hits the LLM endpoint.
4. **Stream hold-back.** A character-level state machine buffers any partial
   `<` until it either completes a known sentinel (substitute + emit) or is
   proven incidental (emit verbatim). Bounded hold-back (64 chars), so latency
   is ~one sentinel.
5. **Round-trip.** Restored text reaches the user; they never see a placeholder.

---

## Detected PII categories

Eight categories, BIOES-tagged at the token level, aggregated to character spans:

| category | sentinel tag | examples |
|---|---|---|
| person | `<PERSON_n>` | Joel Barmettler, Dr. Schmidt |
| email | `<EMAIL_n>` | joel@example.com |
| phone | `<PHONE_n>` | +41 44 123 45 67 |
| address | `<ADDRESS_n>` | Bahnhofstrasse 1, 8001 Zürich |
| url | `<URL_n>` | https://internal.example.ch/user/42 |
| date | `<DATE_n>` | 1990-01-02, March 4th |
| account number | `<ACCOUNT_n>` | CH93 0076 2011 6238 5295 7 |
| secret | `<SECRET_n>` | API keys, bearer tokens |

Custom label spaces are supported via the underlying [`opf`](https://github.com/openai/privacy-filter)
fine-tuning machinery. A dedicated Swiss-finetuned model (AHV, CH-IBAN, CH
phone/address formats, DE/FR/IT-CH names) is the planned paid tier.

---

## Speed

Measured end-to-end on the `openai/privacy-filter` model, fp32 CPU, Intel Core
Ultra 9 185H (see [`experiments/benchmark.py`](experiments/benchmark.py) to
reproduce):

| input size | tokens | detection latency | throughput |
|---|---:|---:|---:|
| short (one sentence) | 13 | 77 ms | 168 tok/s |
| medium (short email) | 95 | 214 ms | 444 tok/s |
| long (~3 paragraphs) | 760 | 796 ms | 954 tok/s |
| huge (6k tokens) | 6,080 | 17.4 s | 349 tok/s |

Peak throughput at ~760-token inputs. For multi-kilobyte documents, chunking
into ~1 k-token windows is faster than feeding the whole document at once.

Streaming deanonymization is O(n) with a bounded 64-char hold-back — no
measurable overhead beyond the LLM's own token cadence.

---

## Limitations

- **Base model is English-centric.** The upstream
  `openai/privacy-filter` is optimized for English, Latin scripts. Non-English
  Swiss content (DE/FR/IT) degrades gracefully but misses region-specific
  patterns (AHV numbers, CH-IBAN, `+41` phone layouts). A Swiss-finetuned model
  is the right fix — we're working on it.
- **Not a compliance guarantee.** gheim is one layer in a privacy-by-design
  approach. Test with your local policy references before production.
- **LLM-side mangling.** Some models occasionally rewrite sentinels
  (e.g. `<Person_1>` or `**<PERSON_1>**`). Lowercase variants are not
  substituted by default. Markdown wrapping is handled. Survival rates across
  GPT-4o / Claude / Gemini are part of our evaluation plan.
- **Detection offsets in the browser.** `@huggingface/transformers` doesn't
  emit character offsets for aggregated entities as of v4. gheim's JS
  `LocalDetector` reconstructs offsets via left-to-right substring search (see
  `experiments/chrome/` and the fallback tests).

---

## Reproducing the CPU benchmark

```bash
git clone https://github.com/joelbarmettlerUZH/gheim.git
cd gheim
uv sync --all-packages
uv run python experiments/benchmark.py
```

Downloads ~3 GB on first run. Runs on CPU by default.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

The default model weights (`openai/privacy-filter`) are also Apache 2.0. The
future Swiss-finetuned model will ship under a separate commercial license.

---

## Citation

```bibtex
@software{barmettler2026gheim,
  author = {Barmettler, Joel},
  title  = {gheim: PII Round-Trip for LLM APIs},
  year   = {2026},
  url    = {https://github.com/joelbarmettlerUZH/gheim}
}
```
