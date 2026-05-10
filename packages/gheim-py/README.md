<div align="center">

<p align="center">
  <img src="https://raw.githubusercontent.com/joelbarmettlerUZH/gheim/main/assets/logo.png" alt="gheim" width="360">
</p>

<p align="center"><strong>gheim (Python). PII round-trip for LLM APIs.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/gheim/"><img src="https://img.shields.io/pypi/v/gheim?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"></a>
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
uv add gheim                          # core: pairs with a RemoteDetector or GHEIM_API_KEY
uv add "gheim[local]"                 # + torch and transformers for on-device detection
uv add "gheim[openai]"                # + drop-in OpenAI client
uv add "gheim[local,openai]"          # both
```

## Model choice

`LocalDetector` runs a token-classification model in process. Two models
are recommended depending on the deployment context. Both implement the
same 33-class BIOES output schema and are interchangeable in
`LocalDetector` and the rest of the API.

| Model | Best for | Parameters | Notes |
|---|---|---:|---|
| [`joelbarmettler/gheim-ch-559m`](https://huggingface.co/joelbarmettler/gheim-ch-559m) | Swiss-market text (de_CH, fr_CH, it_CH, rm, en) with CH-format account numbers (IBAN, AHV, VAT-CHE) | 559M | Apache 2.0. Test F1 = 0.916 on Swiss text. |
| [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) | English-first or general use, long-context (up to 128k tokens) | 1.4B (50M active, MoE) | Apache 2.0. Wider language coverage, larger weights. |

```python
from gheim import LocalDetector

# Recommended for Swiss-market text:
det = LocalDetector(model_id="joelbarmettler/gheim-ch-559m")

# Alternative for English or general use:
det = LocalDetector(model_id="openai/privacy-filter")
```

The package default `model_id` is `openai/privacy-filter`. Set
`GHEIM_DEFAULT_MODEL=joelbarmettler/gheim-ch-559m` in the environment, or
pass `model_id` explicitly, to opt into the Swiss model.

## Drop-in OpenAI client

```python
from gheim.openai import OpenAI

client = OpenAI()  # same constructor args as openai.OpenAI
r = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi, my name is Joel"}],
)
# r.choices[0].message.content contains "Joel".
# OpenAI only ever saw "<PERSON_1>".
```

Streaming:

```python
stream = client.chat.completions.create(..., stream=True)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Async:

```python
from gheim.openai import AsyncOpenAI
client = AsyncOpenAI()
r = await client.chat.completions.create(...)
```

Per-call overrides:

```python
from gheim import Session
session = Session()  # reuse across calls for multi-turn coherent sentinels
r = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    gheim_session=session,    # or gheim_detector=...
)
```

## Framework-agnostic

```python
from gheim import Session, LocalDetector, anonymize_text, deanonymize_text

session = Session(detector=LocalDetector(model_id="joelbarmettler/gheim-ch-559m"))
clean = anonymize_text("Hi, my name is Joel", session)
# ... call any LLM with clean ...
final = deanonymize_text(response_text, session)
```

Streaming deanonymizer:

```python
from gheim import deanonymize_stream
for chunk in deanonymize_stream(my_chunk_iterator, session):
    print(chunk, end="", flush=True)
```

Chat-message helpers:

```python
from gheim import anonymize_messages

redacted = anonymize_messages(messages, session)  # preserves role, name, tool_call_id
```

## Detector backends

```python
from gheim import LocalDetector, RemoteDetector, default_detector

# Local inference. Weights download to the HF cache on first use.
det = LocalDetector(model_id="joelbarmettler/gheim-ch-559m",
                    device="auto", dtype=torch.bfloat16)

# Remote inference against your own gheim-server or api.gheim.ch.
det = RemoteDetector(base_url="http://your-host:8080", api_key="...")

# default_detector() picks remote if GHEIM_API_KEY is set, else local.
det = default_detector()
```

## Composite detector (recommended for production)

For categories where structure is verifiable by checksum (CH-IBAN, AHV,
VAT-CHE, credit cards, common token formats) the package ships a regex
catalogue under `gheim.detectors.composite` that pairs with the model
detector. The composite detector applies regex first, masks matched
spans, then runs the model on the remainder. This pushes effective
recall on `account_number`, `private_phone`, and `private_url` close
to 1.0 with high precision; the underlying ML model handles person
names, addresses, and dates.

## License

Apache 2.0. Bundled model weights are inherited from the upstream
license of the model you select.
