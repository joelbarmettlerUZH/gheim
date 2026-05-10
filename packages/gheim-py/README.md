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

See the [monorepo README](https://github.com/joelbarmettlerUZH/gheim) for the
product pitch, diagram, and speed tables.

## Install

```bash
uv add gheim                          # core — needs a RemoteDetector or GHEIM_API_KEY
uv add "gheim[local]"                 # + torch / transformers for on-device detection
uv add "gheim[openai]"                # + drop-in OpenAI client
uv add "gheim[local,openai]"          # both
```

## Drop-in OpenAI client

```python
from gheim.openai import OpenAI

client = OpenAI()  # same args as openai.OpenAI
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

session = Session(detector=LocalDetector())
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

redacted = anonymize_messages(messages, session)  # preserves role/name/tool_call_id
```

## Detector backends

```python
# Local inference — downloads weights to ~/.cache/gheim/ on first use
from gheim import LocalDetector
det = LocalDetector(device="auto", dtype=torch.bfloat16)

# Remote — your own gheim-server or api.gheim.ch
from gheim import RemoteDetector
det = RemoteDetector(base_url="http://your-host:8080", api_key="...")
```

`default_detector()` picks remote if `GHEIM_API_KEY` is set, else local.
