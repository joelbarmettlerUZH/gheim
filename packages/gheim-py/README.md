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

`LocalDetector` runs a token-classification model in process. The
package's default model is
[`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m)
— a 560M xlm-roberta-large fine-tune optimised for Swiss-market PII
(test strict F1 0.910, char F1 0.946 on Swiss text, see
[MODEL_CARD.md](https://github.com/joelbarmettlerUZH/gheim/blob/main/MODEL_CARD.md)).
Any HuggingFace token-classification model that emits the same 33-class
BIOES schema can be substituted via the `model_id` constructor arg.

| Model | Best for | Parameters | Notes |
|---|---|---:|---|
| [`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m) **(default)** | Production / commercial. Swiss court / parliament / web text with CH-format account numbers (IBAN, AHV, VAT-CHE) | 560M | Apache 2.0. Test strict F1 0.910, char F1 0.946. |
| [`joelbarmettler/gheim-ch-560m-research`](https://huggingface.co/joelbarmettler/gheim-ch-560m-research) | Research / non-commercial. Stronger cross-domain transfer on Swiss-news text (swissner PER char F1 0.90 vs 0.70 on the default) | 560M | **CC BY-NC-SA 4.0 + Reuters research-only rider.** In-distribution numbers identical to the default. |
| [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) | English-first or general use, long-context (up to 128k tokens) | 1.4B (50M active, MoE) | Apache 2.0. Wider language coverage, larger weights. |

```python
from gheim import LocalDetector

# Default — Swiss-tuned, 560M, Apache 2.0:
det = LocalDetector()

# Stronger cross-domain transfer (research, non-commercial):
det = LocalDetector(model_id="joelbarmettler/gheim-ch-560m-research")

# Alternative for English or general use:
det = LocalDetector(model_id="openai/privacy-filter")
```

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

Custom endpoint or key (e.g. OpenRouter, local vLLM):

```python
client = OpenAI(api_key="sk-or-...", base_url="https://openrouter.ai/api/v1")
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

session = Session(detector=LocalDetector())  # gheim-ch-560m by default
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

## Wrapped endpoints

The drop-in `OpenAI` / `AsyncOpenAI` clients automatically protect every
text-carrying endpoint: `chat.completions`, `responses`, `completions` (legacy),
`embeddings`, `moderations`, `audio.speech`, `audio.transcriptions`,
`audio.translations`, `images.generate`, `images.edit`. Tool-call arguments and
SSE delta chunks are restored on the way back. See the
[monorepo README](https://github.com/joelbarmettlerUZH/gheim) for the full
coverage matrix and the embeddings caveat.

### Strict mode

`gheim_strict=True` (default) raises `RuntimeError` if you call an unwrapped
endpoint (`beta.assistants`, `batches`, `files`, `uploads`, `fine_tuning`,
`vector_stores`). The error message names `client.raw.<path>` as the documented
escape hatch.

```python
client = OpenAI(gheim_strict=False)  # downgrade to one-time warnings
client.raw.beta.assistants.create(...)  # always works regardless of strict mode
```

## Detector backends

```python
import torch
from gheim import LocalDetector, RemoteDetector, default_detector

# Local inference. Weights download to the HF cache on first use.
# `model_id` defaults to "joelbarmettler/gheim-ch-560m"; pass
# `dtype=torch.bfloat16` for half-precision GPU inference.
det = LocalDetector(device="auto", dtype=torch.bfloat16)

# Remote inference against your own gheim-server or api.gheim.ch.
det = RemoteDetector(base_url="http://your-host:8080", api_key="...")

# default_detector() picks remote if GHEIM_API_KEY is set, else the composite
# detector (checksum regex + calibrated local model, see below).
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
names, addresses, and dates. `default_detector()` returns this composite
(over `CalibratedDetector`) unless `GHEIM_API_KEY` is set; use
`CalibratedDetector()` directly for the model alone.

### Long inputs

Inputs longer than the model's 512-token limit are split into overlapping
windows (`chunk_tokens=510`, `chunk_overlap_tokens=32`) and the spans merged.
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
a person (`propagate_names=True`, the default). A longer propagated occurrence
replaces shorter model fragments it contains (`Käthi` inside `Zbinden Käthi`).
`propagate_surnames=True` also propagates the individual capitalised parts of
each detected name, honorifics such as "Frau" excluded: higher recall, more
false positives. `propagate_names=False` turns it off.

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
its tokens (as `aggregation_strategy="simple"` does), computed from the same
logits the label decision used. Clear cases score near 1.0; ambiguous ones, such
as a surname that is also a company name, score visibly lower. Regex spans score
1.0.

### Your own patterns

Formats that only your deployment has (insurance card numbers, hospital record
numbers, internal case IDs, ...) are not built in. Pass them as `CustomPattern` rules; they run in the
regex front-end, are masked before the model sees the text, and win over a
model span they overlap.

```python
import re
from gheim import CompositeDetector, CustomPattern, CalibratedDetector

detector = CompositeDetector(
    model=CalibratedDetector(),
    patterns=[
        # whole match is the PII
        CustomPattern("insurance_card", r"\b80756(?:[ .\-]?\d){15}\b"),
        # hospital / case numbers after a label word (Swiss hospital letters, DE/FR/IT/EN);
        # reported as account_number, only the digits (group 1) are masked
        CustomPattern(
            "account_number",
            r"\b(?:PID|FID|IPP|MRN|Patienten-?(?:nummer|nr\.?|id)|Patienten ?nr\.?|Fall-?(?:nummer|nr\.?)|Fall|"
            r"KG-?Nr\.?|Patient ?ID|case|N° ?(?:patient|de séjour|dossier|IPP)|N\. ?(?:paziente|caso)|"
            r"numero ?(?:paziente|caso))[ ]?[:#]?[ ]?(\d{6,11})\b",
            group=1,
        ),
        # a label word anchors the match; only group 1 (the ID) is masked
        CustomPattern(
            "id_document",
            r"\b(?i:pass|ausweis|id-?karte)(?i:[ \-]?(?:nr\.?|nummer))?[ ]?[:#]?[ ]?([A-Z]\d{7})\b",
            group=1,
            flags=0,  # label words are case-insensitive via (?i:...); the ID letter is not
        ),
    ],
)
```

`label` becomes the placeholder tag (`insurance_card` -> `<INSURANCE_CARD_1>`),
so it must start with a letter and contain only letters, digits and
underscores; reuse a built-in label such as `account_number` to share its tag.
An optional `validator` callable can reject a match, for example on a checksum.

## LiteLLM proxy plugin

A guardrail for the [LiteLLM](https://docs.litellm.ai) proxy, in the role of the
Presidio guardrail. Requests are anonymized before they reach any provider;
responses are restored on the way back, **chunk by chunk when streaming**.
Only a possibly-incomplete sentinel (a few characters) is held back, so
time-to-first-token and chunk cadence are preserved. Presidio's
`output_parse_pii` buffers the whole stream and re-emits it as one chunk.

```bash
pip install "gheim[litellm,local]"      # drop `local` when using a remote detector
```

```yaml
# config.yaml
guardrails:
  - guardrail_name: gheim-pii
    litellm_params:
      guardrail: gheim.litellm.GheimGuardrail
      mode: pre_call          # post_call is added automatically
      default_on: true
      # detector: remote      # remote | local | calibrated | "module:factory"
      # api_key: os.environ/GHEIM_API_KEY
```

LiteLLM releases that predate importable guardrail paths (e.g. v1.83) only
load classes from a file next to `config.yaml`. Put a one-line
`gheim_guardrail.py` there — `from gheim.litellm import GheimGuardrail` — and
use `guardrail: gheim_guardrail.GheimGuardrail`.

Every detector knob is available in YAML, under `litellm_params`. The default
is the composite detector (checksum regexes + `gheim-ch-560m`), as in Python:

```yaml
    litellm_params:
      guardrail: gheim.litellm.GheimGuardrail
      mode: pre_call
      default_on: true
      detector: composite        # composite | calibrated | local | remote | "module:factory"
      detector_model: joelbarmettler/gheim-ch-560m
      o_bias: 0.5                # calibrated / composite
      device: auto               # auto | cpu | cuda | cuda:N
      dtype: bfloat16            # name of a torch dtype
      chunk_tokens: 510
      chunk_overlap_tokens: 32
      batch_size: 8
      trim_whitespace: true
      propagate_names: true      # composite: mask later occurrences of a found name
      propagate_surnames: false  # composite: also the single capitalised name parts
      custom_patterns:           # composite: regex rules for your own formats
        - label: case_id         # becomes <CASE_ID_1>
          regex: 'Dossier\s+(ZH-\d{4}-\d{5})'
          group: 1               # capture group that is the PII (0 = whole match)
          flags: IGNORECASE      # name, "A|B" or a list (default IGNORECASE)
          validator: mypkg.checks:is_valid   # optional, str -> bool
          name: court-dossier    # optional, for error messages
      normalizers:               # per-label canonical forms (needs gheim[normalizers])
        private_phone: e164
        private_date: iso_date
      api_key: os.environ/GHEIM_API_KEY      # remote detector
      api_base: https://api.gheim.ch
      timeout: 30
      restore_output: true       # false: return the model's anonymized output as-is
      session_ttl_seconds: 3600  # drop state of calls that never reach a response hook
```

A knob the chosen detector would ignore (say `o_bias` with `remote`, or
`custom_patterns` with `local`) is a startup error, not a silent no-op.
Setting `api_key` (or `GHEIM_API_KEY`) without `detector` selects `remote`.

Scope: `/chat/completions` — text content (incl.
multimodal text parts), restored in `content`, reasoning and tool-call
arguments (JSON-escaped), streaming and non-streaming. The sentinel mapping
stays in process memory and is never written to request metadata or logs.
Raw-byte passthrough streams (e.g. native Anthropic SSE) are not restored.

Live tests against the official Docker image: `tests/litellm_live/run_live.sh`.

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
