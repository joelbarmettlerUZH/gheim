<div align="center">

<p align="center">
  <img src="https://raw.githubusercontent.com/joelbarmettlerUZH/gheim/main/assets/logo.png" alt="gheim" width="360">
</p>

<p align="center"><strong>gheim-server. PII detection HTTP server, pairs with the <code>gheim</code> client libraries.</strong></p>

<p align="center">
  <a href="https://github.com/joelbarmettlerUZH/gheim/pkgs/container/gheim-server"><img src="https://img.shields.io/badge/ghcr.io-gheim--server-blue" alt="ghcr"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0"></a>
</p>

</div>

See the [monorepo README](https://github.com/joelbarmettlerUZH/gheim) for the
product pitch.

## Run

```bash
docker run -p 8080:8080 ghcr.io/joelbarmettlerUZH/gheim-server:latest
```

With API-key auth:

```bash
docker run -p 8080:8080 \
  -e GHEIM_API_KEYS=key_a,key_b \
  ghcr.io/joelbarmettlerUZH/gheim-server:latest
```

The image bakes the `openai/privacy-filter` weights at build time. No
HuggingFace download at runtime. `HF_HUB_OFFLINE=1` — works in air-gapped
environments.

## Wire protocol

```
POST /v1/detect
Authorization: Bearer <key>            # optional if GHEIM_API_KEYS is unset

{ "model": "openai/privacy-filter", "text": "Hi, my name is Joel" }
```

```
200 OK
{
  "model": "openai/privacy-filter",
  "spans": [
    { "label": "private_person", "start": 15, "end": 19, "score": 0.999 }
  ],
  "elapsed_ms": 74.2
}
```

That's the entire surface. The server only ever sees input text and returns
character-offset spans — no LLM-bound traffic, no response data. This is the
foundation of the Swiss-data-residency story: customers can point
`gheim.RemoteDetector` at a CH-hosted instance of this server without
introducing a full LLM proxy.

## Health

```
GET /health   → { "status": "ok", "version": "0.1.0" }
```

## Point the client at it

Python:

```python
from gheim import RemoteDetector
det = RemoteDetector(base_url="http://your-host:8080", api_key="key_a")
```

JavaScript:

```ts
import { RemoteDetector } from "gheim";
const det = new RemoteDetector({ baseUrl: "http://your-host:8080", apiKey: "key_a" });
```

## Environment variables

| var | default | purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | bind address |
| `PORT` | `8080` | bind port |
| `GHEIM_API_KEYS` | *(unset)* | comma-separated allowlist; unset = open mode |
| `GHEIM_DEFAULT_MODEL` | `openai/privacy-filter` | model id used when the request body omits it |
| `GHEIM_ALLOWED_MODELS` | *= `GHEIM_DEFAULT_MODEL`* | comma-separated model allowlist |
| `GHEIM_SKIP_WARMUP` | *(unset)* | if `1`, skip model preload on startup (useful in tests) |

## Run from source

```bash
uv run gheim-server
# or:
uv run uvicorn gheim_server.main:app --host 0.0.0.0 --port 8080
```
