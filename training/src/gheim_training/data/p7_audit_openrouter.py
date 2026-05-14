"""P7 audit: re-label the P6 580-chunk sample via four named OpenRouter LLMs.

Asks four publicly-accessible 2026-era LLMs the same per-language PII-extraction
question the dataset's own Gemma labeller was asked (same prompt, same schema,
``temperature=0`` and ``seed=0``), and writes the parsed spans per-model so the
downstream scoring script can compute a majority-vote consensus and an F1 of
the released dataset's labels against that consensus.

Why P7 supersedes P6 for the paper's label-noise audit: the P6 four-way audit
used opaque "subagents" with no model name attached, so the number was
not independently reproducible. P7 names every model, fixes temperature and
seed, and saves the raw responses, so the result can be reproduced from the
released sample by anyone with an OpenRouter key.

Inputs
------
- ``data/p6_audit_sample.jsonl`` — the 580-chunk sample (gold = dataset labels).
- ``OPENROUTER_API_KEY`` env var.

Outputs (per model)
-------------------
- ``data/p7_audit_<safe_slug>.jsonl`` — one JSON record per chunk:
  ``{"id": ..., "language": ..., "model": ..., "spans": [...], "raw": ...}``
  where ``spans`` is the parsed list of ``{"value": ..., "label": ...}`` entries
  whose ``value`` is verifiable as a substring of the chunk text. The raw
  string is kept for audit reproducibility.

Run
---
    uv run python -m gheim_training.data.p7_audit_openrouter

The script is resumable: each run skips chunks whose ``id`` already appears
in the per-model output file.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from gheim_training.data.gemma.prompts import build_messages

# The four 2026-era models we audit against. All four exist on OpenRouter
# under these exact slugs as of the audit run (verified by hitting
# https://openrouter.ai/api/v1/models).
MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v4-pro",
    "minimax/minimax-m2.7",
    "z-ai/glm-5.1",
]

SAMPLE_PATH = Path("data/p6_audit_sample.jsonl")
OUT_DIR = Path("data")

# Conservative concurrency — OpenRouter routes to many providers and
# 8 in-flight requests is a polite default that finishes 580 chunks in
# a few minutes per model without tripping rate limits.
CONCURRENCY = 8
TIMEOUT = 120.0
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_OUTPUT_TOKENS = 1024
RETRIES = 3


def _safe_slug(model_id: str) -> str:
    """``moonshotai/kimi-k2.6`` -> ``moonshotai_kimi-k2.6``."""
    return model_id.replace("/", "_")


def _load_sample() -> list[dict]:
    with SAMPLE_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def _parse_spans(text: str, raw: str) -> list[dict]:
    """Pull ``{"spans": [...]}`` out of the model response and keep only
    spans whose ``value`` is a verbatim substring of ``text`` (so we can
    score against value-string equality the same way p6 did)."""
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        # Some models wrap the JSON in prose or fences; try the
        # outermost { ... } block.
        i = raw.find("{")
        j = raw.rfind("}")
        if i < 0 or j <= i:
            return []
        try:
            obj = json.loads(raw[i : j + 1])
        except Exception:
            return []
    spans_raw = obj.get("spans", [])
    if not isinstance(spans_raw, list):
        return []
    out: list[dict] = []
    for s in spans_raw:
        if not isinstance(s, dict):
            continue
        v = s.get("value")
        lbl = s.get("label")
        if not isinstance(v, str) or not isinstance(lbl, str):
            continue
        v = v.strip()
        if not v or not lbl:
            continue
        # Defensive: must appear verbatim in the chunk.
        if v not in text:
            continue
        out.append({"value": v, "label": lbl})
    return out


async def _call_model(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    chunk: dict,
    sem: asyncio.Semaphore,
) -> dict:
    """One chat-completion call with retries on 429/5xx."""
    msgs = build_messages(chunk["text"], chunk.get("language", "de_ch"))
    body = {
        "model": model,
        "messages": msgs,
        "temperature": 0,
        "seed": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/joelbarmettlerUZH/gheim",
        "X-Title": "gheim P7 audit",
    }
    last_err = ""
    async with sem:
        for attempt in range(RETRIES):
            try:
                r = await client.post(ENDPOINT, json=body, headers=headers, timeout=TIMEOUT)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = f"transport: {e!r}"
                await asyncio.sleep(2**attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"http {r.status_code}: {r.text[:200]}"
                await asyncio.sleep(2**attempt)
                continue
            if r.status_code >= 400:
                last_err = f"http {r.status_code}: {r.text[:300]}"
                break
            try:
                d = r.json()
            except Exception as e:
                last_err = f"json decode: {e!r}"
                break
            choices = d.get("choices") or []
            if not choices:
                last_err = f"no choices: {json.dumps(d)[:300]}"
                break
            content = (choices[0].get("message") or {}).get("content") or ""
            spans = _parse_spans(chunk["text"], content)
            return {
                "id": chunk["id"],
                "language": chunk.get("language"),
                "model": model,
                "spans": spans,
                "raw": content,
            }
    return {
        "id": chunk["id"],
        "language": chunk.get("language"),
        "model": model,
        "spans": [],
        "raw": "",
        "error": last_err,
    }


async def _label_with_model(model: str, chunks: list[dict], api_key: str) -> None:
    out_path = OUT_DIR / f"p7_audit_{_safe_slug(model)}.jsonl"
    done = _already_done(out_path)
    todo = [c for c in chunks if c["id"] not in done]
    print(f"[{model}] {len(done)} cached, {len(todo)} to label", flush=True)
    if not todo:
        return
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        with out_path.open("a") as fh:
            tasks = [_call_model(client, api_key, model, c, sem) for c in todo]
            done_count = 0
            errors = 0
            for coro in asyncio.as_completed(tasks):
                rec = await coro
                if rec.get("error"):
                    errors += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done_count += 1
                if done_count % 25 == 0 or done_count == len(todo):
                    elapsed = time.monotonic() - t0
                    rate = done_count / elapsed if elapsed > 0 else 0.0
                    eta = (len(todo) - done_count) / rate if rate > 0 else 0.0
                    print(
                        f"  [{model}] {done_count}/{len(todo)} "
                        f"({rate:.1f}/s, eta {eta:.0f}s, errors={errors})",
                        flush=True,
                    )


async def _amain() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    chunks = _load_sample()
    print(f"Loaded {len(chunks)} chunks from {SAMPLE_PATH}")
    # Run all four models concurrently. Each has its own per-model
    # semaphore (CONCURRENCY in-flight per model), so the total
    # in-flight ceiling is len(MODELS) * CONCURRENCY = 32. OpenRouter
    # routes per-model and the audit's polite-default rate limits are
    # well under what a single API key is allowed.
    await asyncio.gather(*[_label_with_model(m, chunks, api_key) for m in MODELS])
    print("All models done.")


if __name__ == "__main__":
    asyncio.run(_amain())
