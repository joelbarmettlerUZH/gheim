"""Quality smoke-test for two candidate v2 third-labellers.

Question we're trying to answer: for the v2 dataset build, is it worth
adding a third LLM signal (alongside Gemma + Qwen + regex)? And if so,
which of these two candidates is better?

  - deepseek-ai/DeepSeek-V4-Flash (158B params, OpenRouter only)
  - nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning

Both are 2026-era models with thinking / reasoning enabled. We re-use
the P7 four-LLM audit infrastructure: same 580-chunk sample, same
per-language Gemma prompt, ``temperature=0`` + ``seed=0``, same
``response_format={"type":"json_object"}``. Only the model slug and
output filenames change.

After this script writes per-model labellings, run
``p7_extra_score`` (in this same module) to compute:

  - F1 of each candidate vs the existing 4-LLM majority consensus
  - F1 of each candidate vs the dataset's own labels
  - The candidate's per-(category, language) breakdown so we can see
    where it agrees / disagrees most

If a candidate's F1 vs the consensus is ≥0.65 it's worth scaling up to
the full 2.3M chunks; if below 0.5 it's mostly noise and we skip.

Cost estimate at OpenRouter paid pricing:
  - DeepSeek-V4-Flash: ~580 × (~2k input + ~300 output tokens) at
    \$0.13/M in + \$0.25/M out → roughly \$0.30
  - Nemotron-3-Nano-Omni-30B-A3B: ~\$0.10 (free tier also available)

Run
---
    uv run python -m gheim_training.data.v2.p7_extra_smoke
"""
from __future__ import annotations

import asyncio
import os
import sys

import httpx

from ..p7_audit_openrouter import (
    CONCURRENCY,
    _label_with_model,
    _load_sample,
)

EXTRA_MODELS = [
    "deepseek/deepseek-v4-flash",
    # `:free` suffix is required for the Nemotron Omni-Reasoning variant —
    # OpenRouter has no paid endpoint listed for this slug, and without
    # the suffix the call returns "No endpoints found" 404.
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]


async def _amain() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    chunks = _load_sample()
    print(f"Loaded {len(chunks)} chunks")
    # Run both candidates in parallel — independent files, no conflict.
    await asyncio.gather(*[_label_with_model(m, chunks, api_key)
                          for m in EXTRA_MODELS])
    print("Both candidates done.")


if __name__ == "__main__":
    asyncio.run(_amain())
