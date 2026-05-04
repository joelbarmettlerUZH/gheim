"""Drive Apertus-8B with the labeling prompt; parse its JSON outputs.

Reuses the vLLM client from ``data.apertus.client`` so we don't load two
copies of the model. The chat template is applied via the tokenizer's
built-in ``apply_chat_template``.

We tolerate two common LLM JSON quirks:
  - markdown code-fence wrapping (```json ... ```)
  - trailing commentary after the array

Anything we can't parse becomes an empty claim list (treated as "no PII"
in the audit, so it's caught immediately).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..apertus.client import DEFAULT_MODEL, ApertusClient
from .prompts import build_messages
from .verify import ApertusClaim


@dataclass
class ApertusLabeler:
    model_id: str = DEFAULT_MODEL
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    # 8192 covers the v2 prompt (~1500 tokens for system + 5 few-shot demos)
    # plus a chunk of up to 1500 chars (~750 tokens) plus 256 output tokens.
    max_model_len: int = 8192
    max_new_tokens: int = 256
    temperature: float = 0.0  # deterministic — labeling, not creativity
    _client: ApertusClient | None = None

    def _ensure(self) -> ApertusClient:
        if self._client is None:
            self._client = ApertusClient(
                model_id=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype="bfloat16",
            )
            self._client._load()
        return self._client

    def label_batch(self, chunks: list[str]) -> list[list[ApertusClaim]]:
        if not chunks:
            return []
        client = self._ensure()
        # Build chat templates manually (the apertus client's `chat()` only
        # supports system+user, but we have multi-turn few-shot demos).
        from vllm import SamplingParams
        rendered = [
            client._tok.apply_chat_template(
                build_messages(c),
                tokenize=False,
                add_generation_prompt=True,
            )
            for c in chunks
        ]
        params = SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
        )
        outputs = client._llm.generate(rendered, params)
        # vLLM returns in input order
        return [_parse_claims(o.outputs[0].text) for o in outputs]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", re.DOTALL)


def _parse_claims(text: str) -> list[ApertusClaim]:
    """Tolerant JSON parser for the unified-prompt schema (each item has
    ``value`` and ``label`` keys). Skips fence prefixes / suffix commentary."""
    if not text:
        return []
    # Strip fences if present
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    # Find the first JSON array
    m = _ARRAY_RE.search(text)
    if not m:
        return []
    try:
        data: Any = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[ApertusClaim] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        v = item.get("value")
        if not isinstance(v, str):
            continue
        lab = item.get("label", "")
        if not isinstance(lab, str):
            lab = ""
        out.append(ApertusClaim(value=v, label=lab))
    return out
