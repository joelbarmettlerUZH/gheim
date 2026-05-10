"""vLLM-backed Gemma client for translation + labeling work.

Default model: ``cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`` — Gemma 4 26B MoE
(3.8B active per token), AWQ-4bit quantized to ~16 GB. Fits comfortably on
one 4090; we run with ``tensor_parallel_size=2`` for headroom and so the
KV cache can grow large for batched generation.

Mirrors the structure of ``apertus.client.ApertusClient``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"


@dataclass
class GemmaClient:
    model_id: str = DEFAULT_MODEL
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    _llm: Any = None
    _tok: Any = None

    def _load(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype="auto",
        )

    def chat(
        self,
        prompts: Iterable[tuple[str, str]],  # (system, user) pairs
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[str]:
        return self.chat_messages(
            [
                [{"role": "system", "content": s}, {"role": "user", "content": u}]
                for s, u in prompts
            ],
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        )

    def chat_messages(
        self,
        message_lists: Iterable[list[dict]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[str]:
        self._load()
        from vllm import SamplingParams
        rendered = [
            self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            for msgs in message_lists
        ]
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self._llm.generate(rendered, params)
        return [o.outputs[0].text.strip() for o in outputs]
