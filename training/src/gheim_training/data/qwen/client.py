"""vLLM-backed Qwen 3.6 35B-A3B client for second-labeller v2 work.

Default model: ``cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`` — Qwen 3.6 35B MoE
(3B active per token, same architecture profile as the Gemma 4 26B-A4B
labeller used in v1), AWQ-4bit quantized to ~17 GB. Fits comfortably on
one 4090; we run with ``tensor_parallel_size=2`` for headroom and so the
KV cache can grow large for batched generation.

Mirrors the structure of ``gemma.client.GemmaClient`` so the two
labellers are interchangeable from the runner's perspective.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


@dataclass
class QwenClient:
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

    def chat_messages(
        self,
        message_lists: Iterable[list[dict]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
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
