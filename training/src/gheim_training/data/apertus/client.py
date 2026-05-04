"""vLLM-backed Apertus client.

Default model: ``swiss-ai/Apertus-8B-Instruct-2509`` (dense, vLLM-supported).
Used by ``generate.py`` to drive batched generation of Layer-3 examples.

The client wraps vLLM's ``LLM.generate`` with a small chat-template wrapper so
callers can pass (system, user) message pairs directly.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"


@dataclass
class ApertusClient:
    model_id: str = DEFAULT_MODEL
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    dtype: str = "bfloat16"
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
            dtype=self.dtype,
        )

    def chat(
        self,
        prompts: Iterable[tuple[str, str]],  # (system, user) pairs
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_p: float = 0.95,
    ) -> list[str]:
        """Two-turn chat: system + user. For multi-turn few-shot use chat_messages."""
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
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_p: float = 0.95,
    ) -> list[str]:
        """Multi-turn chat. Each entry in ``message_lists`` is a full
        [{role, content}, ...] sequence (system + few-shot user/assistant
        pairs + final user)."""
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
