"""vLLM-backed Nemotron 3 Nano Omni Reasoning client.

Default model: ``nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`` —
NVIDIA's official FP8 quantization of the 33B-A3B Nemotron Reasoning
model (~33 GB total). Fits on 2× RTX 4090 with TP=2 (16.5 GB per GPU);
Ada Lovelace has native FP8 tensor cores so vLLM accelerates the
matmuls directly via the ``compressed-tensors`` quant backend.

Mirrors the structure of ``gemma.client.GemmaClient`` and
``qwen.client.QwenClient`` so the runner code is interchangeable.

Quality note: FP8 keeps far more dynamic range than 4-bit AWQ/INT4,
so this is the closest-to-BF16 quant we can fit on the 4090s. NVIDIA's
own release notes peg FP8 at near-lossless vs BF16 for typical
generation. Throughput should land slightly under AWQ on the same
hardware (4-bit weights have higher arithmetic intensity per byte).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"


@dataclass
class NemotronClient:
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
