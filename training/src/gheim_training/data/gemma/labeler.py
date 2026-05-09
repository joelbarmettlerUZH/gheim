"""Gemma 4 26B-A4B labeler with structured-output JSON decoding.

Produces gheim 8-category PII spans for real Swiss text. The output
schema is enforced at decode time by vLLM's xgrammar backend, so:

  - The model can never emit malformed JSON.
  - The `label` field is restricted to our 8 canonical categories at the
    token level — no need for downstream label normalisation.
  - The `value` field is a free string; we verify it appears verbatim in
    the source chunk (M1 gate — same approach as gheim-1's apertus_label).

Pipeline per chunk:

   chunk_text
     → Gemma (structured output → JSON {spans: [{value, label}]})
     → for each claim: locate verbatim in text via str.find
                       → drop hallucinations
                       → drop overlaps (first-by-start wins)
     → Example with character-offset Spans

Run a smoke labeling on hand-picked chunks via:
    uv run python -m gheim_training.data.gemma.poc_label_swiss
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..label_space import CATEGORIES
from ..schema import Example, Language, Span
from .client import GemmaClient
from .prompts import build_messages

# Hand-built JSON schema (xgrammar enforces this token-by-token at decode
# time). Avoiding pydantic.model_json_schema because its json_schema_extra
# REPLACES (not merges) the auto-generated properties section, silently
# dropping required fields — caught when the structured output produced
# `{"spans":[{"label":"X"}]}` (no `value`) on a chunk where the same model
# in free-form mode emitted the full {value, label} pair correctly.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["spans"],
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["value", "label"],
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "Verbatim substring of the input text.",
                    },
                    "label": {
                        "type": "string",
                        "enum": list(CATEGORIES),
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True, slots=True)
class _Claim:
    value: str
    label: str


def _parse_json(text: str) -> list[_Claim]:
    """xgrammar guarantees this parses, but be defensive."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    raw_spans = obj.get("spans") if isinstance(obj, dict) else None
    if not isinstance(raw_spans, list):
        return []
    out: list[_Claim] = []
    for s in raw_spans:
        if not isinstance(s, dict):
            continue
        v = s.get("value")
        lab = s.get("label")
        if not isinstance(v, str) or not isinstance(lab, str):
            continue
        if lab not in CATEGORIES:
            continue
        out.append(_Claim(value=v, label=lab))
    return out


def _verify_and_locate(
    chunk_text: str,
    claims: Iterable[_Claim],
) -> list[Span]:
    """Locate each claim's value verbatim in the text. Drop hallucinations
    and overlapping later spans (first-by-start wins).

    M1 gate (verbatim match): if the value doesn't appear character-for-
    character in the source, the model invented or paraphrased it; drop.
    """
    resolved: list[Span] = []
    for claim in claims:
        v = claim.value.strip()
        if len(v) < 2:
            continue
        idx = chunk_text.find(v)
        if idx < 0:
            continue
        end = idx + len(v)
        # Trim accidental whitespace at the boundaries
        while idx < end and chunk_text[idx].isspace():
            idx += 1
        while end > idx and chunk_text[end - 1].isspace():
            end -= 1
        if end <= idx:
            continue
        resolved.append(Span(start=idx, end=end, label=claim.label))

    # Resolve overlaps: first-by-start wins (matches gheim-1's verify_and_combine)
    resolved.sort(key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    last_end = -1
    for sp in resolved:
        if sp.start < last_end:
            continue
        out.append(sp)
        last_end = sp.end
    return out


@dataclass
class GemmaLabeler:
    """Wraps a GemmaClient + structured-output sampling for PII labeling."""

    client: GemmaClient | None = None
    # Greedy decoding — labeling is a deterministic extraction task, not
    # generation. Temperature 0 + structured-output schema means the model
    # always picks the highest-probability token at each step within the
    # allowed grammar, so identical input → identical output.
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512  # 512 tokens of JSON ≈ 50 spans, plenty

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = GemmaClient()

    def label_batch(
        self,
        chunks: list[str],
        languages: list[Language] | None = None,
        *,
        doc_ids: list[str] | None = None,
        subset: str = "",
    ) -> list[Example | None]:
        """Label N chunks. Returns one Example per chunk (or None if no PII).

        ``languages`` and ``doc_ids`` (if provided) must be the same length
        as ``chunks``. They're attached to each Example for downstream
        provenance — important for the publishable dataset.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        assert self.client is not None
        self.client._load()
        # Per-chunk language routing: each prompt is built with rules +
        # few-shot demos in the chunk's detected language, which our PoC
        # showed is critical (English-only prompts on German chunks
        # collapsed Gemma to defaulting "{spans:[]}" on obvious dates).
        chunk_langs: list[Language] = (
            languages if languages is not None
            else ["de_ch"] * len(chunks)
        )
        message_lists = [
            build_messages(c, language=la)
            for c, la in zip(chunks, chunk_langs, strict=True)
        ]
        rendered = [
            self.client._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            for msgs in message_lists
        ]
        sampling = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            structured_outputs=StructuredOutputsParams(json=SCHEMA),
        )
        outputs = self.client._llm.generate(rendered, sampling)
        raw_texts = [o.outputs[0].text for o in outputs]

        results: list[Example | None] = []
        for i, raw in enumerate(raw_texts):
            chunk = chunks[i]
            lang: Language = chunk_langs[i]
            doc_id = doc_ids[i] if doc_ids else ""
            claims = _parse_json(raw)
            spans = _verify_and_locate(chunk, claims)
            if not spans:
                # Empty-PII chunk — STILL emit an Example with empty spans
                # so the publishable dataset has explicit negative-control
                # rows. Caller can decide whether to keep them.
                results.append(Example(
                    text=chunk,
                    spans=[],
                    language=lang,
                    source="apertus",  # closest schema slot for "LLM-labeled real text"
                    template_id=f"layer5v3_{subset}" if subset else None,
                    meta={"doc_id": doc_id, "labeler": "gemma-4-26B-A4B-AWQ",
                          "raw_response_chars": len(raw)},
                ))
                continue
            results.append(Example(
                text=chunk,
                spans=spans,
                language=lang,
                source="apertus",
                template_id=f"layer5v3_{subset}" if subset else None,
                meta={"doc_id": doc_id, "labeler": "gemma-4-26B-A4B-AWQ",
                      "n_claims_total": len(claims),
                      "n_claims_dropped": len(claims) - len(spans)},
            ))
        return results
