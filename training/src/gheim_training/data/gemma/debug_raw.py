"""Debug: print raw Gemma output for one chunk so we can see what the model
actually emits. Temporarily bypasses the labeler's parsing/verifying."""
from __future__ import annotations

from .client import GemmaClient
from .labeler import SCHEMA
from .prompts import build_messages

CHUNK_DE = (
    "A.d Am 19. Juli 2010 gelangte X. an die Aufsichtsbehörde und "
    "beschwerte sich über die im Konkursprotokoll vom 2. Juni 2010 "
    "aufgeführten ordentlichen Konkurskosten im Umfang von Fr. 6'924.25. "
    "Die Aufsichtsbehörde wies die Beschwerde am 13. Juni 2012 ab."
)


def main() -> None:
    print("=== Debug: raw Gemma output on a known-PII German chunk ===\n")
    print(f"CHUNK:\n{CHUNK_DE}\n")

    client = GemmaClient()
    client._load()

    # Variant A: no structured output (free text)
    from vllm import SamplingParams
    msgs = build_messages(CHUNK_DE, language="de_ch")
    rendered = [client._tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )]

    print(f"--- Rendered prompt length: {len(rendered[0])} chars ---")
    print("--- Last 400 chars of prompt ---")
    print(rendered[0][-400:])
    print()

    # A: free-form (no structured output)
    free = client._llm.generate(
        rendered, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512),
    )
    print("--- A: FREE-FORM output (no schema) ---")
    print(free[0].outputs[0].text)
    print()

    # B: structured output
    from vllm.sampling_params import StructuredOutputsParams
    structured = client._llm.generate(
        rendered, SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=512,
            structured_outputs=StructuredOutputsParams(json=SCHEMA),
        ),
    )
    print("--- B: STRUCTURED-OUTPUT output (schema-constrained) ---")
    print(structured[0].outputs[0].text)
    print()

    # C: try with the OPENAI-style chat template path - maybe gemma's
    # chat template handles system+messages differently
    print("--- system prompt length: ---", len(msgs[0]["content"]))
    print("--- # messages: ---", len(msgs))


if __name__ == "__main__":
    main()
