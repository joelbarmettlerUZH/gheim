"""D3 smoke test: can Gemma 4 26B-A4B (AWQ-4bit) translate AI4Privacy
template scaffolds into Romansh + Swiss German while preserving the
[LABEL_INDEX] placeholders verbatim?

Pulls 10 English ``masked_text`` records from ``ai4privacy/pii-masking-300k``
where placeholders are present, asks Gemma to translate each into RM (Romansh,
Rumantsch Grischun) and GSW (Schwiizerdütsch), and reports:
  - placeholder preservation rate (every [LABEL_N] in source MUST appear
    verbatim in target)
  - rough fluency snapshot (printed for human eyeball)
  - throughput (chunks/sec on 2× RTX 4090 with tp=2)

Run:
    uv run python -m gheim_training.data.gemma.d3_translation_smoke
"""
from __future__ import annotations

import re
import time

from datasets import load_dataset

from .client import GemmaClient

_PLACEHOLDER_RE = re.compile(r"\[[A-Z]+_\d+\]")


def _placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(text))


def _pick_english_templates(n: int) -> list[str]:
    """Pull n English masked_text records from pii-masking-openpii-1m that
    contain >= 2 distinct placeholders.

    1m uses the clean ``[LABEL_INDEX]`` format (e.g. ``[GIVENNAME_1]``,
    ``[CITY_2]``) consistently across all records — vs 300k which uses
    ad-hoc placeholders not aligned to the gold span labels. We sample
    from the (much smaller, faster) validation split.
    """
    out: list[str] = []
    ds = load_dataset("ai4privacy/pii-masking-openpii-1m", split="validation",
                      streaming=True)
    for r in ds:
        if (r.get("language") or "").lower() != "en":
            continue
        m = r.get("masked_text") or ""
        ph = _placeholders(m)
        if len(ph) >= 2 and len(m) <= 2000:
            out.append(m)
            if len(out) >= n:
                return out
    return out


_SYSTEM = (
    "You are a careful translator. The user gives you English text "
    "containing placeholders in the form [LABEL_INDEX] (e.g. [GIVENNAME_1], "
    "[CITY_2], [DATE_1]). Translate the surrounding natural-language text "
    "into the target language while keeping every placeholder EXACTLY "
    "as-is — same spelling, same brackets, same number suffix. Do NOT "
    "translate the placeholders. Do NOT add or remove placeholders. "
    "Return ONLY the translated text, no preamble, no explanation."
)


def _user(target_lang: str, text: str) -> str:
    return f"Translate the following text into {target_lang}:\n\n{text}"


def main() -> None:
    print("=== D3: Gemma 4 26B-A4B-AWQ translation smoke test ===")
    templates = _pick_english_templates(10)
    print(f"Pulled {len(templates)} English templates with >=3 placeholders.")

    client = GemmaClient()
    print("Warming up vLLM (loading model)…")
    t0 = time.time()
    client._load()
    print(f"  load complete in {time.time() - t0:.0f}s")

    rm_prompts = [(_SYSTEM, _user("Romansh (Rumantsch Grischun)", t))
                  for t in templates]
    gsw_prompts = [(_SYSTEM, _user("Swiss German (Schwiizerdütsch, Zürich variant)", t))
                   for t in templates]

    print("\n--- Romansh translations ---")
    t0 = time.time()
    rm_outs = client.chat(rm_prompts)
    rm_elapsed = time.time() - t0
    _report("RM", templates, rm_outs, rm_elapsed)

    print("\n--- Swiss German translations ---")
    t0 = time.time()
    gsw_outs = client.chat(gsw_prompts)
    gsw_elapsed = time.time() - t0
    _report("GSW", templates, gsw_outs, gsw_elapsed)

    print("\n=== Throughput estimate ===")
    total = rm_elapsed + gsw_elapsed
    n = 2 * len(templates)
    print(f"  {n} translations in {total:.1f}s = {n/total:.2f} chunks/sec")
    print(f"  → 30k templates × 2 langs / {n/total:.2f} = "
          f"{2*30000/(n/total)/3600:.1f} hours")


def _report(lang: str, src: list[str], outs: list[str], elapsed: float) -> None:
    n_perfect = 0
    n_partial = 0
    n_broken = 0
    for s, o in zip(src, outs, strict=True):
        src_ph = _placeholders(s)
        out_ph = _placeholders(o)
        if src_ph == out_ph:
            n_perfect += 1
            tag = "✓"
        elif src_ph.issubset(out_ph):
            n_partial += 1
            tag = "~"  # extras added but none lost
        elif out_ph.issubset(src_ph):
            n_partial += 1
            tag = "−"  # some lost
        else:
            n_broken += 1
            tag = "✗"
        print(f"\n  [{tag}] src placeholders: {sorted(src_ph)}")
        print(f"      out placeholders: {sorted(out_ph)}")
        print(f"      EN: {s[:150]}…")
        print(f"      {lang}: {o[:200]}…")
    n = len(src)
    print(f"\n  Summary {lang}: {n_perfect}/{n} perfect, "
          f"{n_partial}/{n} partial, {n_broken}/{n} broken "
          f"({elapsed:.1f}s for {n} translations = {n/elapsed:.2f} chunks/sec)")


if __name__ == "__main__":
    main()
