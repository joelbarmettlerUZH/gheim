"""V3 synthetic chunk generator runner.

Consumes the 28 hand-written template files in templates/ and produces
the synthetic v3 layer JSONL outputs that the V2-9 balancer ingests.

Output layers (data/layer_v3_*.jsonl):
  - layer_synth_emails.jsonl       — greeting + body[+] + signature compositions
  - layer_synth_docs.jsonl         — DE legal/medical/bank/HR document templates
  - layer_synth_short_form.jsonl   — standalone narrative fragments
  - layer_synth_forms.jsonl        — form/list/CSV-style high-density chunks
  - layer_synth_common_word.jsonl  — common-word surname pos+neg pairs
  - layer_synth_adversarial.jsonl  — adversarial-negative chunks (no PII)

Each chunk goes through optional noise overlay (15% probability) so the
training mix includes realistic surface-form corruption.

Run
---
    uv run python -m gheim_training.data.synth.generate
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .core import (
    Chunk, CompositionStyle, Span, Fragment,
    compose_email, maybe_apply_noise, render_standalone,
    spans_from_values, write_jsonl,
)

# ---------------------------------------------------------------------------
# Per-output volumes (template_renders × Faker name diversity per render
# ≈ N unique chunks). Adjust here to scale up/down.
# ---------------------------------------------------------------------------

EMAIL_VOLUME_PER_LANG = {
    "de_ch": 12_000,
    "fr_ch": 8_000,
    "it_ch": 6_000,
    "rm": 3_000,
    "en": 2_000,
}
DOC_RENDERS_PER_TEMPLATE = 250  # × 51 DE doc templates = ~12.7k doc chunks
SHORT_FORM_RENDERS_PER_TEMPLATE = 80  # × 115 short-form templates = ~9.2k
FORM_RENDERS_PER_TEMPLATE = 200  # × 20 form templates = 4k
COMMON_WORD_RENDERS_PER_TEMPLATE = 60  # × 80 pos+neg = 4.8k
ADVERSARIAL_RENDERS_PER_TEMPLATE = 80  # × 35 = 2.8k

NOISE_PROBABILITY = 0.15
SEED = 20260524


# ---------------------------------------------------------------------------
# Template loaders
# ---------------------------------------------------------------------------

def _load(module_name: str) -> list[Fragment]:
    """Lazily import templates.<module_name>.TEMPLATES."""
    import importlib
    m = importlib.import_module(f"gheim_training.data.synth.templates.{module_name}")
    return list(m.TEMPLATES)


def _load_common_word() -> tuple[list[Fragment], dict[str, list[tuple[str, str]]]]:
    """Common-word templates also have a parallel SPANS dict for positives."""
    import importlib
    m = importlib.import_module(
        "gheim_training.data.synth.templates.common_word_de",
    )
    return list(m.TEMPLATES), dict(m.SPANS)


# ---------------------------------------------------------------------------
# Per-output generators
# ---------------------------------------------------------------------------

def _gen_emails_for_lang(lang: str, n_chunks: int,
                         rng: random.Random) -> list[Chunk]:
    """Compose ``n_chunks`` multi-paragraph email chunks by sampling
    one greeting + one body + one signature for the given language."""
    # Template files use bare lang codes (de, fr, it, rm, en).
    de_lang = {"de_ch": "de", "fr_ch": "fr", "it_ch": "it",
               "rm": "rm", "en": "en"}[lang]
    greetings = _load(f"greetings_{de_lang}")
    signatures = _load(f"signatures_{de_lang}")

    # Pick body modules for this language. DE has commerce/banking/HR;
    # others have the generic "_general" pool.
    if lang == "de_ch":
        bodies = (_load("bodies_de_commerce") +
                  _load("bodies_de_banking") +
                  _load("bodies_de_hr"))
    else:
        bodies = _load(f"bodies_{de_lang}_general")

    if not greetings or not bodies or not signatures:
        return []

    out: list[Chunk] = []
    for _ in range(n_chunks):
        # Pick 1-2 bodies per email (most emails are 1 body; ~20% have 2)
        n_bodies = 2 if rng.random() < 0.20 else 1
        body_sample = rng.sample(bodies, k=min(n_bodies, len(bodies)))
        chunk = compose_email(
            greeting=rng.choice(greetings),
            body=body_sample if len(body_sample) > 1 else body_sample[0],
            signature=rng.choice(signatures),
            language=lang,
            source="synthetic_emails",
            rng=rng,
        )
        chunk = maybe_apply_noise(chunk, probability=NOISE_PROBABILITY, rng=rng)
        out.append(chunk)
    return out


def _gen_docs(rng: random.Random) -> list[Chunk]:
    """DE legal/medical/bank/HR document templates, rendered standalone
    DOC_RENDERS_PER_TEMPLATE times each."""
    all_docs = (_load("docs_de_legal") + _load("docs_de_medical") +
                _load("docs_de_bank") + _load("docs_de_hr"))
    out: list[Chunk] = []
    for tpl in all_docs:
        for _ in range(DOC_RENDERS_PER_TEMPLATE):
            chunk = render_standalone(tpl, source="synthetic_docs", rng=rng)
            chunk = maybe_apply_noise(chunk, probability=NOISE_PROBABILITY,
                                      rng=rng)
            out.append(chunk)
    return out


def _gen_short_form(rng: random.Random) -> list[Chunk]:
    """Standalone narrative fragments across all 5 languages."""
    all_sf = (_load("short_form_de") + _load("short_form_fr") +
              _load("short_form_it") + _load("short_form_rm") +
              _load("short_form_en"))
    out: list[Chunk] = []
    for tpl in all_sf:
        for _ in range(SHORT_FORM_RENDERS_PER_TEMPLATE):
            chunk = render_standalone(tpl, source="synthetic_short_form",
                                      rng=rng)
            chunk = maybe_apply_noise(chunk, probability=NOISE_PROBABILITY,
                                      rng=rng)
            out.append(chunk)
    return out


def _gen_forms(rng: random.Random) -> list[Chunk]:
    """DE form/list/CSV-style high-density chunks."""
    forms = _load("forms_de")
    out: list[Chunk] = []
    for tpl in forms:
        for _ in range(FORM_RENDERS_PER_TEMPLATE):
            chunk = render_standalone(tpl, source="synthetic_forms", rng=rng)
            chunk = maybe_apply_noise(chunk, probability=NOISE_PROBABILITY,
                                      rng=rng)
            out.append(chunk)
    return out


def _gen_common_word(rng: random.Random) -> list[Chunk]:
    """Common-word surname pairs. Positives use pre-resolved literal
    spans from the SPANS dict; negatives have empty span lists."""
    import uuid
    templates, spans_map = _load_common_word()
    out: list[Chunk] = []
    for tpl in templates:
        for _ in range(COMMON_WORD_RENDERS_PER_TEMPLATE):
            # No PII markers in these templates — the text is literal.
            # Look up explicit spans for positive templates; negatives
            # get [].
            pair_specs = spans_map.get(tpl.template_id, [])
            # Normalise label aliases ("person" → "private_person").
            from .core.composers import _LABEL_MAP
            spans: list[Span] = []
            cursor = 0
            for value, short_label in pair_specs:
                idx = tpl.text.find(value, cursor)
                if idx < 0:
                    raise RuntimeError(
                        f"common_word: value {value!r} not found in "
                        f"{tpl.template_id}: {tpl.text!r}"
                    )
                spans.append(Span(start=idx, end=idx + len(value),
                                  label=_LABEL_MAP.get(short_label, short_label)))
                cursor = idx + len(value)
            chunk = Chunk(
                id=f"common_word_{uuid.uuid4().hex[:10]}",
                text=tpl.text, spans=spans,
                language=tpl.language, source="synthetic_common_word",
                template_id=tpl.template_id,
                meta={"polarity": "pos" if pair_specs else "neg"},
            )
            chunk = maybe_apply_noise(chunk, probability=NOISE_PROBABILITY,
                                      rng=rng)
            out.append(chunk)
    return out


def _gen_adversarial(rng: random.Random) -> list[Chunk]:
    """Adversarial-negative chunks. No PII markers; renders as chunks
    with empty span lists."""
    import uuid
    advs = _load("adversarial_de")
    out: list[Chunk] = []
    for tpl in advs:
        for _ in range(ADVERSARIAL_RENDERS_PER_TEMPLATE):
            chunk = Chunk(
                id=f"adv_{uuid.uuid4().hex[:10]}",
                text=tpl.text, spans=[],
                language=tpl.language,
                source="synthetic_adversarial",
                template_id=tpl.template_id,
                meta={"polarity": "neg"},
            )
            chunk = maybe_apply_noise(chunk, probability=NOISE_PROBABILITY,
                                      rng=rng)
            out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["emails", "docs", "short_form", "forms",
                             "common_word", "adversarial"],
                    help="Skip one or more output kinds (for partial re-runs).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- emails (per-language) ----
    if "emails" not in args.skip:
        all_emails: list[Chunk] = []
        for lang, n in EMAIL_VOLUME_PER_LANG.items():
            print(f"Generating {n:,} {lang} emails…", flush=True)
            all_emails.extend(_gen_emails_for_lang(lang, n, rng))
        out = args.out_dir / "layer_synth_emails.jsonl"
        n = write_jsonl(out, all_emails)
        print(f"  wrote {n:,} → {out}")

    if "docs" not in args.skip:
        print("Generating DE documents…", flush=True)
        chunks = _gen_docs(rng)
        out = args.out_dir / "layer_synth_docs.jsonl"
        n = write_jsonl(out, chunks)
        print(f"  wrote {n:,} → {out}")

    if "short_form" not in args.skip:
        print("Generating short-form narrative…", flush=True)
        chunks = _gen_short_form(rng)
        out = args.out_dir / "layer_synth_short_form.jsonl"
        n = write_jsonl(out, chunks)
        print(f"  wrote {n:,} → {out}")

    if "forms" not in args.skip:
        print("Generating DE forms…", flush=True)
        chunks = _gen_forms(rng)
        out = args.out_dir / "layer_synth_forms.jsonl"
        n = write_jsonl(out, chunks)
        print(f"  wrote {n:,} → {out}")

    if "common_word" not in args.skip:
        print("Generating common-word surname pairs (DE)…", flush=True)
        chunks = _gen_common_word(rng)
        out = args.out_dir / "layer_synth_common_word.jsonl"
        n = write_jsonl(out, chunks)
        print(f"  wrote {n:,} → {out}")

    if "adversarial" not in args.skip:
        print("Generating adversarial negatives (DE)…", flush=True)
        chunks = _gen_adversarial(rng)
        out = args.out_dir / "layer_synth_adversarial.jsonl"
        n = write_jsonl(out, chunks)
        print(f"  wrote {n:,} → {out}")


if __name__ == "__main__":
    main()
