"""Synthetic chunk generator.

Consumes the hand-written template files in ``templates/`` and produces
synthetic layer JSONL outputs that the balancer ingests.

Output layers (under ``data/``):
  - ``layer_synth_emails.jsonl``        — greeting + body[+] + signature
                                          composer-rendered emails (all 5 langs)
  - ``layer_synth_docs.jsonl``          — DE legal/medical/bank/HR document
                                          templates (high-density, ~7 spans/doc)
  - ``layer_synth_multilang_docs.jsonl``— KYC/HR/bank/medical/secret-leak/
                                          doctor-note documents (de/fr/it)
  - ``layer_synth_short_form.jsonl``    — standalone narrative fragments
  - ``layer_synth_forms.jsonl``         — DE structured layouts
                                          (CSV / vCard / YAML / etc.)
  - ``layer_synth_common_word.jsonl``   — common-word surname pos+neg pairs
                                          (Bach as person vs Bach as creek)
  - ``layer_synth_adversarial.jsonl``   — adversarial-negative chunks (no PII)
  - ``layer_synth_rm_secrets.jsonl``    — RM-context secret slots

15% of chunks get random noise overlay (NBSP / broken caps / no_space /
markdown / HTML entity / stray punct / OCR swap) so the training mix
includes realistic surface-form corruption.

Run
---
    uv run python -m gheim_training.data.synth.generate
"""
from __future__ import annotations

import argparse
import random
import uuid
from pathlib import Path

from .core import (
    Chunk, CompositionStyle, Fragment, Span,
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
DOC_RENDERS_PER_TEMPLATE = 250         # × 51 DE doc templates = ~12.7k
MULTILANG_DOC_RENDERS_PER_TEMPLATE = 1_400  # × 18 templates = 25.2k (replaces old Layer 1)
SHORT_FORM_RENDERS_PER_TEMPLATE = 80   # × 115 short-form templates = ~9.2k
FORM_RENDERS_PER_TEMPLATE = 200        # × 20 form templates = 4k
COMMON_WORD_RENDERS_PER_TEMPLATE = 60  # × 80 pos+neg = 4.8k
ADVERSARIAL_RENDERS_PER_TEMPLATE = 80  # × 35 = 2.8k
RM_SECRETS_RENDERS_PER_TEMPLATE = 40   # × 20 templates = 800

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
    bare = {"de_ch": "de", "fr_ch": "fr", "it_ch": "it",
            "rm": "rm", "en": "en"}[lang]
    greetings = _load(f"greetings_{bare}")
    signatures = _load(f"signatures_{bare}")

    # Pick body modules for this language. DE has commerce/banking/HR;
    # others have the generic "_general" pool.
    if lang == "de_ch":
        bodies = (_load("bodies_de_commerce") +
                  _load("bodies_de_banking") +
                  _load("bodies_de_hr"))
    else:
        bodies = _load(f"bodies_{bare}_general")

    if not greetings or not bodies or not signatures:
        return []

    out: list[Chunk] = []
    for _ in range(n_chunks):
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


def _gen_standalone_layer(templates: list[Fragment], source: str,
                          n_renders_per_template: int,
                          rng: random.Random) -> list[Chunk]:
    """Render each template ``n_renders_per_template`` times standalone."""
    out: list[Chunk] = []
    for tpl in templates:
        for _ in range(n_renders_per_template):
            chunk = render_standalone(tpl, source=source, rng=rng)
            chunk = maybe_apply_noise(chunk, probability=NOISE_PROBABILITY, rng=rng)
            out.append(chunk)
    return out


def _gen_common_word(rng: random.Random) -> list[Chunk]:
    """Common-word surname pairs. Positives use pre-resolved literal
    spans from the SPANS dict; negatives have empty span lists."""
    templates, spans_map = _load_common_word()
    out: list[Chunk] = []
    for tpl in templates:
        for _ in range(COMMON_WORD_RENDERS_PER_TEMPLATE):
            pair_specs = spans_map.get(tpl.template_id, [])
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
                    choices=["emails", "docs", "multilang_docs", "short_form",
                             "forms", "common_word", "adversarial", "rm_secrets"],
                    help="Skip one or more output kinds (for partial re-runs).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "emails" not in args.skip:
        all_emails: list[Chunk] = []
        for lang, n in EMAIL_VOLUME_PER_LANG.items():
            print(f"Generating {n:,} {lang} emails…", flush=True)
            all_emails.extend(_gen_emails_for_lang(lang, n, rng))
        n = write_jsonl(args.out_dir / "layer_synth_emails.jsonl", all_emails)
        print(f"  wrote {n:,} → layer_synth_emails.jsonl")

    if "docs" not in args.skip:
        print("Generating DE legal/medical/bank/HR documents…", flush=True)
        all_docs = (_load("docs_de_legal") + _load("docs_de_medical") +
                    _load("docs_de_bank") + _load("docs_de_hr"))
        chunks = _gen_standalone_layer(all_docs, "synthetic_docs",
                                       DOC_RENDERS_PER_TEMPLATE, rng)
        n = write_jsonl(args.out_dir / "layer_synth_docs.jsonl", chunks)
        print(f"  wrote {n:,} → layer_synth_docs.jsonl")

    if "multilang_docs" not in args.skip:
        print("Generating multilang KYC/HR/bank/medical/secret/doctor docs…",
              flush=True)
        templates = _load("multilang_docs")
        chunks = _gen_standalone_layer(templates, "synthetic_multilang_docs",
                                       MULTILANG_DOC_RENDERS_PER_TEMPLATE, rng)
        n = write_jsonl(args.out_dir / "layer_synth_multilang_docs.jsonl", chunks)
        print(f"  wrote {n:,} → layer_synth_multilang_docs.jsonl")

    if "short_form" not in args.skip:
        print("Generating short-form narrative…", flush=True)
        all_sf = (_load("short_form_de") + _load("short_form_fr") +
                  _load("short_form_it") + _load("short_form_rm") +
                  _load("short_form_en"))
        chunks = _gen_standalone_layer(all_sf, "synthetic_short_form",
                                       SHORT_FORM_RENDERS_PER_TEMPLATE, rng)
        n = write_jsonl(args.out_dir / "layer_synth_short_form.jsonl", chunks)
        print(f"  wrote {n:,} → layer_synth_short_form.jsonl")

    if "forms" not in args.skip:
        print("Generating DE structured layouts…", flush=True)
        chunks = _gen_standalone_layer(_load("forms_de"), "synthetic_forms",
                                       FORM_RENDERS_PER_TEMPLATE, rng)
        n = write_jsonl(args.out_dir / "layer_synth_forms.jsonl", chunks)
        print(f"  wrote {n:,} → layer_synth_forms.jsonl")

    if "common_word" not in args.skip:
        print("Generating common-word surname pairs (DE)…", flush=True)
        chunks = _gen_common_word(rng)
        n = write_jsonl(args.out_dir / "layer_synth_common_word.jsonl", chunks)
        print(f"  wrote {n:,} → layer_synth_common_word.jsonl")

    if "adversarial" not in args.skip:
        print("Generating adversarial negatives (DE)…", flush=True)
        chunks = _gen_adversarial(rng)
        n = write_jsonl(args.out_dir / "layer_synth_adversarial.jsonl", chunks)
        print(f"  wrote {n:,} → layer_synth_adversarial.jsonl")

    if "rm_secrets" not in args.skip:
        print("Generating RM-context secret slots…", flush=True)
        chunks = _gen_standalone_layer(_load("rm_secrets"),
                                       "synthetic_rm_secrets",
                                       RM_SECRETS_RENDERS_PER_TEMPLATE, rng)
        n = write_jsonl(args.out_dir / "layer_synth_rm_secrets.jsonl", chunks)
        print(f"  wrote {n:,} → layer_synth_rm_secrets.jsonl")


if __name__ == "__main__":
    main()
