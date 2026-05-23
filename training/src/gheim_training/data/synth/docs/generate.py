"""Generate Layer 1 synthetic examples by sampling templates and filling slots.

Run:
    uv run python -m gheim_training.data.synth.generate \\
        --out data/layer1.jsonl --n 50000 --seed 17
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import cast

from ...schema import Example, Language, write_jsonl
from .. import faker_ch as F
from . import templates_de, templates_fr, templates_it
from .template import render

LANGUAGE_BANKS = {
    "de_ch": templates_de,
    "fr_ch": templates_fr,
    "it_ch": templates_it,
}


def generate(n: int, lang_weights: dict[str, float] | None = None) -> list[Example]:
    weights = lang_weights or {"de_ch": 0.5, "fr_ch": 0.3, "it_ch": 0.2}
    langs = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]

    out: list[Example] = []
    for _ in range(n):
        lang = random.choices(langs, weights=probs, k=1)[0]
        bank = LANGUAGE_BANKS[lang]
        tpl_id, tpl_str, factory = bank.random_template()
        text, spans = render(tpl_str, factory())
        ex = Example(
            text=text,
            spans=spans,
            language=cast(Language, lang),
            source="synthetic",
            template_id=tpl_id,
        )
        ex.validate_offsets()
        out.append(ex)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    F.seed_all(args.seed)
    examples = generate(args.n)
    n = write_jsonl(args.out, examples)
    print(f"wrote {n} examples → {args.out}")


if __name__ == "__main__":
    main()
