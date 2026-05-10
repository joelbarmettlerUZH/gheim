"""Drive Apertus to produce Layer-3 examples.

Run:
    uv run python -m gheim_training.data.apertus.generate \\
        --lang de_ch fr_ch it_ch rm gsw \\
        --n 10000 \\
        --out data/layer3_apertus.jsonl

For a small smoke test:
    uv run python -m gheim_training.data.apertus.generate \\
        --lang de_ch --n 50 --out data/smoke.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from ..schema import Example, write_jsonl
from ..synthetic import faker_ch as F
from . import prompts as P
from . import verify
from .client import DEFAULT_MODEL, ApertusClient


def _sample_languages(langs: list[str], n: int, weights: dict[str, float] | None) -> list[str]:
    w = weights or dict.fromkeys(langs, 1.0)
    probs = [w.get(lang, 1.0) for lang in langs]
    total = sum(probs)
    probs = [p / total for p in probs]
    return random.choices(langs, weights=probs, k=n)


def run(
    *,
    languages: list[str],
    n: int,
    out_path: Path,
    model_id: str = DEFAULT_MODEL,
    batch_size: int = 32,
    max_new_tokens: int = 512,
    temperature: float = 0.85,
    seed: int = 17,
) -> int:
    F.seed_all(seed)
    random.seed(seed)
    client = ApertusClient(model_id=model_id)

    plan = _sample_languages(languages, n, weights=None)
    examples: list[Example] = []
    n_dropped_missing = 0
    n_dropped_overlap = 0

    for batch_start in range(0, n, batch_size):
        batch_langs = plan[batch_start:batch_start + batch_size]
        # v2 few-shot path: build_messages returns (messages, request) — the
        # full multi-turn chat with 3 in-context demos.
        messages_and_reqs = [P.build_messages(lang) for lang in batch_langs]
        messages = [m for m, _ in messages_and_reqs]
        requests = [r for _, r in messages_and_reqs]
        outputs = client.chat_messages(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for req, text in zip(requests, outputs, strict=True):
            try:
                ex = verify.build_example(
                    text,
                    req.slot_bag,
                    language=req.language,
                    template_id=f"apertus_{req.scenario[:24]}",
                )
            except verify.SlotVerificationError:
                n_dropped_missing += 1
                continue
            except ValueError:
                n_dropped_overlap += 1
                continue
            examples.append(ex)
        print(
            f"  batch {batch_start // batch_size + 1}: "
            f"kept {len(examples)} so far, "
            f"dropped {n_dropped_missing} (missing) / {n_dropped_overlap} (overlap)"
        )

    written = write_jsonl(out_path, examples)
    print(
        f"\nwrote {written} examples → {out_path}\n"
        f"  total dropped: {n_dropped_missing + n_dropped_overlap} of {n} "
        f"({(n_dropped_missing + n_dropped_overlap) / max(1, n):.1%})"
    )
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", required=True,
                    choices=("de_ch", "fr_ch", "it_ch", "rm", "gsw"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    run(
        languages=args.lang,
        n=args.n,
        out_path=args.out,
        model_id=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
