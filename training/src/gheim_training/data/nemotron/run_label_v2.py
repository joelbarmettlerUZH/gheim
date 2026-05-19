"""V2 third-labeller pass via Nemotron 3 Nano Omni Reasoning (FP8).

Mirrors ``qwen.run_label_v2``: streams the same v1 chunk corpus
(``data/layer5v4_lang_fix.jsonl``), applies the same per-language
Gemma prompt at ``temperature=0`` with NVIDIA's official FP8 quant of
Nemotron-3-Nano-Omni-30B-A3B-Reasoning, writes one JSON record per
chunk to ``data/layer5v4_nemotron.jsonl``.

The output schema parallels the Qwen runner's:

    {
      "id": "<chunk id>",
      "language": "de_ch | fr_ch | it_ch | rm | en",
      "nemotron_spans": [{"value": "...", "label": "..."}, ...],
      "nemotron_raw": "<raw model output>"
    }

So the v2 build's ``merge_signals`` step picks up Nemotron spans the
same way it picks up Qwen spans, and the per-span ``signals`` field
in the published v2 dataset will list every labeller (gemma, qwen,
nemotron, regex) that flagged each span.

Order of operations
-------------------
The current Qwen run uses both GPUs. **Launch this AFTER Qwen finishes**
(or after deliberately killing it). Running both simultaneously will
exhaust the 2× RTX 4090 setup.

Smoke-test mode
---------------
For a quick sanity check (~5 minutes after model load), run with
``--limit 16`` to label 16 chunks and verify output format. Compare
against the existing Qwen and Gemma outputs for the same chunk ids
to spot any obvious quality regressions before kicking off the full
~3-day run.

Run
---
    # Smoke test
    uv run python -m gheim_training.data.nemotron.run_label_v2 \\
        --out /tmp/nemotron_smoke.jsonl --batch-size 16 --limit 16

    # Full pass (kick off after Qwen finishes)
    uv run python -m gheim_training.data.nemotron.run_label_v2 \\
        --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..gemma.labeler import _parse_json, _verify_and_locate
from ..gemma.prompts import build_messages
from .client import DEFAULT_MODEL, NemotronClient

DEFAULT_IN = Path("data/layer5v4_lang_fix.jsonl")
DEFAULT_OUT = Path("data/layer5v4_nemotron.jsonl")
DEFAULT_BATCH = 256
DEFAULT_MAX_NEW_TOKENS = 512  # thinking is disabled at the chat-template
                               # level, so the model goes straight to JSON
                               # output. 512 tokens covers ~50 spans of
                               # JSON, plenty for a 400-800 char chunk.


def _already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def _stream_inputs(path: Path, skip: set[str]):
    """Yield (id, text, language) for chunks not yet labelled,
    GROUPED BY LANGUAGE so consecutive batches share the same
    ~2000-token Gemma prompt prefix and vLLM's prefix cache hits.

    Without grouping, the cache thrashes (5 languages cycle through
    each batch, every chunk evicts the prior prefix). With grouping,
    we pay the prefill cost once per language for the whole run.

    Loading the full id+lang index into RAM is fine: 2.3M records ×
    ~80 bytes/record ≈ 184 MB, negligible.
    """
    # Pass 1: index by language so we can iterate in language order.
    by_lang: dict[str, list[tuple[str, str, str]]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("id")
            if not cid or cid in skip:
                continue
            text = rec.get("text") or ""
            if not text:
                continue
            lang = rec.get("language", "de_ch")
            by_lang.setdefault(lang, []).append((cid, text, lang))
    # Pass 2: yield language-by-language. Order languages by size
    # descending so the biggest cache-hit wins compound first.
    for lang in sorted(by_lang, key=lambda l: -len(by_lang[l])):
        yield from by_lang[lang]


def _label_batch(client: NemotronClient, batch: list[tuple[str, str, str]],
                 max_new_tokens: int) -> list[dict]:
    """One vLLM call for ``len(batch)`` chunks.

    We tried enabling xgrammar via
    ``StructuredOutputsParams(json=SCHEMA)`` (the same short-circuit
    the Gemma labeller uses, where it gave a ~10x speedup) but on
    Nemotron the FSM-state overhead in vLLM's xgrammar backend
    actually slowed us down (~8.2 chunks/s with xgrammar vs ~9.2/s
    without). Suspect: Nemotron's tokenizer vocab is much larger
    than Gemma's, so the per-token grammar mask is dominant.

    Free decoding + downstream verify_and_locate is what wins here.
    """
    from vllm import SamplingParams

    client._load()
    msgs_per_chunk = [
        build_messages(text, language=lang)  # type: ignore[arg-type]
        for _, text, lang in batch
    ]
    # Nemotron's default chat template ends with `<think>\n` (same as
    # Qwen3 family). With thinking enabled the model produces ~500
    # tokens of reasoning before any JSON — at our 16-chunk smoke
    # rate that worked out to 0.2 chunks/s, i.e. ~133 days for the
    # full 2.3M corpus. Not viable.
    #
    # For Qwen we accepted the slowdown because the chain-of-thought
    # noticeably improved labelling quality. For Nemotron's third
    # signal we explicitly trade quality for tractable wall-time:
    # disabling thinking lets the model emit JSON immediately and
    # gets us 30-50× throughput. Majority-vote across Gemma + Qwen
    # + Nemotron absorbs any per-labeller quality variance.
    rendered = [
        client._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for msgs in msgs_per_chunk
    ]
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    outputs = client._llm.generate(rendered, params)
    out: list[dict] = []
    for (cid, text, lang), o in zip(batch, outputs, strict=True):
        raw = o.outputs[0].text
        claims = _parse_json(raw)
        spans = _verify_and_locate(text, claims)
        out.append({
            "id": cid,
            "language": lang,
            "nemotron_spans": [
                {"value": text[sp.start:sp.end], "label": sp.label}
                for sp in spans
            ],
            "nemotron_raw": raw,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, stop after labelling this many chunks "
                         "(used for smoke tests).")
    args = ap.parse_args()

    done = _already_done(args.out_path)
    print(f"Resuming: {len(done):,} chunks already labelled in {args.out_path}",
          flush=True)
    print(f"Loading model {DEFAULT_MODEL} (tp=2, FP8)…", flush=True)
    client = NemotronClient()

    n_processed = 0
    t0 = time.monotonic()
    batch: list[tuple[str, str, str]] = []
    with args.out_path.open("a") as fh:
        for cid, text, lang in _stream_inputs(args.in_path, done):
            batch.append((cid, text, lang))
            if len(batch) < args.batch_size:
                continue
            recs = _label_batch(client, batch, args.max_new_tokens)
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_processed += len(batch)
            elapsed = time.monotonic() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0.0
            eta = (len(done) + len(_stream_inputs.__code__.co_varnames)) if False else 0  # placeholder
            print(f"  {n_processed:>7,} labelled  ({rate:.1f}/s)", flush=True)
            batch = []
            if args.limit is not None and n_processed >= args.limit:
                break
        if batch and (args.limit is None or n_processed < args.limit):
            recs = _label_batch(client, batch, args.max_new_tokens)
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_processed += len(batch)

    total = time.monotonic() - t0
    print(f"\nDone. Labelled {n_processed:,} chunks in {total/60:.1f}min "
          f"({n_processed/total:.1f}/s).", flush=True)


if __name__ == "__main__":
    main()
