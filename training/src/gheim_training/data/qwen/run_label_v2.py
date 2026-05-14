"""V2-3: re-label the existing v1 chunks with Qwen 3.6 35B-A3B.

Reads the canonical Gemma-labelled corpus
(``data/layer5v4_lang_fix.jsonl``, ~2.3M chunks) and applies the same
per-language PII prompt with Qwen instead, so each chunk has both a
``gemma`` and a ``qwen`` labelling. Combined with the regex layer,
this gives the v2 build three independent signals per span and lets
downstream steps build a high-precision intersection
(``gheim-ch-pii-v2-strict``) and a high-recall union
(``gheim-ch-pii-v2-recall``).

Output schema (one line per labelled chunk, append-mode + resumable):

    {
      "id": "<chunk id>",
      "language": "de_ch | fr_ch | it_ch | rm | en",
      "qwen_spans": [{"value": "...", "label": "..."}, ...],
      "qwen_raw": "<raw model output, kept for forensics>"
    }

Only the chunk id + language are read from the input file (the chunk
text is held in the line we're processing, so we don't need to look up
text from elsewhere).

Run:

    uv run python -m gheim_training.data.qwen.run_label_v2 \\
        --in data/layer5v4_lang_fix.jsonl \\
        --out data/layer5v4_qwen.jsonl \\
        --batch-size 256

The script is resumable: each run skips chunks whose ``id`` already
appears in the output file. For a clean restart, delete the output file
(or pass ``--out`` with a fresh path).

GPU plan: tensor_parallel_size=2 on 2× RTX 4090, AWQ-4bit Qwen at
~17 GB per card, batch=256 chunks per vLLM call. Throughput should be
roughly the same as the original Gemma run (~325k chunks/h), so the
full 2.3M takes ~7 hours of wall time. Launch overnight.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..gemma.labeler import (
    SCHEMA,
    _parse_json,
    _verify_and_locate,
)
from ..gemma.prompts import build_messages
from ..schema import Language
from .client import DEFAULT_MODEL, QwenClient

DEFAULT_IN = Path("data/layer5v4_lang_fix.jsonl")
DEFAULT_OUT = Path("data/layer5v4_qwen.jsonl")
DEFAULT_BATCH = 256
DEFAULT_MAX_NEW_TOKENS = 512  # 50+ spans of JSON, plenty


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
    """Yield (id, text, language) for chunks not yet labelled."""
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
            yield cid, text, lang


def _label_batch(client: QwenClient, batch: list[tuple[str, str, str]],
                 max_new_tokens: int) -> list[dict]:
    """One vLLM call for ``len(batch)`` chunks. Returns a list of records."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    client._load()
    msgs_per_chunk = [
        build_messages(text, language=lang)  # type: ignore[arg-type]
        for _, text, lang in batch
    ]
    rendered = [
        client._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        for msgs in msgs_per_chunk
    ]
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        structured_outputs=StructuredOutputsParams(json=SCHEMA),
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
            "qwen_spans": [
                {"value": text[sp.start:sp.end], "label": sp.label}
                for sp in spans
            ],
            "qwen_raw": raw,
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
    print(f"Loading model {DEFAULT_MODEL} (tp=2)…", flush=True)
    client = QwenClient()

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
            print(f"  {n_processed:>7,} labelled  ({rate:.0f}/s)", flush=True)
            batch = []
            if args.limit is not None and n_processed >= args.limit:
                break
        # Tail batch
        if batch and (args.limit is None or n_processed < args.limit):
            recs = _label_batch(client, batch, args.max_new_tokens)
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_processed += len(batch)
    total = time.monotonic() - t0
    print(f"\nDone. Labelled {n_processed:,} chunks in {total/3600:.1f}h "
          f"({n_processed/total:.1f}/s).", flush=True)


if __name__ == "__main__":
    main()
