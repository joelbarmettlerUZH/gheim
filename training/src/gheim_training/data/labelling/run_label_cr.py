"""V2-4: re-label commercial-register chunks with a format-aware prompt.

The standard v1 Gemma pass under-recalls director and board-member
names in Swiss commercial-register listings (the P7 audit found 156
person-FPs of this type, see paper §3.5 pattern 3). This runner:

1. Streams the v1 chunk corpus (``data/layer5v4_lang_fix.jsonl``).
2. Filters to CR chunks using
   :func:`gheim_training.data.labelling.cr_identify.is_commercial_register`.
3. Re-labels each filtered chunk with Qwen 3.6 35B-A3B using the
   CR-aware prompt
   (:func:`gheim_training.data.labelling.cr_prompts.build_cr_messages`),
   which calls out the contact-block format and emphasises that ALL
   named persons should be flagged.
4. Writes the per-chunk Qwen output to
   ``data/layer5v4_qwen_cr.jsonl`` in the same schema as the main
   ``run_label_v2`` runner.

The output is consumed by the v2 build as a *targeted* second-labeller
signal (in addition to the full-corpus Qwen pass), specifically
boosting recall on CR-format chunks.

Order of operations
-------------------
This runner needs the GPUs free; **launch it AFTER the main
``run_label_v2`` pass completes** (or before it starts). Running both
simultaneously will exhaust the 2× RTX 4090 setup and crash one with
an OOM.

Run
---
    uv run python -m gheim_training.data.labelling.run_label_cr \\
        --in data/layer5v4_lang_fix.jsonl \\
        --out data/layer5v4_qwen_cr.jsonl \\
        --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..labelling.gemma.labeler import SCHEMA, _parse_json, _verify_and_locate
from ..labelling.qwen.client import DEFAULT_MODEL, QwenClient
from ..schema import Language
from .cr_identify import is_commercial_register
from .cr_prompts import build_cr_messages

DEFAULT_IN = Path("data/layer5v4_lang_fix.jsonl")
DEFAULT_OUT = Path("data/layer5v4_qwen_cr.jsonl")
DEFAULT_BATCH = 256
DEFAULT_MAX_NEW_TOKENS = 512


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


def _stream_cr_chunks(path: Path, skip: set[str]):
    """Yield (id, text, language) for CR-format chunks not yet labelled."""
    n_total = 0
    n_cr = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            cid = rec.get("id")
            if not cid or cid in skip:
                continue
            text = rec.get("text") or ""
            if not text:
                continue
            lang = rec.get("language", "de_ch")
            if not is_commercial_register(text, lang):
                continue
            n_cr += 1
            yield cid, text, lang
    print(f"  streamed {n_total:,} chunks, {n_cr:,} matched CR pattern",
          flush=True)


def _label_batch(client: QwenClient, batch: list[tuple[str, str, str]],
                 max_new_tokens: int) -> list[dict]:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    client._load()
    msgs_per_chunk = [
        build_cr_messages(text, language=lang)  # type: ignore[arg-type]
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
            "qwen_cr_spans": [
                {"value": text[sp.start:sp.end], "label": sp.label}
                for sp in spans
            ],
            "qwen_cr_raw": raw,
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
    print(f"Resuming: {len(done):,} CR chunks already labelled in {args.out_path}",
          flush=True)
    print(f"Loading model {DEFAULT_MODEL} (tp=2)…", flush=True)
    client = QwenClient()

    n_processed = 0
    t0 = time.monotonic()
    batch: list[tuple[str, str, str]] = []
    with args.out_path.open("a") as fh:
        for cid, text, lang in _stream_cr_chunks(args.in_path, done):
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
            print(f"  {n_processed:>7,} CR chunks labelled  ({rate:.1f}/s)",
                  flush=True)
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
    print(f"\nDone. Re-labelled {n_processed:,} CR chunks in "
          f"{total/60:.1f}min.", flush=True)


if __name__ == "__main__":
    main()
