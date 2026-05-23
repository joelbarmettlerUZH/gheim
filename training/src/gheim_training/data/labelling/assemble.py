"""V2-6: assemble the v2 dataset from four signal sources.

Inputs (one record per chunk in each file, indexed by ``id``):

  - ``data/layer5v4.jsonl``          — raw Gemma spans (v1 baseline)
  - ``data/layer5v4_qwen.jsonl``     — Qwen3.6-35B-A3B (thinking ON)
  - ``data/layer5v4_nemotron.jsonl`` — Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
  - regex catalogue (gheim.detectors.composite._find_regex_spans_with_subtype),
    computed on-the-fly from each chunk's text

For each chunk this script:

1. Converts the Gemma + Qwen + Nemotron span lists into ``_RawSpan``
   tuples with values located in the chunk text (drops hallucinated
   values that don't appear verbatim).
2. Runs the regex catalogue on the chunk text, attaching the
   ``regex_subtype`` per match (iban_ch / ahv / vat_che / credit_card /
   swiss_phone / openai_key / github_pat / aws_access_key).
3. Calls :func:`merge_signals` to collapse the four signal lists into
   a deduplicated ``V2Span`` list with per-span ``signals`` + ``confidence``
   + ``regex_subtype``.
4. Writes one ``V2Example`` per chunk to ``data/assembled.jsonl``,
   tracking which labellers actually saw the chunk.

Memory: the two LLM files (~2-3 GiB each) are indexed into RAM at
startup. That's ~5 GiB peak for the indexes plus a small streaming
overhead — fine on the dev box.

Run
---
    uv run python -m gheim_training.data.labelling.assemble
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path

from gheim.detectors.composite import _find_regex_spans_with_subtype

from .merge import _RawSpan, from_value_pair, merge_signals
from .schema import V2Example, V2Span, write_jsonl

DEFAULT_GEMMA = Path("data/layer5v4.jsonl")
DEFAULT_QWEN = Path("data/layer5v4_qwen.jsonl")
DEFAULT_NEMOTRON = Path("data/layer5v4_nemotron.jsonl")
DEFAULT_OUT = Path("data/assembled.jsonl")


def _load_spans_index(path: Path, spans_key: str) -> dict[str, list[dict]]:
    """Read a labeller's JSONL into {id: spans} for fast lookup.

    ``spans_key`` is the per-record field holding the span list
    (``"qwen_spans"`` for Qwen, ``"nemotron_spans"`` for Nemotron,
    ``"spans"`` for Gemma's raw output)."""
    out: dict[str, list[dict]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("id")
            if cid is None:
                continue
            out[cid] = rec.get(spans_key, []) or []
    return out


def _to_raw_spans(text: str, spans: Iterable[dict]) -> list[_RawSpan]:
    """Convert ``[{"value":..., "label":...}]`` claims to ``_RawSpan``,
    dropping values that don't appear verbatim in the chunk text."""
    out: list[_RawSpan] = []
    for sp in spans:
        v = sp.get("value")
        lbl = sp.get("label")
        if not isinstance(v, str) or not isinstance(lbl, str):
            continue
        raw = from_value_pair(text, v, lbl)
        if raw is not None:
            out.append(raw)
    return out


def assemble(
    gemma_path: Path = DEFAULT_GEMMA,
    qwen_path: Path = DEFAULT_QWEN,
    nemotron_path: Path = DEFAULT_NEMOTRON,
    out_path: Path = DEFAULT_OUT,
    *,
    limit: int | None = None,
) -> dict:
    """Build the v2 dataset and return a summary dict."""
    print(f"Indexing Qwen ({qwen_path})…", flush=True)
    qwen_index = _load_spans_index(qwen_path, "qwen_spans")
    print(f"  {len(qwen_index):,} chunks indexed", flush=True)

    print(f"Indexing Nemotron ({nemotron_path})…", flush=True)
    nemotron_index = _load_spans_index(nemotron_path, "nemotron_spans")
    print(f"  {len(nemotron_index):,} chunks indexed", flush=True)

    # Track which labellers actually saw each chunk so the V2Example
    # records the correct n_candidate_signals (3 if Nemotron didn't
    # cover this chunk, otherwise 4 — see merge_signals).
    n_chunks = 0
    n_skipped_no_qwen = 0
    n_skipped_no_nemotron = 0
    span_counts = {"gemma": 0, "qwen": 0, "nemotron": 0, "regex": 0}
    out: list[V2Example] = []

    t0 = time.monotonic()
    with gemma_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["id"]
            text = rec.get("text") or ""
            if not text:
                continue
            n_chunks += 1

            gemma_spans = _to_raw_spans(text, rec.get("spans") or [])
            qwen_claims = qwen_index.get(cid)
            if qwen_claims is None:
                n_skipped_no_qwen += 1
                qwen_spans: list[_RawSpan] = []
            else:
                qwen_spans = _to_raw_spans(text, qwen_claims)

            nemo_claims = nemotron_index.get(cid)
            if nemo_claims is None:
                n_skipped_no_nemotron += 1
                nemo_spans: list[_RawSpan] = []
            else:
                nemo_spans = _to_raw_spans(text, nemo_claims)

            regex_spans = [
                _RawSpan(start=sp.start, end=sp.end, label=sp.label,
                         value=text[sp.start:sp.end], regex_subtype=sub)
                for sp, sub in _find_regex_spans_with_subtype(text)
            ]

            span_counts["gemma"] += len(gemma_spans)
            span_counts["qwen"] += len(qwen_spans)
            span_counts["nemotron"] += len(nemo_spans)
            span_counts["regex"] += len(regex_spans)

            # n_candidate_signals reflects how many of the four
            # labellers actually saw this chunk. If Nemotron's
            # coverage is partial, the denominator drops to 3 for
            # that chunk (otherwise spans Gemma+Qwen agreed on would
            # be capped at 2/4=0.5 confidence even though they
            # represent unanimous coverage).
            n_signals = 2  # gemma + regex always run
            if qwen_claims is not None:
                n_signals += 1
            if nemo_claims is not None:
                n_signals += 1

            v2_spans = merge_signals(
                text,
                gemma=gemma_spans,
                qwen=qwen_spans,
                nemotron=nemo_spans,
                regex=regex_spans,
                n_candidate_signals=n_signals,
            )

            ex = V2Example(
                id=cid,
                text=text,
                language=rec.get("language", "?"),
                subset=rec.get("subset", "?"),
                doc_id=rec.get("doc_id", ""),
                spans=v2_spans,
                labelers=[
                    "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
                    *(["cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"]
                      if qwen_claims is not None else []),
                    *(["nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"]
                      if nemo_claims is not None else []),
                    "regex+checksum:gheim.detectors.composite._find_regex_spans",
                ],
                meta={
                    "n_candidate_signals": n_signals,
                    "synthetic": False,
                },
            )
            out.append(ex)

            if limit is not None and n_chunks >= limit:
                break

            if n_chunks % 50_000 == 0:
                elapsed = time.monotonic() - t0
                rate = n_chunks / elapsed if elapsed > 0 else 0.0
                print(f"  assembled {n_chunks:,} chunks  ({rate:.0f}/s)",
                      flush=True)

    print(f"\nWriting {len(out):,} V2Examples to {out_path}…", flush=True)
    write_jsonl(out_path, out)

    summary = {
        "n_chunks": n_chunks,
        "n_skipped_no_qwen": n_skipped_no_qwen,
        "n_skipped_no_nemotron": n_skipped_no_nemotron,
        "spans_emitted": sum(len(ex.spans) for ex in out),
        "spans_per_source": span_counts,
        "qwen_coverage_pct": 100 * (n_chunks - n_skipped_no_qwen) / n_chunks,
        "nemotron_coverage_pct": 100 * (n_chunks - n_skipped_no_nemotron) / n_chunks,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", type=Path, default=DEFAULT_GEMMA)
    ap.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    ap.add_argument("--nemotron", type=Path, default=DEFAULT_NEMOTRON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only assemble this many chunks "
                         "(for smoke tests).")
    args = ap.parse_args()

    summary = assemble(
        gemma_path=args.gemma,
        qwen_path=args.qwen,
        nemotron_path=args.nemotron,
        out_path=args.out,
        limit=args.limit,
    )
    print()
    print("=" * 70)
    print("V2 ASSEMBLY SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:.1f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:30s} {vv:,}")
        else:
            print(f"  {k:30s} {v:,}")


if __name__ == "__main__":
    main()
