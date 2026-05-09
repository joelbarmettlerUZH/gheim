"""Day 15 English regression check — CLEAN version (no leak).

The earlier ``english_regression.py`` tail-sliced ``data/layer4_en.jsonl``,
which is contaminated because ``english_anchor.load()`` shuffles records
during build. This module uses the AI4Privacy ``validation`` split, which
``english_anchor.load(split='train')`` never touched.

Defensive double-check: hash every record's text and drop any that also
appear in ``data/layer4_en.jsonl`` (should be ~0 collisions because the
upstream split is disjoint from train, but we verify and report).

Run:
    uv run python -m gheim_training.eval.english_regression_clean \\
        --out eval/english_regression_clean.json --n 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..data.ai4privacy import REMAP as _LEGACY_REMAP
from ..data.schema import Example, Language, Span
from .baselines.zero_shot import HFDetector
from .harness import _fmt_f1, evaluate

# AI4Privacy's actual v2 label taxonomy (observed in train+validation),
# extending the legacy REMAP to cover labels that pii-masking-300k actually
# emits but our training-time REMAP drops. Applied ONLY for honest eval —
# training data still uses the legacy REMAP, so gheim-1 was never taught
# half these categories. This is exactly why this eval matters.
_REMAP_V2: dict[str, str | None] = {
    **_LEGACY_REMAP,
    "TEL": "private_phone",
    "IP": "private_url",
    "BUILDING": "private_address",
    "POSTCODE": "private_address",
    "SECADDRESS": "private_address",
    "COUNTRY": "private_address",
    "BOD": "private_date",
    "SOCIALNUMBER": "account_number",
    "IDCARD": "account_number",
    "PASSPORT": "account_number",
    "DRIVERLICENSE": "account_number",
    "PASS": "secret",
    # Explicit drops (no analog in our 8 categories)
    "TITLE": None,
    "SEX": None,
    "GEOCOORD": None,
    "CARDISSUER": None,
    "TIME": None,
}


def _spans_from_record_v2(record: dict, language: str) -> Example | None:
    """Same shape as ai4privacy._spans_from_record but applies the corrected
    v2 REMAP, so the gold reflects what AI4Privacy actually labels."""
    text = record.get("source_text") or record.get("unmasked_text") or record.get("text")
    mask = record.get("privacy_mask") or []
    if not text or not mask or not isinstance(mask, list):
        return None
    spans: list[Span] = []
    for m in mask:
        lab = m.get("label") or m.get("entity_type") or m.get("entity")
        s = m.get("start") if "start" in m else m.get("start_position")
        e = m.get("end") if "end" in m else m.get("end_position")
        if lab is None or s is None or e is None:
            continue
        norm = lab.upper().rstrip("0123456789")
        cat = _REMAP_V2.get(norm)
        if cat is None:
            continue
        if e <= s or e > len(text):
            continue
        if not text[s:e].strip():
            continue
        spans.append(Span(start=int(s), end=int(e), label=cat))
    if not spans:
        return None
    spans.sort(key=lambda sp: (sp.start, -sp.end))
    deduped: list[Span] = []
    last_end = -1
    for sp in spans:
        if sp.start < last_end:
            continue
        deduped.append(sp)
        last_end = sp.end
    if not deduped:
        return None
    from typing import cast
    ex = Example(text=text, spans=deduped, language=cast(Language, language),
                 source="ai4privacy")
    try:
        ex.validate_offsets()
    except ValueError:
        return None
    return ex


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_train_hashes(layer4_path: Path) -> set[str]:
    """Hash every text in layer4_en.jsonl — the training set we want to
    avoid scoring on."""
    out: set[str] = set()
    for line in layer4_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = rec.get("text") or ""
        out.add(_hash(text))
    return out


def _load_validation_held_out(n: int, train_hashes: set[str]) -> tuple[list[Example], int]:
    """Pull English records from AI4Privacy's `validation` split (untouched
    by training) and exclude any that also appear in `layer4_en.jsonl` by
    hash. Returns (examples, n_collisions_dropped)."""
    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-300k", split="validation")
    out: list[Example] = []
    n_collisions = 0
    for rec in ds:
        loc = rec.get("language") or rec.get("locale")
        if loc != "English":
            continue
        text = rec.get("source_text") or rec.get("text") or ""
        if _hash(text) in train_hashes:
            n_collisions += 1
            continue
        ex = _spans_from_record_v2(rec, language="en")
        if ex is None:
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out, n_collisions


def _to_eval_cases(examples: list[Example]) -> list[dict[str, Any]]:
    """Convert Example objects into the dict shape evaluate() expects."""
    return [
        {
            "text": ex.text,
            "lang": ex.language,
            "spans": [{"start": s.start, "end": s.end, "label": s.label}
                      for s in ex.spans],
        }
        for ex in examples
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer4", type=Path, default=Path("data/layer4_en.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("eval/english_regression_clean.json"))
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    print(f"Loading training hashes from {args.layer4}…")
    train_hashes = _load_train_hashes(args.layer4)
    print(f"  {len(train_hashes):,} training texts hashed")

    print(f"\nPulling up to {args.n} held-out English records from "
          f"ai4privacy/pii-masking-300k[validation]…")
    examples, n_collisions = _load_validation_held_out(args.n, train_hashes)
    print(f"  loaded {len(examples)} held-out examples "
          f"({n_collisions} collisions with train dropped)")

    cases = _to_eval_cases(examples)

    detectors: dict[str, Callable[[], Any]] = {
        "base_priv_filter": lambda: HFDetector("openai/privacy-filter"),
        "bakeoff_xlmr":     lambda: HFDetector("checkpoints/bakeoff-xlmr"),
        "composite_xlmr":   _make_composite,
    }

    results: dict[str, dict[str, Any]] = {}
    for name, factory in detectors.items():
        print(f"\n=== {name} ===")
        det = factory()
        t0 = time.time()
        rep = evaluate(det, cases)
        elapsed = time.time() - t0
        results[name] = rep
        print(f"  strict  F1 = {_fmt_f1(rep['strict']['overall']['f1'])}")
        print(f"  overlap F1 = {_fmt_f1(rep['overlap']['overall']['f1'])}  ({elapsed:.0f}s)")

    base_f1 = results["base_priv_filter"]["overlap"]["overall"]["f1"] or 0
    print("\n=== Δ vs base openai/privacy-filter (overlap F1) ===")
    print(f"  base                {base_f1:.3f}")
    for name in ("bakeoff_xlmr", "composite_xlmr"):
        f1 = results[name]["overlap"]["overall"]["f1"] or 0
        delta = f1 - base_f1
        symbol = "✓" if abs(delta) <= 0.05 else "⚠️" if delta > -0.10 else "✗"
        print(f"  {name:<22} {f1:.3f}  Δ={delta:+.3f}  {symbol}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_chunks": len(cases),
        "n_collisions_dropped": n_collisions,
        "source": "ai4privacy/pii-masking-300k[validation]",
        "leak_check": {
            "method": "sha256(text)[:16] of every layer4_en.jsonl record; "
                      "validation records whose hash matches any are dropped",
            "training_hashes_loaded": len(train_hashes),
            "validation_records_dropped_for_collision": n_collisions,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote → {args.out}")


def _make_composite() -> Any:
    from gheim.detectors.composite import CompositeDetector
    return CompositeDetector(model=HFDetector("checkpoints/bakeoff-xlmr"))


if __name__ == "__main__":
    main()
