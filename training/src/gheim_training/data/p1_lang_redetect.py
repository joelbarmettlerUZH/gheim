"""P1b: Re-run language detection on layer5v4, recording honest fasttext output.

Bug observed: fasttext predicts ``rus_Cyrl`` (or other non-target) at
high confidence on some chunks; previous ``detect()`` then fell back to
``_subset_default("fineweb") == "de_ch"``, mislabeling Russian as
de_ch.

Policy here: record the *honest* fasttext label per chunk, no remapping
or default fallback. Languages in our 5-target set use gheim codes
(de_ch/fr_ch/it_ch/rm/en); everything else keeps its raw NLLB label
(rus_Cyrl, pol_Latn, spa_Latn, ...) so we can later filter at the
balancing step. Short / low-confidence text gets marked UNK_*.

We do NOT delete or remap here — old language is preserved as
``language_old``; new label goes into ``language``; raw fasttext
verdict + confidence go into ``language_raw`` / ``language_confidence``.

Inputs:  data/layer5v4_regex_aug.jsonl   (preferred — has regex spans)
                                         (falls back to layer5v4.jsonl)
Output:  data/layer5v4_lang_fix.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .lang_detect import _NLLB_TO_GHEIM, _MIN_CONFIDENCE, _model, _patch_numpy_for_fasttext


IN_PRIMARY = Path("data/layer5v4_regex_aug.jsonl")
IN_FALLBACK = Path("data/layer5v4.jsonl")
OUT_PATH = Path("data/layer5v4_lang_fix.jsonl")
TARGET_LANGS = set(_NLLB_TO_GHEIM.values())


def _redetect(text: str) -> tuple[str, str | None, float | None]:
    """Return (gheim_lang_or_raw, raw_nllb_label, confidence).

    - Target lang (de/fr/it/rm/en) at confidence ≥ MIN: gheim code,
      raw NLLB label, conf.
    - Confident non-target (Russian/Polish/...): raw NLLB label as the
      language (no OTHER prefix), raw NLLB label again, conf.
    - Low-confidence: "UNK_LOWCONF" + raw label + conf.
    - Too short: "UNK_SHORT" + None + None.
    - No prediction: "UNK_NONE" + None + None.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) < 20:
        return ("UNK_SHORT", None, None)
    labels, scores = _model().predict(cleaned, k=1)
    if not labels:
        return ("UNK_NONE", None, None)
    raw = labels[0].removeprefix("__label__")
    conf = float(scores[0])
    if conf < _MIN_CONFIDENCE:
        return ("UNK_LOWCONF", raw, conf)
    if raw in _NLLB_TO_GHEIM:
        return (_NLLB_TO_GHEIM[raw], raw, conf)
    # Honest non-target label (Russian/Polish/Spanish/...).
    return (raw, raw, conf)


def main() -> None:
    _patch_numpy_for_fasttext()
    in_path = IN_PRIMARY if IN_PRIMARY.exists() else IN_FALLBACK
    print(f"Reading: {in_path}")
    print(f"Writing: {OUT_PATH}")
    print()

    transitions: Counter[tuple[str, str]] = Counter()
    new_lang_counts: Counter[str] = Counter()
    n = 0

    with in_path.open() as fin, OUT_PATH.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            n += 1
            old_lang = rec.get("language", "")
            new_lang, raw, conf = _redetect(rec["text"])
            transitions[(old_lang, new_lang)] += 1
            new_lang_counts[new_lang] += 1
            rec["language_old"] = old_lang
            rec["language"] = new_lang
            if raw is not None:
                rec["language_raw"] = raw
            if conf is not None:
                rec["language_confidence"] = round(conf, 4)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 100_000 == 0:
                print(f"  scanned {n:,} chunks", flush=True)

    print()
    print(f"DONE — {n:,} chunks rewritten")
    print()
    print("New language distribution:")
    for lang, c in sorted(new_lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang:<24} {c:>10,}  ({100*c/n:>5.2f}%)")
    print()
    print(f"Top 30 transitions (old → new, count):")
    for (old, new), c in transitions.most_common(30):
        marker = " *" if old != new else ""
        print(f"  {old:<10} → {new:<24} {c:>10,}{marker}")


if __name__ == "__main__":
    main()
