"""P10: Build the bake-off training mix.

  data/gheim_pii_balanced_train.jsonl  (142k: 117k real Swiss + 24.9k synthetic)
+ data/layer2_v2.jsonl                 (AI4Privacy openpii-1m de/fr/it, REMAPped)
+ data/layer4_v2.jsonl                 (AI4Privacy openpii-1m en, REMAPped)
↓
  data/gheim_train_mix.jsonl

The augmentation layer is intentionally narrow:
  - EN anchor: ``N_EN`` AI4P en chunks (any span content) to keep English F1
    from regressing under fine-tuning.
  - Sparse-cell email rescue: ``N_PER_LANG_EMAIL`` chunks each from
    AI4P de/fr/it filtered to (a) region == "CH" and (b) at least one
    ``private_email`` span. These are the cells where gheim-pii is
    genuinely thin (de_ch 671, fr_ch 1,431, it_ch 310 in train).

Categories deliberately NOT supplemented:
  - private_url, secret  — AI4Privacy openpii-1m has zero of these.
  - account_number  — AI4P "account_number" is national IDs / passports /
    driver-licenses, not Swiss-format IBAN/AHV/VAT. Wrong distribution.
  - private_address — AI4P address spans are fragments (PLZ-only,
    BUILDINGNUM-only) due to schema-collapse; would teach wrong patterns.
  - private_person, private_date — gheim-pii already at cap.
  - rm — AI4P has zero Romansh.

The published dataset (val + test) is unchanged. This file is for training
only and is NOT published.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

GHEIM_TRAIN = Path("data/gheim_pii_balanced_train.jsonl")
AI4P_DEFRIT = Path("data/layer2_v2.jsonl")
AI4P_EN = Path("data/layer4_v2.jsonl")
OUT_PATH = Path("data/gheim_train_mix.jsonl")

# Augment volumes (keep small to preserve Swiss specificity)
N_EN = 8_000              # AI4Privacy en anchor
N_PER_LANG_EMAIL = 2_000  # AI4Privacy CH-region email rescue (de/fr/it)
EMAIL_RESCUE_LANGS = ("de_ch", "fr_ch", "it_ch")

SEED = 42


def _ai4p_to_gheim(rec: dict, idx: int | str) -> dict:
    """Normalize an AI4Privacy record to the gheim-pii schema.

    Adds: id, subset, source_dataset, doc_id, chunk_index_in_doc, value
    (per span), labeler, prompt_version, synthetic flag.
    """
    text = rec["text"]
    meta = rec.get("meta") or {}
    pii_id = meta.get("openpii_1m_id", idx)
    spans_out = []
    for sp in rec.get("spans", []):
        s, e, lab = sp["start"], sp["end"], sp["label"]
        spans_out.append({
            "start": s, "end": e, "label": lab,
            "value": text[s:e], "source": "ai4privacy",
        })
    return {
        "id": f"ai4p#{pii_id}",
        "text": text,
        "language": rec["language"],
        "subset": "ai4privacy",
        "source_dataset": "ai4privacy/pii-masking-openpii-1m",
        "doc_id": f"ai4p_{pii_id}",
        "chunk_index_in_doc": 0,
        "spans": spans_out,
        "labeler": "ai4privacy:openpii-1m",
        "prompt_version": "openpii-1m-v1",
        "labeled_at": None,
        "ai4p_region": meta.get("region"),
        "ai4p_lang_code": meta.get("lang_code"),
        "synthetic": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Actually write the mix file (default: dry-run)")
    args = parser.parse_args()

    rng = random.Random(SEED)

    # ---------- Pass 1: count gheim-pii train ----------
    if not GHEIM_TRAIN.exists():
        raise SystemExit(f"Missing {GHEIM_TRAIN}. Run P5 + P6 + P9 first.")
    print(f"Reading: {GHEIM_TRAIN}", flush=True)
    n_gheim = 0
    n_synth = 0
    gheim_per_lang_cat: Counter[tuple[str, str]] = Counter()
    with GHEIM_TRAIN.open() as f:
        for line in f:
            rec = json.loads(line)
            n_gheim += 1
            if rec.get("synthetic"):
                n_synth += 1
            la = rec.get("language", "")
            cats = {sp.get("label", "") for sp in rec.get("spans", [])}
            for c in cats:
                gheim_per_lang_cat[(la, c)] += 1
    print(f"  {n_gheim:,} chunks ({n_synth:,} synthetic, {n_gheim - n_synth:,} real)")

    # ---------- Pass 2: collect AI4P en candidates ----------
    print(f"Reading: {AI4P_EN}", flush=True)
    en_pool: list[dict] = []
    with AI4P_EN.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("language") != "en":
                continue
            if not rec.get("spans"):
                continue
            en_pool.append(rec)
    print(f"  {len(en_pool):,} en candidates with ≥1 span")

    rng.shuffle(en_pool)
    en_picked = en_pool[:N_EN]
    print(f"  picked {len(en_picked):,} for EN anchor")

    # ---------- Pass 3: collect AI4P email-rescue candidates ----------
    print(f"Reading: {AI4P_DEFRIT}", flush=True)
    rescue_pool: dict[str, list[dict]] = {la: [] for la in EMAIL_RESCUE_LANGS}
    n_seen = 0
    with AI4P_DEFRIT.open() as f:
        for line in f:
            rec = json.loads(line)
            n_seen += 1
            la = rec.get("language", "")
            if la not in EMAIL_RESCUE_LANGS:
                continue
            region = (rec.get("meta") or {}).get("region", "")
            if region != "CH":
                continue
            if not any(sp.get("label") == "private_email" for sp in rec.get("spans", [])):
                continue
            rescue_pool[la].append(rec)
            if n_seen % 50_000 == 0:
                print(f"  scanned {n_seen:,}", flush=True)
    for la, pool in rescue_pool.items():
        print(f"  {la}: {len(pool):,} CH-region candidates with ≥1 email span")

    rescue_picked: dict[str, list[dict]] = {}
    for la, pool in rescue_pool.items():
        rng.shuffle(pool)
        rescue_picked[la] = pool[:N_PER_LANG_EMAIL]
        print(f"  picked {len(rescue_picked[la]):,} {la} for email rescue")

    # ---------- Project final cell counts ----------
    print()
    print("=" * 80)
    print("PROJECTED FINAL TRAIN CELL COUNTS")
    print("=" * 80)
    LANGS = ["de_ch", "fr_ch", "it_ch", "rm", "en"]
    CATS = ["account_number", "private_address", "private_date", "private_email",
            "private_person", "private_phone", "private_url", "secret"]
    augment = Counter(gheim_per_lang_cat)
    for rec in en_picked:
        cats = {sp["label"] for sp in rec["spans"]}
        for c in cats:
            augment[("en", c)] += 1
    for la, picks in rescue_picked.items():
        for rec in picks:
            cats = {sp["label"] for sp in rec["spans"]}
            for c in cats:
                augment[(la, c)] += 1

    print(f"{'lang':<8} " + " ".join(f"{c[:5]:>7}" for c in CATS))
    for la in LANGS:
        row = " ".join(f"{augment.get((la, c), 0):>7,}" for c in CATS)
        print(f"{la:<8} {row}")

    n_total = n_gheim + len(en_picked) + sum(len(p) for p in rescue_picked.values())
    print()
    print(f"Total chunks in mix: {n_total:,}")
    print(f"  gheim-pii train (real + synthetic): {n_gheim:,}")
    print(f"  + AI4P en anchor:                   {len(en_picked):,}")
    for la in EMAIL_RESCUE_LANGS:
        print(f"  + AI4P {la} email rescue:           {len(rescue_picked[la]):,}")

    if not args.write:
        print()
        print(f"[DRY-RUN] no file written. Add --write to produce {OUT_PATH}")
        return

    # ---------- Write ----------
    print()
    print(f"[WRITE] producing {OUT_PATH} ...")
    n_written = 0
    with OUT_PATH.open("w") as fout:
        # 1. gheim-pii train (verbatim)
        with GHEIM_TRAIN.open() as fin:
            for line in fin:
                fout.write(line)
                n_written += 1
        # 2. AI4P en
        for i, rec in enumerate(en_picked):
            normalized = _ai4p_to_gheim(rec, i)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            n_written += 1
        # 3. AI4P email rescue
        for la, picks in rescue_picked.items():
            for j, rec in enumerate(picks):
                normalized = _ai4p_to_gheim(rec, f"{la}_{j}")
                fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                n_written += 1
    print(f"  wrote {n_written:,} chunks")


if __name__ == "__main__":
    main()
