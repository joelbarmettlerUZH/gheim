# gheim-1: Swiss-market PII detector

**Hub IDs**:
- `joelbarmettler/gheim-1` — full model (xlm-roberta-large fine-tune, 559M params)
- `joelbarmettler/gheim-1-composite` — `gheim-1` + deterministic regex
  front-end with checksum validation (IBAN-CH, AHV, VAT-CHE, Luhn, etc.)

**License**: Apache-2.0
**Base model**: [`FacebookAI/xlm-roberta-large`](https://huggingface.co/FacebookAI/xlm-roberta-large) (Apache-2.0)
**Languages**: German (de_CH), French (fr_CH), Italian (it_CH), Romansh (rm),
Swiss German (gsw), and English (regression-checked, not the primary target)
**Task**: Token classification — 33 BIOES tags over 8 PII categories
(`account_number`, `private_address`, `private_date`, `private_email`,
`private_person`, `private_phone`, `private_url`, `secret`)

This card reports **only numbers measured on a held-out test set**
(`test_v1`, 800 chunks / 837 spans, sampled from corpora that were never
seen during training or model selection). No claims beyond what's in
`eval/`.

---

## TL;DR

| Detector | overlap F1 | strict F1 | FP/PII-free chunk |
|---|---|---|---|
| **gheim-1** (xlm-roberta-large fine-tune) | **0.817** | **0.777** | **0.120** |
| gheim-1-composite (gheim-1 + regex front-end) | 0.814 | 0.774 | 0.120 |
| Baseline: openai/privacy-filter (zero-shot, our base candidate) | 0.415 | 0.343 | 0.440 |
| Baseline: SwissBERT-NER + regex | 0.410 | 0.308 | 2.670 |
| Baseline: Microsoft Presidio + Swiss recognizers | 0.298 | 0.251 | 4.060 |

gheim-1 beats every off-the-shelf alternative by **≥0.40 overlap F1** on
real Swiss text and emits **3.7× fewer FPs on PII-free controls** than the
next-best alternative (zero-shot openai/privacy-filter).

---

## What changed vs the previous best baseline

The original CLAUDE.md plan started from `openai/privacy-filter` (1.4B
MoE). Our bake-off (`eval/bakeoff_matrix.json`) showed that `xlm-roberta-large`
(550M dense) substantially outperforms it on this dataset:

| Base model | overlap F1 (test_v1) | strict F1 | params |
|---|---|---|---|
| **xlm-roberta-large** | **0.817** | **0.777** | 559M dense |
| ZurichNLP/swissbert | 0.770 | 0.735 | 270M XMOD |
| openai/privacy-filter | 0.619 | 0.475 | 1.4B MoE |
| Zero-shot privacy-filter | 0.415 | 0.343 | 1.4B MoE |

The hyperparam sweep (`eval/sweep_matrix.json`, 6 runs varying LR / batch /
epochs / freeze) didn't move the needle — variance ≤ 0.02 F1 around the
default. We ship the bake-off baseline (`bakeoff-xlmr`) as `gheim-1`.

---

## Per-language production readiness (gheim-1, overlap F1, Wilson 95% CI)

| Language | overlap F1 | 95% CI half-width | n_gold | readiness |
|---|---|---|---|---|
| de_CH | 0.852 | ±0.048 | 210 | ✓ deploy with monitoring |
| fr_CH | 0.841 | ±0.054 | 176 | ✓ deploy with monitoring |
| it_CH | 0.774 | ±0.066 | 151 | ⚠️ marginal — review high-risk surfaces |
| rm    | 0.796 | ±0.045 | 300 | ⚠️ marginal — review high-risk surfaces |

Readiness thresholds (defined in `eval/day15_report.py`):
- ✅ ≥0.90 — ship-ready (high confidence)
- ✓ ≥0.80 — acceptable (deploy with monitoring)
- ⚠️ ≥0.70 — marginal (humans-in-loop recommended)
- ✗ <0.70  — not ready

**gsw (Swiss German) is unmeasured on test_v1** — the held-out corpora
(Apertus shard 4 + FineWeb-2-Swiss filtered tail) yielded no gsw chunks
that survived language detection. Treat gsw deployment as experimental;
the only gsw signal in training was 858 examples from Layer 3 (Apertus
slot-filling).

## Per-(language × category) breakdown (gheim-1-composite, overlap F1)

|  | de_ch | fr_ch | it_ch | rm |
|---|---|---|---|---|
| private_person | 0.86 (n=158) | 0.88 (n=113) | 0.75 (n=104) | 0.84 (n=282) |
| private_date   | 0.84 (n=38) | 0.82 (n=45) | 0.83 (n=42) | **0.31 (n=15)** |
| private_url    | 0.75 (n=10) | **0.59 (n=12)** | **0.67 (n=4)** | **0.00 (n=3)** |
| private_address| 0.80 (n=2)  | 0.75 (n=3)  | n/a (n=0)  | n/a (n=0)  |
| private_phone  | 0.80 (n=2)  | **0.50 (n=3)** | 1.00 (n=1) | n/a (n=0)  |
| account_number, private_email, secret | n=0 across all languages on test_v1 | | | |

These low-count cells (n<20) come with very wide CIs (Wilson half-width
≥0.20). Categories with no test_v1 gold spans were primarily evaluated
during synthetic Layer 1 validation (Faker_CH; perfect P/R by construction
on those).

## Known weaknesses surfaced by this eval

1. **Romansh date strings** (private_date F1 = 0.31). RM date phrasing
   ("ils 12 da fanadur 2013") is rare in training. Prefer the regex
   front-end (composite) for date detection on RM text — the regex catches
   numeric forms (DD.MM.YYYY) that the model misses.
2. **URL detection in fr_CH/it_CH/rm** (overlap F1 0.0–0.67). The model
   inconsistently flags free-text URLs in non-DE languages. Composite
   patches the high-confidence shapes (`https://…`) but doesn't fix
   plaintext domain references.
3. **it_CH overall** (0.774). Italian Switzerland is the smallest language
   slice of the corpus (Layer 5 + 6 contributed ~3k examples). Recall on
   Italian person names is the dominant error mode.
4. **Address eval is undersampled on real text** (n=5 on test_v1). The
   prod-readiness claim for private_address rests on `address_gold_v1.jsonl`
   (75 spans, F1 = 0.428 overlap) and synthetic Layer 1 (high but
   construction-bound). Address detection on free-form Swiss prose remains
   the lowest-confidence category.

## English regression (held-out, no leak)

Measured on 1,000 records sampled from `ai4privacy/pii-masking-300k`'s
**`validation` split**. Layer 4 was sourced exclusively from the `train`
split, so `validation` was never seen during training. Defensive
hash-collision check against `data/layer4_en.jsonl`: **0 collisions** out
of 7,946 candidate English validation records — the splits are cleanly
disjoint.

| Detector | overlap F1 | strict F1 | Δ vs base (overlap) |
|---|---|---|---|
| **Base openai/privacy-filter (zero-shot)** | **0.873** | **0.823** | — |
| gheim-1 (bakeoff_xlmr) | 0.583 | 0.531 | **−0.291** |
| gheim-1-composite | 0.598 | 0.545 | −0.275 |

**gheim-1 regresses substantially on English** (−29pp vs base). This is
24pp worse than the CLAUDE.md "within 5pp" non-regression target.

### Root cause: AI4Privacy label-taxonomy gap in our training data

While building this eval we discovered our REMAP in `ai4privacy.py` was
written against AI4Privacy v1 label names (`LASTNAME, BUILDINGNUMBER,
ZIPCODE, PHONE_NUMBER, IPV4, SOCIALNUM, DOB, PASSWORD, …`). The actual
`pii-masking-300k` dataset emits v2 names (`LASTNAME, BUILDING, POSTCODE,
TEL, IP, SOCIALNUMBER, BOD, PASS, …`) plus categories with no v1 analog
(`PASSPORT, DRIVERLICENSE, IDCARD, COUNTRY`). The unmapped names — about
half the labels by token count — were silently dropped during build
(`ai4privacy.py:127`: `cat is None → continue`).

Concretely, the model was **trained on labels for** person, email, city,
state, street, date, and (some) account-numbers — but trained to **predict
O** for `TEL/IP/PASSPORT/DRIVERLICENSE/IDCARD/SOCIALNUMBER/BUILDING/`
`POSTCODE/SECADDRESS/COUNTRY/BOD/PASS`. Layer 4 (English) and Layer 2
(de/fr/it from the same dataset) are both affected.

This was masked by our prior eval (`english_regression.py`) which used
the same broken REMAP for both train and "test" gold, making both sides
agree to ignore those categories. The corrected eval applies a v2 REMAP
to validation gold (`english_regression_clean.py`) and exposes the gap.

### Implications

- Base privacy-filter is the right choice for English deployment if PII
  categories beyond person/address/date/email matter.
- **gheim-1 should not be deployed for English-PII detection** until
  Layer 2 + Layer 4 are rebuilt with the v2 REMAP and the model is
  re-trained. We expect this to recover most of the 29pp gap because
  the base has clearly proven the schema is learnable on this data.
- Swiss-language numbers (test_v1, address_gold_v1) are not affected
  because they were labeled by us, not AI4Privacy. The bake-off and
  prod-readiness numbers above stand.

Reproduce:
```bash
uv run python -m gheim_training.eval.english_regression_clean \
    --out eval/english_regression_clean.json --n 1000
```

## Composite vs bare model

The composite detector (`packages/gheim-py/src/gheim/detectors/composite.py`)
runs a deterministic regex front-end with checksum validation
(IBAN-CH mod-97-10, AHV EAN-13 mod-10, VAT-CHE mod-11-10, Visa/MC Luhn,
phone, email, URL, explicit dates) before the model. On `test_v1` overall
F1, composite ties bare gheim-1 (0.814 vs 0.817 — within noise). The
composite's value is **not** F1 — it's:
- Deterministic recall on structured PII regardless of model behaviour
- Checksum-failed candidates fall through to the model rather than being
  silently labeled wrong
- Auditability: every regex match is traceable to a single, testable rule

For most production deployments, ship the composite. For benchmarking or
research, the bare model is fine.

## Reproduction

```bash
git clone https://github.com/<org>/gheim
cd gheim
git checkout gheim-1-bakeoff
uv sync
# Pull data layers (~1.2GB) — see training/README.md
make eval-day15           # reproduces the headline numbers
```

Full eval matrix in `eval/bakeoff_matrix.json`,
`eval/sweep_matrix.json`, `eval/ablation_matrix.json`,
`eval/day15_report.json`.

## Training data

| Layer | Source | Examples | Languages |
|---|---|---|---|
| 1 | Faker_CH + handwritten templates (real Swiss addresses via Geonames CH) | 50,000 | DE/FR/IT-CH |
| 2 | AI4Privacy `pii-masking-300k` de/fr/it | 60,000 | DE/FR/IT |
| 3 | Apertus-8B slot-filled (verbatim) | 7,000 | DE/FR/IT/RM/GSW |
| 4 | English anchor (AI4Privacy en) | 20,000 | EN |
| 5 | Apertus-labeled real Swiss text (Entscheidsuche + Curia Vista) | 31,000 | DE/FR/IT |
| 6 | Apertus-labeled Romansh PII-rich (`apertus-pretrain-romansh`) | 15,000 | RM |
| 7 | Hand-labeled hard examples (5× upweighted) | 1,500 | DE/FR/IT |
| 8 | Gemma + Qwen consensus labels | 180,000 | DE/FR/IT/RM |

Layer 7 ablation (`eval/ablation_matrix.json`) showed Layer 7 contributed
−0.004 F1 (within noise). Layer 8 ablation showed Layer 8 was load-bearing:
removing it dropped test_v1 overall F1 by 0.17. The 5×-upweight on Layer 7
did not justify its labeling cost; future iterations should drop it.

## Citation

```
@misc{gheim2026swiss,
  title  = {gheim-1: Swiss-market PII detector},
  author = {Joel Barmettler},
  year   = {2026},
  url    = {https://huggingface.co/joelbarmettler/gheim-1},
}
```

Underlying corpora: AI4Privacy `pii-masking-300k` (Apache-2.0),
`swiss-ai/apertus-pretrain-swiss` (per-subset; Entscheidsuche court
decisions are public, Curia Vista parliamentary records are public),
`swiss-ai/apertus-pretrain-romansh` (CC-BY-4.0 — attribution required).

## Limitations and intended use

**Intended use**: pre-publication / pre-storage screening of Swiss-language
free text for PII categories above. Designed as a recall-favouring detector
for human-in-the-loop redaction workflows.

**Out of scope**:
- Legal compliance certification (this is a tool, not a guarantee)
- Detecting PII outside the 8 listed categories (e.g. medical record IDs,
  Swiss vehicle plates, AHV variants beyond the standard format)
- **English PII detection** — confirmed regression vs base; use
  `openai/privacy-filter` for English until our Layer 2/4 REMAP gap is
  fixed (see "English regression" above)
- Production deployment on `gsw` (Swiss German) — unmeasured
- Adversarial inputs designed to evade detection
