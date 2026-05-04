# test_v1 — held-out test set for the gheim-1 bake-off

This file documents `test_v1_pool.jsonl` (the candidate pool) and the
forthcoming `test_v1.jsonl` (the gold set after Days 4-6 labeling). Both
are gitignored data artefacts; this README is checked in so the
provenance is auditable.

## Why this set exists

The previous "v3 holdout" was used for checkpoint selection across
v2/v3/v4 — by the time v4 was scored on it, it was a dev set, not a
holdout (FEEDBACK.md §4.3). For the bake-off (CLAUDE.md Week 2), we need
a real test set that:

- Has never been touched during model training or selection
- Is large enough that Wilson 95% CI on overall F1 is ≤ ±0.04
- Covers the multilingual Swiss surface (DE-CH / FR-CH / IT-CH / RM)
- Includes PII-free controls so we can measure false-positive rate
- Reports both strict and overlap F1 (overlap is the production-meaningful
  number for redaction; strict measures boundary precision)

## Source

`data/raw/apertus-swiss-holdout/data/001_grouped_data_00001.parquet`
plus `data/raw/apertus-romansh/monolingual/rm_wikipedia_new.jsonl.gz`.

The holdout shard is FineWeb-Swiss web text reserved for held-out
evaluation. Romansh is sparse on FineWeb-Swiss (RM is ~0.5% of CH);
we top up from rm.wikipedia for RM coverage.

Defensive dedup: every chunk's `doc_id` is checked against
`data/prod_inputs.jsonl` (the 318k chunks that fed the prod labeling
pipeline that produced Layers 5-9). Verified zero overlap on the
holdout shard. The Wikipedia source has ~15% prior overlap; those
chunks are skipped.

## Stratification

Sampling is stratified by (language × presence). `presence` is a
prefilter flag: `pii_dense` if at least one structured-PII regex hit
(IBAN-CH, AHV, VAT, email, phone, CC, date, ALL-CAPS surname); `control`
otherwise. Subagents see different prompts for the two — dense chunks
get "find all PII"; controls get "verify there is no PII" (FP measurement).

| language | pii_dense | control | total |
|---|---:|---:|---:|
| de_ch | 150 | 50 | 200 |
| fr_ch | 150 | 50 | 200 |
| it_ch | 150 | 50 | 200 |
| rm    | 150 | 50 | 200 |
| **total** | 600 | 200 | **800** |

## Known limitations

1. **All non-RM content is web text (FineWeb-Swiss).** No legal text in
   test_v1. The legal-text trend line continues to be `audit_gold.jsonl`
   (Entscheidsuche + Curia Vista). Model card must report both.

2. **RM coverage is healthy now (200 chunks, 150 PII-dense)** but
   rm.wikipedia still over-represents historical / encyclopedic content
   relative to the rest of the test set. Per-language RM CI is ≈ ±0.07
   for n=200 — comparable to the other languages.

3. **EN intentionally excluded.** gheim's primary deployment is Swiss;
   English regression is measured separately on a `layer4_en` validation
   slice on Day 15. No EN content in test_v1.

4. **GSW unmeasured.** Same reason as RM — corpus is sparse. Lingua has
   no GSW model; existing GSW would mostly tag as `de_ch`.

## Lang-detect bug found and fixed during build

The lingua wrapper in `data/lang_detect.py` had three false-positive
patterns that surfaced during the test_v1 build, all locked in by
regression tests in `training/tests/test_lang_detect.py`:

- `"che"` was in the RM marker set, but it's the most common word in
  Italian. ~94% of the initial RM bucket was Italian text.
- After dropping `"che"`, `"ils"` (3rd-person FR pronoun) drove all
  remaining RM samples to be French.
- After adding a lingua-first override, real Romansh content was
  misclassified as English: lingua doesn't ship a Romansh model so it
  defaults unfamiliar text to EN at very high confidence.

Fixed in three passes:

1. Dropped ambiguous tokens (`che`, `schi`, `nun`, `ils`, `ina`, `ino`,
   `tar`, `dapi`, `duai`) from `_RM_MARKERS`.
2. Added a lingua-confidence override: when lingua is ≥0.80 confident
   on DE/FR/IT, trust it over the RM marker count.
3. Excluded EN from the override — for EN, fall through to the marker
   count + final fallback. This way real Romansh content with low marker
   density still gets routed correctly via the romansh subset hint.

## Pipeline

```
       holdout parquet         rm.wikipedia.jsonl.gz
              │                          │
        stream_chunks                _iter_romansh
              │                          │
       per-chunk: detect_lang + prefilter (pii_dense/control)
              │                          │
              └──────────┬───────────────┘
                         ▼
            stratified sample by (lang, presence)
                         │
                         ▼
             data/test_v1_pool.jsonl  (locked, chmod 444)
                         │
                         ▼
           [Days 4-5: 4-way subagent labeling]
                         │
                         ▼
             data/test_v1.jsonl  (frozen gold)
```

## Reproduction

```bash
uv run python -m gheim_training.eval.build_test_v1 \
    --out data/test_v1_pool.jsonl \
    --target-per-lang 150 \
    --controls-per-lang 50 \
    --seed 17
```

Deterministic given the same input shards and seed. Re-running
overwrites the pool — current is locked `chmod 444` to prevent
accidental clobbering. Backup at `data/_backups/test_v1_pool.jsonl.bak`.

## Discipline

- The pool file is for **labeling input only**.
- The forthcoming `data/test_v1.jsonl` (gold) is **NEVER** used for
  early stopping or `metric_for_best_model` (configs hard-code
  `eval_split: validation`).
- Score test_v1 once per model in the bake-off matrix (Day 9), once
  per leader candidate after the sweep (Day 12), once per ablation
  (Day 13), once for the composite (Day 14). That's it.
- If we ever score the same model on test_v1 twice and pick the better
  number, we've turned it into a dev set. Don't.
