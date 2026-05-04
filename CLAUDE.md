# CLAUDE.md — gheim-1 bake-off branch

This branch (`gheim-1-bakeoff`) is a clean restart, scoped to **3 weeks**,
focused on the Phase 3 multi-base bake-off that was skipped in the original
plan. It supersedes everything on `swiss-pii-finetune`.

The previous branch is preserved at tag `frozen-data-ceiling-v4` and at
`origin/swiss-pii-finetune`. That work hit F1 = 0.518 on an OOD holdout
under one base model (`openai/privacy-filter`) without ever testing the
alternatives — see `FEEDBACK.md` on the old branch for the full audit.

## Goal

Determine the best base model + training configuration for Swiss-market
PII detection by running a controlled bake-off, then ship
`joelbarmettler/gheim-1` with honest, reproducible numbers from a real
held-out test set.

## Hardware

- 2 × RTX 4090, 24 GB each, no NVLink (PCIe). Use DDP, not FSDP.
- All bake-off runs sized for ~6h on the pair.

## What we keep from the old branch

These pieces survive the restart because they were validated and are
load-bearing:

- **33-class label-space alignment** (`data/label_space.py`) — byte-identical
  to `openai/privacy-filter`'s `id2label`. For non-priv bases (XLM-R, SwissBERT)
  the head is randomly initialised; for priv it reuses pretrained weights.
- **Swiss address generator** (`data/synthetic/swiss_address.py`,
  `swiss_geo.py`) — Geonames-CH backed, per-language street pools.
- **Faker_CH IBAN/AHV/VAT/CC checksums** (`data/synthetic/faker_ch.py`).
- **Apertus labeler verifier** (`data/apertus_label/verify.py`) — 5-gate
  validator (M1–M5 production config). Used to produce Layer 5 in the
  carried-forward dataset.
- **Eval harness** (`eval/harness.py`) — with the F1 div-by-zero
  semantic fix.
- **Locked data layers** (in `data/`, gitignored): Layers 1–8 are
  reused as-is. No new data generation in this plan.

## What we drop

- `_BoundaryWeightedTrainer` — A1 boundary loss didn't beat standard CE
  (v4.1: F1 = 0.46 vs v4: F1 = 0.54). Stripped.
- M2 ALL-CAPS person regex — already removed in Layer 8 because it taught
  the model "ALL-CAPS = person" which mis-fired on legal acronyms.
- The whole `active_learning/` module — Phase 4 was the wrong axis to
  iterate on before establishing the model ceiling.
- Layer 9 and Layer B5 — generated but excluded from the bake-off so the
  experiment is comparable across bases. May be revisited after we have
  a leader.
- Layer 4 (English anchor) — kept on disk but documented as a known
  limitation if we ship without it; the English-anchor regression check
  in Day 15 will tell us if Swiss-only training drops English F1 below
  the base.

## Phases

### Week 1 — branch cutover, baselines, new test set

**Day 1: branch cutover.** Done in commit-1 of this branch.

**Day 2: eval baselines.** Run zero-shot `openai/privacy-filter` and
`swissner+regex` on the existing `audit_gold` and `address_gold_v1`. These
become trend lines. Record in `eval/baselines.json`.

**Day 3–6: build `test_v1`.** ≥600 chunks, ≥1500 spans, ≥200 address spans.
Stratify by (language × source × category presence). Source from corpora
not previously used (different shard of `apertus-pretrain-swiss`,
unsampled FineWeb-2-Swiss). 4-way subagent labeling, hand-resolve
disagreements. Lock `chmod 444`. **Never used for early stopping or
checkpoint selection.**

**Day 7: sanity report.** Wilson 95% CI ≤ ±0.04 on overall F1. Top up
address spans if < 200.

### Week 2 — the bake-off

**Day 8: train 3 base models.**

| Run | Base | Config |
|---|---|---|
| `bakeoff-priv` | `openai/privacy-filter` (1.4B MoE) | `configs/bakeoff_priv.yaml` |
| `bakeoff-xlmr` | `FacebookAI/xlm-roberta-large` (550M dense) | `configs/bakeoff_xlmr.yaml` |
| `bakeoff-swissbert` | `ZurichNLP/swissbert-large` (270M dense) | `configs/bakeoff_swissbert.yaml` |

Identical training config across all three: LR 2e-5, 3 epochs, batch 64
effective, AdamW, validation-loss early stopping, validation-F1 best-model
selection. Only `base_model` differs.

**Day 9: bake-off matrix.** Score 3 checkpoints + zero-shot + swissner+regex
on `test_v1`, `audit_gold`, `address_gold_v1`. Per-language, per-entity,
strict + overlap F1. Write `eval/bakeoff_matrix.json`.

**Gate**: leader must beat `swissner+regex` by ≥3 F1 on `test_v1` overall.
If not, ship the composite detector in Week 3 and document `gheim-1` as a
category-specific tool.

**Day 10–12: hyperparam sweep on the leader (6 runs).**

| Run | Variation |
|---|---|
| sweep-1 | Leader baseline (Day 8) |
| sweep-2 | LR 5e-6 |
| sweep-3 | LR 5e-5 |
| sweep-4 | gradient_accumulation_steps=2 (effective batch 256) |
| sweep-5 | num_train_epochs=5, val-loss early stop |
| sweep-6 | Embeddings frozen |

Pick best as `gheim-1-leader-tuned`.

### Week 3 — ablations + composite + ship

**Day 13: ablations.** Train leader minus Layer 8 and minus Layer 7. First
isolated measurement of what those data investments actually contributed.

**Day 14: CompositeDetector.** Port `stitch.py` from `swiss-pii-finetune`
as a basis. Build `packages/gheim-py/src/gheim/detectors/composite.py`
per Phase 5a: regex front-end (IBAN/AHV/VAT/CC/phone/email/URL with
checksums) → mask matched spans → leader on remainder.

**Day 15: English regression check + model card.** Run base + leader on
a validation slice of `data/layer4_en.jsonl`. Write `MODEL_CARD.md` with
numbers from `bakeoff_matrix.json` only. Tag `gheim-1-released-v1`.

**Day 16: Hub publication.** Push `joelbarmettler/gheim-1` and
`joelbarmettler/gheim-1-composite`. Update gheim README.

## Scoring discipline (do not skip)

1. **`test_v1` is touched ONCE per model**, in Day 9 and Day 13. Never
   used to pick a checkpoint, never used for early stopping. If we
   accidentally tune on it, we have no test set.
2. **Validation split is the only thing that drives `metric_for_best_model`
   and early stopping.** Configs hard-code `eval_split: validation`.
3. **One change per run.** If a sweep run varies LR, only LR changes —
   not the seed, not the data, not the schedule.
4. **`_f1` returns `None` for sparse cells** (no gold AND no predictions).
   Aggregates that average over per-category F1 must skip the `None`s,
   not coerce to 0. Reporters print `n/a` for those.

## Files — what's on this branch

The carried-forward training package only contains the load-bearing
modules. The full inventory of what was excluded vs ported is in
`/home/joelbarmettler/.claude/plans/inherited-beaming-eclipse.md`.

Key ported paths:

- `training/src/gheim_training/data/{label_space,bioes,schema,lang_detect,build,ai4privacy,english_anchor}.py`
- `training/src/gheim_training/data/synthetic/{swiss_address,swiss_geo,faker_ch,template,templates_*,generate}.py`
- `training/src/gheim_training/data/apertus_label/{prefilter,verify,stream,prompts,generate,audit}.py`
- `training/src/gheim_training/eval/{harness,swissner}.py` + `baselines/{zero_shot,swissbert_ner}.py`
- `training/src/gheim_training/train.py` (boundary-loss subclass stripped)
- `training/configs/{bakeoff_priv,bakeoff_xlmr,bakeoff_swissbert,accelerate_ddp}.yaml`

## Files NOT to touch (load-bearing)

- `training/src/gheim_training/data/label_space.py` — alphabetical category
  order is locked to `openai/privacy-filter`'s id2label. Reordering breaks
  pretrained head reuse. Has an assertion test.
- `training/src/gheim_training/data/bioes.py` — char→token alignment is
  shared by every layer. Changing it requires re-running every test.
- `packages/gheim-py/src/gheim/core/sentinels.py` `LABEL_TO_TAG` — the
  fine-tuned model emits these category names; sentinel mapping depends
  on them.

## What this plan deliberately does NOT do

- Generate any new training data (the dataset is frozen at Layers 1–8)
- Add Layer 9 or Layer B5 to the bake-off
- Tune the M1–M5 prompts further
- Re-do the Apertus audit (the gate failed; we accept this and rely on
  Layer 8 LLM-ensemble consensus + the Day 13 ablation to bound the impact)
- Rewrite history on `swiss-pii-finetune`

## Forecasted F1 — recalibrated

Original plan forecast 0.92–0.95 on handcrafted v2. v4 reality on OOD
holdout was 0.518. Recalibrated expectation, in line with the senior
review:

| Cut | Honest target on `test_v1` |
|---|---|
| Overall | **0.78–0.85** |
| de_ch / fr_ch / it_ch | 0.80+ |
| RM | 0.65–0.75 |
| GSW | unmeasured — flag in model card |
| private_address (overlap) | 0.80+ |
| private_person | 0.80+ |
| English (no regression) | within 5pp of base |

If the bake-off leader beats those numbers, we ship. If it doesn't, the
model card honestly states what it does and what it doesn't.
