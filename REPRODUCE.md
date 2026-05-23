# Reproducing the gheim-ch dataset and model

This document is the canonical end-to-end recipe for building the
published `gheim-ch-pii` dataset and training the `gheim-ch` PII
detector from raw inputs. Every script invoked here lives in
`training/src/gheim_training/`.

## Prerequisites

| Item | Notes |
|---|---|
| Hardware | 2 × RTX 4090 (24 GB) for LLM labelling + DDP training. Single 4090 also works for the final training step. |
| Python | 3.13 (project pins via `pyproject.toml`) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| GPU drivers | CUDA 12.x with bf16 support |
| Disk | ~120 GB free under `data/` for the labelling intermediates |
| Geonames-CH | `data/raw/CH.txt` (download instructions in `data/synth/swiss_geo.py`) |
| Apertus pre-train Swiss corpus | the project consumes its public shards directly via the HF datasets cache |

Install once:

```bash
uv sync
```

## Pipeline overview

```
[Swiss text corpus]
       │
       │   labelling/ — three LLM labellers + regex
       ▼
data/assembled.jsonl  (~2.3M labelled chunks, per-span provenance)
       │
       │   + 50k Faker_CH document templates    (synth/docs/)
       │   + 880  Gemma-slot-fill gap-fill      (synth/slot_fill_gemma.py)
       │   + 800  RM secret templates           (synth/templates/)
       │   + 2.7k name-pattern templates         (synth/templates/)
       │   + 64k composer-rendered chunks        (synth/generate.py)
       ▼
balance.py  (5 phases: signal floor, per-doc cap, per-value cap,
             per-cell cap, two-tier negatives)
       │
       ▼
data/balanced.jsonl  (~221k chunks)
       │
       │   split.py (80/10/10 stratified by language × source)
       ▼
data/balanced_{train,val,test}.jsonl
       │
       │   build_hf.py
       ▼
data/built/  (HuggingFace DatasetDict)
       │
       │   train.py + configs/train.yaml + configs/accelerate_ddp.yaml
       ▼
checkpoints/gheim-ch/  (xlm-roberta-large fine-tune, ~70 min on 2×4090)
       │
       │   eval/probe.py, eval/eval_on_test.py, eval/per_lang_cat.py
       ▼
eval/  (probe report, per-(lang × cat) F1 matrix, test-set headline F1)
```

## Step 1 — Label the real-text corpus (~6h total)

Three LLM labellers run independently on the same chunks, then a
deterministic regex catalogue + the four supply spans to the merge
step. The 4-LLM OpenRouter audit (paper-anchor only, not part of the
published labels) runs separately on a 580-chunk sample.

```bash
# Each labeller takes 1-2 hours on 2x4090 with vLLM TP=2.
uv run python -m gheim_training.data.labelling.gemma.run_label
uv run python -m gheim_training.data.labelling.qwen.run_label_v2
uv run python -m gheim_training.data.labelling.nemotron.run_label_v2

# Format-aware re-labelling of commercial-register patterns
uv run python -m gheim_training.data.labelling.run_label_cr

# Merge all signals → data/assembled.jsonl (always applies gazetteer
# + label-conflict resolution; see labelling/merge.py docstring)
uv run python -m gheim_training.data.labelling.assemble

# (optional) 4-LLM audit for label-noise estimate
uv run python -m gheim_training.data.analysis.audit_openrouter
uv run python -m gheim_training.data.analysis.audit_score
```

## Step 2 — Generate synthetic data (~10 min)

```bash
# Composer-framework synthetic layers (emails, docs, short-form,
# forms, common-word, adversarial) from the 28 hand-written template
# files in synth/templates/.
uv run python -m gheim_training.data.synth.generate

# Legacy KYC/HR/banking document layer (Faker_CH + Geonames-CH)
uv run python -m gheim_training.data.synth.docs.generate \
    --out data/layer1.jsonl --n 50000 --seed 17

# Gemma slot-fill gap-fill (rare-category data)
uv run python -m gheim_training.data.synth.slot_fill_gemma

# Hand-written supplementary layers (templates + Faker fill)
uv run python -m gheim_training.data.synth.legacy_synth_rm_secrets
uv run python -m gheim_training.data.synth.legacy_synth_name_patterns
```

## Step 3 — Balance + split + build the HF dataset (~15 min)

```bash
uv run python -m gheim_training.data.balance      # → data/balanced.jsonl
uv run python -m gheim_training.data.split        # → data/balanced_{train,val,test}.jsonl
uv run python -m gheim_training.data.build_hf     # → data/built/
```

## Step 4 — Train (~70 min on 2 × RTX 4090)

```bash
uv run accelerate launch \
    --config_file training/configs/accelerate_ddp.yaml \
    -m gheim_training.train \
    --config training/configs/train.yaml
```

Outputs the fine-tuned checkpoint to `checkpoints/gheim-ch/`.

## Step 5 — Evaluate (~5 min)

```bash
# 212-case forensic edge-case probe
uv run python -m gheim_training.eval.probe \
    --model checkpoints/gheim-ch \
    --report-out eval/probe_report.json

# Test-set headline F1 (touched ONCE per model)
uv run python -m gheim_training.eval.eval_on_test \
    --backend hf --model checkpoints/gheim-ch \
    --dataset data/built --split test \
    --out eval/test_report.json

# Per-(language × category) F1 matrix
uv run python -m gheim_training.eval.per_lang_cat \
    --model checkpoints/gheim-ch \
    --dataset-dir data/built --split test \
    --out eval/per_lang_cat.json
```

## Determinism notes

All randomness in the synthetic generation + balancing pipeline uses
explicit seeds (defined as module-level `SEED` constants). Two
sources of non-determinism remain:

1. **LLM labellers** (Gemma / Qwen / Nemotron) — run at `temperature=0`
   but vLLM's output is not bit-deterministic across hardware. Re-running
   the labelling step produces labels that match to ~99% of spans on
   identical inputs but the remaining ~1% differ. Downstream balancing /
   training absorbs this through majority-vote merging.
2. **HuggingFace Trainer** — uses `seed: 20260524` in `configs/train.yaml`.
   Re-runs on identical hardware are bit-deterministic; cross-GPU runs
   land within ±0.3 pp on overall F1.

## Output verification

After step 3, the `data/balanced_summary.json` file records the
expected per-(language × source) chunk counts and signal-strength
distribution. Compare against `data/balanced_summary.json` from this
commit to confirm the pipeline produced the expected output.

After step 5, compare against `eval/probe_report.json` (212 cases,
expect ≥98% perfect) and `eval/per_lang_cat.json` (expect overall char
F1 ≥ 0.93).

## Dataset layers — what's in each layer file

| File | Source | Chunks | Note |
|---|---|---:|---|
| `data/assembled.jsonl` | LLM + regex labelling | ~2.3M | Real text from Apertus pre-train Swiss |
| `data/layer1.jsonl` | Faker_CH templates | 50,000 | KYC / HR / banking / medical / secret-leak / doctor-note documents (de/fr/it) |
| `data/layer9.jsonl` | Gemma slot-fill | 880 | Gap-fill for rare-category cells |
| `data/layer_rm_secrets.jsonl` | hand templates | 800 | Closes (rm × secret) cell |
| `data/layer_name_patterns.jsonl` | hand templates | 2,700 | Greetings, signatures, common-word surnames |
| `data/layer_synth_emails.jsonl` | composer | 31,000 | Multi-paragraph email compositions, all 5 langs |
| `data/layer_synth_docs.jsonl` | composer | 12,750 | DE legal / medical / bank / HR documents |
| `data/layer_synth_short_form.jsonl` | composer | 9,200 | Standalone narrative, all 5 langs |
| `data/layer_synth_forms.jsonl` | composer | 4,000 | DE structured layouts (CSV / vCard / YAML / ...) |
| `data/layer_synth_common_word.jsonl` | composer | 4,800 | Common-word surname positive / negative pairs |
| `data/layer_synth_adversarial.jsonl` | composer | 2,800 | Lookalike-but-not-PII negatives |

After balancing, the published dataset is ~221k chunks across 5
languages × 8 PII categories with full per-span provenance.
