<p align="center">
  <img src="https://github.com/joelbarmettlerUZH/gheim/blob/main/assets/logo.png" alt="gheim" width="360">
</p>

# gheim training

Pipeline that produces [`joelbarmettler/gheim-ch-560m`](https://huggingface.co/joelbarmettler/gheim-ch-560m)
and the companion [`joelbarmettler/gheim-ch-pii-171k`](https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-171k)
dataset. A fine-tune of `xlm-roberta-large` on Swiss-domain text, scored
against a held-out test split and against seven external PII / NER
systems (see [`MODEL_CARD.md`](../MODEL_CARD.md) for the full comparison).

## Layout

- `configs/`: hyperparameter and accelerate configs for the bake-off
  runs and the leader's hyperparameter sweep.
- `src/gheim_training/data/`: the 7-phase curation pipeline that builds
  `gheim-ch-pii-171k` from the Apertus pretrain corpora plus a synthetic
  gap-fill layer. Each script is a phase; see
  [`DATASET_CARD.md`](../DATASET_CARD.md) for the curation procedure.
- `src/gheim_training/data/synthetic/`: Faker-CH name and address
  generators backed by Geonames-CH, plus the Gemma-driven slot-fill
  generator for the synthetic gap-fill chunks.
- `src/gheim_training/eval/`: the comparison eval pipeline (consolidated
  to four scripts: `eval_on_ours.py`, `eval_on_external.py`,
  `replicate_native.py`, `report.py`).
- `src/gheim_training/train.py`: HF Trainer entrypoint.

## Sequence

1. Build the dataset (`data/p*.py` scripts in order).
2. Bake-off three base models on the same training mix (`configs/bakeoff_*.yaml`).
3. Hyperparameter sweep on the leader (`configs/sweep_v2_*.yaml`).
4. Score the leader on the held-out test split once (`eval/eval_on_ours.py`).
5. Score the comparison contestants and the cross-domain external benchmarks
   (`eval/eval_on_ours.py` and `eval/eval_on_external.py`).
6. Render the comparison section (`eval/report.py`) and paste into
   [`MODEL_CARD.md`](../MODEL_CARD.md).
