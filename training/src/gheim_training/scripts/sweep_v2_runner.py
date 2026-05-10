"""wandb-sweep entrypoint for the v2 bake-off (P12).

Each sweep config (sweep_v2_{priv,xlmr,swissbert}.yaml) sets:
  - model         : "priv" | "xlmr" | "swissbert"   (constant per sweep)
  - learning_rate : 5e-6, 1e-5, 2e-5, 3e-5, 5e-5    (swept)
  - llrd          : 1.0 (off), 0.95, 0.9            (swept)

That's 5 × 3 = 15 grid cells per model. LR is the highest-impact knob
across model sizes; layer-wise LR decay (LLRD) is the most underexplored
knob in our existing configs and matters most for large pretrained
encoders — it lets lower transformer layers preserve general features
while the head adapts to the task.

Each run:
  - Trains on data/built_v2 (gheim-pii train + AI4P aug = 156k chunks)
  - Caps at 500 optimizer steps (~ ⅓ epoch at effective batch 128)
  - Evals once at end on validation
  - Saves only the best checkpoint by overall_f1
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# model → base config file (relative to repo root)
MODEL_TO_CONFIG: dict[str, str] = {
    "priv":      "training/configs/bakeoff_priv.yaml",
    "xlmr":      "training/configs/bakeoff_xlmr.yaml",
    "swissbert": "training/configs/bakeoff_swissbert.yaml",
}

# Stage-1 sweep budget (per run).
#
# Two modes:
#   "steps" (default — backward-compatible): cap at SWEEP_MAX_STEPS optimizer
#                                            steps. Cheap ranking signal but
#                                            absolute F1 values are meaningless
#                                            because training was cut early.
#                                            Per-run ~5/12/25 min. Total per
#                                            model ~1.25h / ~3h / ~6.25h.
#   "epochs":                                full num_train_epochs=1 on the
#                                            train mix (~13 min swissbert /
#                                            ~30 min xlm-r / ~60 min priv).
#                                            F1 converged enough to rank
#                                            configs honestly. Total per
#                                            model ~3.25h / ~7.5h / ~15h.
#
# Activate the epoch mode when launching subsequent agents:
#   SWEEP_BUDGET=epochs wandb agent <sweep_id>
SWEEP_BUDGET = os.environ.get("SWEEP_BUDGET", "epochs")
SWEEP_MAX_STEPS = int(os.environ.get("SWEEP_MAX_STEPS", "500"))
SWEEP_NUM_EPOCHS = float(os.environ.get("SWEEP_NUM_EPOCHS", "1"))


def main() -> None:
    import wandb
    run = wandb.init()
    if run is None:
        sys.exit("wandb.init() returned None — sweep agent context expected.")

    model = wandb.config.get("model")
    lr = wandb.config.get("learning_rate")
    llrd = wandb.config.get("llrd")

    if model not in MODEL_TO_CONFIG:
        raise ValueError(
            f"unknown model {model!r}; expected one of {sorted(MODEL_TO_CONFIG)}"
        )
    if lr is None or llrd is None:
        raise ValueError(f"missing learning_rate or llrd in wandb.config: {dict(wandb.config)}")

    # Compose deterministic ids from the swept values
    lr_tag = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    llrd_tag = f"llrd{int(llrd * 100):03d}"
    variant = f"{model}_lr{lr_tag}_{llrd_tag}"
    output_dir = f"./checkpoints/sweep_v2_{variant}"
    run_name = f"sweep-v2-{variant}"

    # Have train.py log to THIS sweep run instead of creating a new one
    os.environ["WANDB_RUN_ID"] = run.id
    os.environ["WANDB_RESUME"] = "allow"
    run.finish()

    repo_root = Path(__file__).resolve().parents[4]
    base_cfg = repo_root / MODEL_TO_CONFIG[model]
    # SwissBERT's XMOD lang_adapter has fp32 weights; bf16 autocast causes a
    # dtype mismatch on the residual put-back ("Index put requires source
    # and destination dtypes match"). Force fp32 for swissbert.
    accel_cfg = (
        "training/configs/accelerate_ddp_fp32.yaml"
        if model == "swissbert"
        else "training/configs/accelerate_ddp.yaml"
    )
    cmd = [
        "uv", "run", "accelerate", "launch",
        "--config_file", str(repo_root / accel_cfg),
        "-m", "gheim_training.train",
        "--config", str(base_cfg),
        # Point at the new built dataset (gheim-pii train_mix + val + test)
        "--override", "dataset_dir=./data/built_v2",
        # Sweep variants
        "--override", f"learning_rate={lr}",
        "--override", f"llrd={llrd}",
        # Save only one checkpoint (the best one by val F1) per sweep cell
        "--override", "save_total_limit=1",
        # Bookkeeping
        "--override", f"output_dir={output_dir}",
        "--override", f"wandb_run_name={run_name}",
    ]
    # Budget mode: either step-capped (cheap ranking) or full-epoch (honest F1)
    if SWEEP_BUDGET == "steps":
        cmd.extend([
            "--override", f"max_steps={SWEEP_MAX_STEPS}",
            "--override", "eval_steps=500",
            "--override", "save_steps=500",
        ])
    else:  # "epochs" (default)
        cmd.extend([
            "--override", f"num_train_epochs={SWEEP_NUM_EPOCHS}",
            # Eval + save once at the end of the epoch
            "--override", "eval_strategy=epoch",
            "--override", "save_strategy=epoch",
        ])
    # bf16 off for swissbert
    if model == "swissbert":
        cmd.extend(["--override", "bf16=false"])

    print(f"[sweep-v2] {variant} → {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=repo_root)
    sys.exit(rc)


if __name__ == "__main__":
    main()
