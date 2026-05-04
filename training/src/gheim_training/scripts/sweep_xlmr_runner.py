"""wandb-sweep entrypoint for the xlmr leader hyperparameter sweep.

The CLAUDE.md plan calls for 6 specific runs varying ONE axis from the
baseline at a time. wandb's grid sweep wants a parameter mapping, so we
expose a single ``variant`` parameter and translate it to the right
override(s) in this thin wrapper:

    variant = "lr5e6"     → --override learning_rate=5e-6
    variant = "lr5e5"     → --override learning_rate=5e-5
    variant = "batch256"  → --override gradient_accumulation_steps=2
    variant = "epoch5"    → --override num_train_epochs=5
    variant = "freeze"    → --override freeze_embeddings=true
    variant = "baseline"  → no overrides (rerun of bakeoff_xlmr to confirm
                            reproducibility within the sweep tracking)

For each variant we set a unique output_dir and wandb_run_name so
checkpoints don't collide and the sweep dashboard groups cleanly.

Launched indirectly by ``wandb agent`` after registering
``training/configs/sweep_xlmr.yaml`` with ``wandb sweep``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# variant → list of --override args
VARIANT_OVERRIDES: dict[str, list[str]] = {
    "baseline": [],
    "lr5e6":    ["learning_rate=5e-6"],
    "lr5e5":    ["learning_rate=5e-5"],
    "batch256": ["gradient_accumulation_steps=2"],
    "epoch5":   ["num_train_epochs=5"],
    "freeze":   ["freeze_embeddings=true"],
}


def main() -> None:
    import wandb
    # wandb agent injects the sweep parameter values into wandb.config when
    # the wrapper does wandb.init() with no project/sweep id (it reads the
    # environment that the agent set up).
    run = wandb.init()
    if run is None:
        sys.exit("wandb.init() returned None — wandb agent context expected.")
    variant = wandb.config.get("variant")
    if variant not in VARIANT_OVERRIDES:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of "
            f"{sorted(VARIANT_OVERRIDES)}",
        )
    overrides = VARIANT_OVERRIDES[variant]
    output_dir = f"./checkpoints/sweep_xlmr_{variant}"
    run_name = f"sweep-xlmr-{variant}"

    # Tell train.py to also use this same wandb run id (so the sweep agent
    # captures the metrics under the same wandb run rather than creating a
    # duplicate). We finish() this wrapper run before exec'ing accelerate;
    # train.py will then init its own wandb under WANDB_RUN_ID.
    os.environ["WANDB_RUN_ID"] = run.id
    os.environ["WANDB_RESUME"] = "allow"
    run.finish()

    repo_root = Path(__file__).resolve().parents[4]
    cmd = [
        "uv", "run", "accelerate", "launch",
        "--config_file", str(repo_root / "training/configs/accelerate_ddp.yaml"),
        "-m", "gheim_training.train",
        "--config", str(repo_root / "training/configs/bakeoff_xlmr.yaml"),
        "--override", f"output_dir={output_dir}",
        "--override", f"wandb_run_name={run_name}",
    ]
    for kv in overrides:
        cmd.extend(["--override", kv])

    print(f"sweep variant={variant!r}: launching {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=repo_root)
    sys.exit(rc)


if __name__ == "__main__":
    main()
