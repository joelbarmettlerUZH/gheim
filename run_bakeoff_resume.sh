#!/bin/bash
# Day 8 RESUME: bakeoff-priv finished cleanly; xlmr + swissbert failed
# (wandb timeout / wrong repo id). Re-launches just the missing 2.
set -u

ROOT="/home/joelbarmettler/projects/gheim"
cd "$ROOT" || exit 1

run_one() {
  local name="$1"
  local config="training/configs/bakeoff_${name}.yaml"
  local log="logs/bakeoff_${name}.log"
  echo "==========================================" | tee -a "$log"
  echo "RESUMING bakeoff-${name} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  echo "==========================================" | tee -a "$log"
  local start_ts
  start_ts=$(date +%s)
  if uv run accelerate launch \
      --config_file training/configs/accelerate_ddp.yaml \
      -m gheim_training.train \
      --config "$config" >> "$log" 2>&1; then
    local end_ts
    end_ts=$(date +%s)
    local dur=$((end_ts - start_ts))
    echo "FINISHED bakeoff-${name} after $((dur / 60))m at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  else
    echo "FAILED bakeoff-${name} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  fi
}

run_one xlmr
run_one swissbert

echo "==========================================" | tee -a logs/bakeoff.log
echo "RESUMED RUNS DONE at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/bakeoff.log
echo "==========================================" | tee -a logs/bakeoff.log
