#!/bin/bash
# Day 13 ablation runner: 2 xlmr trainings serially.
set -u
ROOT="/home/joelbarmettler/projects/gheim"
cd "$ROOT" || exit 1

run_one() {
  local name="$1"
  local config="training/configs/ablation_xlmr_${name}.yaml"
  local log="logs/ablation_xlmr_${name}.log"
  echo "==========================================" | tee -a "$log"
  echo "STARTING ablation_xlmr_${name} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  echo "==========================================" | tee -a "$log"
  local start_ts=$(date +%s)
  if uv run accelerate launch \
      --config_file training/configs/accelerate_ddp.yaml \
      -m gheim_training.train \
      --config "$config" >> "$log" 2>&1; then
    local end_ts=$(date +%s)
    local dur=$((end_ts - start_ts))
    echo "FINISHED ablation_xlmr_${name} after $((dur / 60))m at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  else
    echo "FAILED ablation_xlmr_${name} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  fi
}

# no_l8 first because it's the more important data point (bigger ablation)
run_one no_l8
run_one no_l7

echo "==========================================" | tee -a logs/ablations.log
echo "ABLATIONS DONE at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/ablations.log
echo "==========================================" | tee -a logs/ablations.log
