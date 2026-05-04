#!/bin/bash
# Day 8 bake-off training runner: 3 base models serially.
#
# Each model trains via accelerate DDP across 2× RTX 4090. Estimated ~6h
# each. Per-run logs go to logs/bakeoff_<base>.log. If a run fails, the
# next one still runs (the bake-off matrix in Day 9 needs at least 2 of
# the 3 to compare).
set -u

ROOT="/home/joelbarmettler/projects/gheim"
cd "$ROOT" || exit 1

run_one() {
  local name="$1"
  local config="training/configs/bakeoff_${name}.yaml"
  local log="logs/bakeoff_${name}.log"
  echo "==========================================" | tee -a "$log"
  echo "STARTING bakeoff-${name} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
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

# Order matters only for time pressure. priv first because it's the
# carried-over base; if anything is structurally broken, it shows up here.
run_one priv
run_one xlmr
run_one swissbert

echo "==========================================" | tee -a logs/bakeoff.log
echo "ALL BAKE-OFF RUNS DONE at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/bakeoff.log
echo "==========================================" | tee -a logs/bakeoff.log
