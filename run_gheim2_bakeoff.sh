#!/bin/bash
# gheim-2 bake-off training runner.
#
# Uses the new gold-core dataset (Layer 1 + Layer 2.v2 + Layer 4.v2 from
# openpii-1m with the unified 3-schema REMAP). Each model trains via
# accelerate DDP across 2× RTX 4090. Estimated ~6h each. Per-run logs go
# to logs/gheim2_<base>.log.
#
# Tie-break for leader selection (per user direction):
#   1. F1 within 0.02 → smaller params win
#   2. → Swiss-specific origin wins
# So if SwissBERT (270M, ZurichNLP) comes within 0.02 of XLM-R (550M),
# SwissBERT wins. If only XLM-R is within 0.02 of priv (1.4B MoE), XLM-R
# wins. Etc.
set -u

ROOT="/home/joelbarmettler/projects/gheim"
cd "$ROOT" || exit 1

run_one() {
  local name="$1"
  local config="training/configs/gheim2_${name}.yaml"
  local log="logs/gheim2_${name}.log"
  echo "==========================================" | tee -a "$log"
  echo "STARTING gheim2-${name} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
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
    echo "FINISHED gheim2-${name} after $((dur / 60))m at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  else
    echo "FAILED gheim2-${name} at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
  fi
}

# xlm-r first: previous gheim-1 leader at F1=0.817; we want the cleanest
# data-impact comparison head-to-head. priv next: 1.4B reference. swissbert
# last because XMOD requires fp32 (slower) and any structural issues there
# we already debugged in gheim-1.
run_one xlmr
run_one priv
run_one swissbert

echo "==========================================" | tee -a logs/gheim2_bakeoff.log
echo "ALL GHEIM-2 BAKE-OFF RUNS DONE at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/gheim2_bakeoff.log
echo "==========================================" | tee -a logs/gheim2_bakeoff.log
