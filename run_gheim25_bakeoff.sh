#!/bin/bash
# gheim-2.5 bake-off: gold core + Layer 5.v3 (Gemma-labeled real Swiss text).
#
# Skips priv (lagged in both gheim-1 and gheim-2; not worth retraining).
# xlmr first (gheim-2 leader candidate by tie-break), then swissbert
# (was the actual gheim-2 leader by tie-break — needs to defend).
#
# swissbert uses fp32 accelerate config to avoid XMOD bf16/fp32 dtype
# mismatch (same workaround as gheim-1 and gheim-2).
set -u

ROOT="/home/joelbarmettler/projects/gheim"
cd "$ROOT" || exit 1

start_overall=$(date +%s)

echo "==========================================" | tee -a logs/gheim25_bakeoff.log
echo "STARTING gheim25-xlmr at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/gheim25_xlmr.log
echo "==========================================" | tee -a logs/gheim25_xlmr.log
start_xlmr=$(date +%s)
if uv run accelerate launch \
    --config_file training/configs/accelerate_ddp.yaml \
    -m gheim_training.train \
    --config training/configs/gheim25_xlmr.yaml >> logs/gheim25_xlmr.log 2>&1; then
  end_xlmr=$(date +%s)
  echo "FINISHED gheim25-xlmr after $(((end_xlmr - start_xlmr) / 60))m at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/gheim25_xlmr.log
else
  echo "FAILED gheim25-xlmr at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/gheim25_xlmr.log
fi

echo "==========================================" | tee -a logs/gheim25_swissbert.log
echo "STARTING gheim25-swissbert at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/gheim25_swissbert.log
echo "==========================================" | tee -a logs/gheim25_swissbert.log
start_sb=$(date +%s)
if uv run accelerate launch \
    --config_file training/configs/accelerate_ddp_fp32.yaml \
    -m gheim_training.train \
    --config training/configs/gheim25_swissbert.yaml >> logs/gheim25_swissbert.log 2>&1; then
  end_sb=$(date +%s)
  echo "FINISHED gheim25-swissbert after $(((end_sb - start_sb) / 60))m at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/gheim25_swissbert.log
else
  echo "FAILED gheim25-swissbert at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/gheim25_swissbert.log
fi

end_overall=$(date +%s)
echo "==========================================" | tee -a logs/gheim25_bakeoff.log
echo "ALL GHEIM-2.5 BAKE-OFF RUNS DONE in $(((end_overall - start_overall) / 60))m at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a logs/gheim25_bakeoff.log
echo "==========================================" | tee -a logs/gheim25_bakeoff.log
