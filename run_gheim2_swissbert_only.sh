#!/bin/bash
# Re-run just gheim2-swissbert with the fp32 accelerate config.
#
# The first attempt used accelerate_ddp.yaml (mixed_precision: bf16) and
# crashed at step 0 with "Index put requires source/destination dtypes
# match, got Float for the destination and BFloat16 for the source" —
# XMOD's lang_adapter scatter does the language-adapter routing in fp32
# and HF Trainer's bf16 autocast can't bridge the dtype boundary.
#
# accelerate_ddp_fp32.yaml uses mixed_precision: 'no' which makes the
# whole training run fp32. Slower per step (~2× swissbert wall time vs
# bf16 baseline) but avoids the crash. Same workaround as gheim-1.
set -u

ROOT="/home/joelbarmettler/projects/gheim"
cd "$ROOT" || exit 1

config="training/configs/gheim2_swissbert.yaml"
log="logs/gheim2_swissbert.log"

echo "==========================================" | tee -a "$log"
echo "STARTING gheim2-swissbert (fp32) at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
echo "==========================================" | tee -a "$log"

start_ts=$(date +%s)
if uv run accelerate launch \
    --config_file training/configs/accelerate_ddp_fp32.yaml \
    -m gheim_training.train \
    --config "$config" >> "$log" 2>&1; then
  end_ts=$(date +%s)
  dur=$((end_ts - start_ts))
  echo "FINISHED gheim2-swissbert after $((dur / 60))m at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
else
  echo "FAILED gheim2-swissbert at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$log"
fi
