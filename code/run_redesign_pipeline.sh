#!/bin/bash
# Sequential pipeline for the advisor's Priority 1/2/3 redesign runs.
# Single 6GB GPU -> everything runs one job at a time, in order.
set -uo pipefail
cd ~/projects/clean-data-encoder/code
source ~/.cde_gpu_env

SCRUB_FLAGS="use_rope=true use_rmsnorm=true use_swiglu=true model_dim=512 num_layers=8 num_heads=8 batch_size=32 early_stopping_patience=10 early_stopping_min_delta=0.001 epochs=100 require_gpu=true"

log() { echo "[pipeline $(date '+%H:%M:%S')] $*"; }

log "=== STEP 1/4: per-record ablation, SCRUB v3, seed 42 ==="
uv run python run_ablation.py --sweep-name perrecord_s42 --per-record --train \
  --dataset russia_065 --seed 42 $SCRUB_FLAGS
log "step 1 exit code: $?"

log "=== STEP 2/4: per-record ablation, SCRUB v3, seed 123 (replication check) ==="
uv run python run_ablation.py --sweep-name perrecord_s123 --per-record --train \
  --dataset russia_065 --seed 123 $SCRUB_FLAGS
log "step 2 exit code: $?"

log "=== STEP 3/4: aggregate the two per-record seeds ==="
uv run python aggregate_multiseed.py perrecord_s42 perrecord_s123
log "step 3 exit code: $?"

log "=== waiting for the gpu+pretrained uv sync to finish, if it hasn't already ==="
while pgrep -f 'uv sync --extra gpu --extra pretrained' > /dev/null; do sleep 20; done
log "sync confirmed done"

log "=== STEP 4/4: matched-architecture RQ1 comparison (pretrained vs random-init) ==="
log "--- 4a: pretrained (from checkpoint), batch=8, seed 42 ---"
uv run python train.py --run matched_pretrained_s42 dataset=russia_065 \
  encoder_type=pretrained pretrained_random_init=false batch_size=8 epochs=100 \
  early_stopping_patience=10 early_stopping_min_delta=0.001 seed=42 require_gpu=true --eval-after
log "4a exit code: $?"

log "--- 4b: random-init, SAME architecture/tokenizer/batch size, seed 42 ---"
uv run python train.py --run matched_randominit_s42 dataset=russia_065 \
  encoder_type=pretrained pretrained_random_init=true batch_size=8 epochs=100 \
  early_stopping_patience=10 early_stopping_min_delta=0.001 seed=42 require_gpu=true --eval-after
log "4b exit code: $?"

log "=== PIPELINE COMPLETE ==="
