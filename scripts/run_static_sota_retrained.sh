#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${GPU_ID:-0}"; SEED="${SEED:-0}"; MODELS="${MODELS:-staeformer stid}"
DATASETS="${DATASETS:-PEMSD3-stream PEMSD8-mini PEMSD4-large}"; RESUME="${RESUME:-1}"
EPOCHS="${EPOCHS:-100}"; PATIENCE="${PATIENCE:-20}"; ALLOW_LARGE="${ALLOW_LARGE:-0}"
STATIC_RUN_TAG="${STATIC_RUN_TAG:-}"
PROTOCOLS="${PROTOCOLS:-current}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for model in $MODELS; do
 for dataset in $DATASETS; do
  for protocol in $PROTOCOLS; do
   batch="${BATCH_SIZE:-16}"; [[ "$model" == stid ]] && batch="${STID_BATCH_SIZE:-64}"; [[ "$model" == dlinear ]] && batch="${DLINEAR_BATCH_SIZE:-128}"; [[ "$model" == patchtst ]] && batch="${PATCHTST_BATCH_SIZE:-32}"
   cmd=(python "$ROOT/scripts/run_static_sota_retrained.py" --model "$model" --dataset "$dataset" --seed "$SEED" --gpu "$GPU_ID" --epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$batch" --protocol "$protocol")
   [[ -n "$STATIC_RUN_TAG" ]] && cmd+=(--run-tag "$STATIC_RUN_TAG")
   [[ "$RESUME" == 1 ]] && cmd+=(--resume) || cmd+=(--no-resume)
   [[ "$ALLOW_LARGE" == 1 ]] && cmd+=(--allow-large)
   "${cmd[@]}"
  done
 done
done
