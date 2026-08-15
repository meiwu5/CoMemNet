#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"

for variant in sampler_feature sampler_random no_replay target_no_target loss_mae_only static retrained; do
    echo "[core] variant=${variant} seed=${SEED}"
    python main.py \
        --conf "config/reviewer/PEMSD3-stream/${variant}.json" \
        --dataset PEMSD3-stream \
        --gpuid "${GPU_ID}" \
        --seed "${SEED}"
done
