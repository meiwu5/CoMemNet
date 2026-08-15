#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${GPU_ID:-0}"; SEED="${SEED:-0}"; RESUME="${RESUME:-1}"; DATASETS="${DATASETS:-PEMSD3-stream PEMSD4-large PEMSD8-mini}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; EAC_ROOT="$ROOT/baseline/EAC-official"
python "$ROOT/scripts/generate_eac_configs.py"
python -c 'import torch,torch_geometric; print("torch",torch.__version__,"pyg",torch_geometric.__version__)'
for dataset in ${DATASETS}; do
 config="$ROOT/config/baseline/eac/$dataset.json"; run_root="$ROOT/res/baseline/EAC/$dataset/eac-$SEED"; output="$run_root/metrics/summary.json"
 if [[ "$RESUME" == 1 && -f "$output" ]]; then echo "[resume] skip EAC: dataset=$dataset seed=$SEED"; continue; fi
 echo "[run] EAC: dataset=$dataset seed=$SEED"
 (cd "$EAC_ROOT" && python main.py --conf "$config" --gpuid "$GPU_ID" --seed "$SEED" --method EAC --load_first_year 0)
 python "$ROOT/scripts/collect_eac_results.py" --dataset "$dataset" --seed "$SEED" --config "$config" --log "$run_root/eac.log" --output "$output"
done
