#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${GPU_ID:-0}"; SEEDS="${SEEDS:-0 1 2}"; RESUME="${RESUME:-1}"; DATASETS="${DATASETS:-PEMSD3-stream PEMSD4-large PEMSD8-mini}"
python -c 'import sys,torch; cap=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None; arch=set(torch.cuda.get_arch_list()) if torch.cuda.is_available() else set(); bad=cap is not None and cap >= (12,0) and "sm_120" not in arch; print("ERROR: install CUDA 12.8+ PyTorch with sm_120",file=sys.stderr) if bad else None; sys.exit(1 if bad else 0)'
is_complete() {
 local dataset="$1" seed="$2" config="config/reviewer/$1/sampler_feature.json"
 python -c 'import hashlib,json,pathlib,sys; ds,seed,cfg=sys.argv[1],int(sys.argv[2]),pathlib.Path(sys.argv[3]); expected=hashlib.sha256(cfg.read_bytes()).hexdigest(); ok=False
for p in pathlib.Path("res/reviewer",ds).glob("**/metrics/summary.json"):
 try:
  d=json.loads(p.read_text()); ok |= d.get("dataset")==ds and d.get("variant")=="sampler_feature" and int(d.get("seed",-1))==seed and d.get("config_hash")==expected
 except Exception: pass
sys.exit(0 if ok else 1)' "$dataset" "$seed" "$config"
}
for dataset in ${DATASETS}; do for seed in ${SEEDS}; do
 if [[ "$RESUME" == 1 ]] && is_complete "$dataset" "$seed"; then echo "[resume] skip final main: dataset=$dataset seed=$seed"; else echo "[run] final main: dataset=$dataset seed=$seed"; python main.py --conf "config/reviewer/$dataset/sampler_feature.json" --dataset "$dataset" --gpuid "$GPU_ID" --seed "$seed"; fi
done; done
python scripts/aggregate_final_main.py --root res/reviewer --seeds "$SEEDS" --datasets "$DATASETS" --output res/reviewer/final_main_multiseed.json
