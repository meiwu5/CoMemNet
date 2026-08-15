#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0}"
RESUME="${RESUME:-1}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
EXPERIMENT_GROUPS="${REVIEWER_GROUPS:-${1:-r2_round2_scaling}}"
SCALING_SOURCE="${SCALING_SOURCE:-PEMSD4-large}"
SCALING_RATIOS="${SCALING_RATIOS:-0.25 0.50 0.75}"
SCALING_DATASETS="${SCALING_DATASETS:-PEMSD4-scale25 PEMSD4-scale50 PEMSD4-scale75 PEMSD4-large}"
D3_SCALING_DATASETS="${D3_SCALING_DATASETS:-PEMSD3-scale50 PEMSD3-stream}"

python -c 'import sys, torch; cap=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None; arch=set(torch.cuda.get_arch_list()) if torch.cuda.is_available() else set(); bad=cap is not None and cap >= (12, 0) and "sm_120" not in arch; print("ERROR: this GPU requires a PyTorch CUDA 12.8+ wheel containing sm_120. Install with: pip install --upgrade torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128", file=sys.stderr) if bad else None; sys.exit(1 if bad else 0)'

is_complete() {
  local dataset="$1" variant="$2" seed="$3" config_path="config/reviewer/${dataset}/${variant}.json"
  python -c 'import hashlib,json,pathlib,sys; dataset,variant,seed,root,config_path=sys.argv[1],sys.argv[2],int(sys.argv[3]),pathlib.Path(sys.argv[4]),pathlib.Path(sys.argv[5]); expected=hashlib.sha256(config_path.read_bytes()).hexdigest(); ok=False
for path in root.glob("**/metrics/summary.json"):
    try:
        r=json.loads(path.read_text())
        if r.get("dataset")==dataset and r.get("variant")==variant and int(r.get("seed",-1))==seed and r.get("config_hash")==expected:
            ok=True; break
    except Exception:
        pass
sys.exit(0 if ok else 1)' "$dataset" "$variant" "$seed" "res/reviewer/$dataset" "$config_path"
}

run_many() {
  local dataset="$1"; shift
  local variant seed
  for variant in "$@"; do
    for seed in ${SEEDS}; do
      if [[ "${RESUME}" == "1" ]] && is_complete "$dataset" "$variant" "$seed"; then
        echo "[resume] skip completed: dataset=${dataset} variant=${variant} seed=${seed}"
        continue
      fi
      echo "[run] dataset=${dataset} variant=${variant} seed=${seed}"
      python main.py --conf "config/reviewer/${dataset}/${variant}.json" --dataset "$dataset" --gpuid "$GPU_ID" --seed "$seed"
    done
  done
}

prepare_scaling() {
  if [[ "$SKIP_PREPARE" == "1" ]]; then
    echo "[skip_prepare] reuse existing scaling datasets and configs"
    return
  fi
  python scripts/prepare_scaling_datasets.py --source "$SCALING_SOURCE" --ratios ${SCALING_RATIOS}
  python scripts/generate_round2_configs.py
}

prepare_d3_scaling() {
  if [[ "$SKIP_PREPARE" == "1" ]]; then
    echo "[skip_prepare] reuse existing D3 scaling dataset and configs"
    return
  fi
  python scripts/prepare_scaling_datasets.py --source PEMSD3-stream --target-prefix PEMSD3-scale --ratios 0.50
  python scripts/generate_round2_configs.py
}

for group in ${EXPERIMENT_GROUPS}; do
  case "$group" in
    r2_round2_100epoch_prepare)
      python scripts/generate_round2_100epoch_configs.py
      ;;
    r2_round2_pemsd3_100epoch_core)
      python scripts/generate_round2_100epoch_configs.py
      run_many "PEMSD3-stream" round2_comemnet_100epoch round2_current_retrained_100epoch
      ;;
    r2_round2_scaling_100epoch_core)
      python scripts/generate_round2_100epoch_configs.py
      for dataset in ${SCALING_DATASETS}; do
        run_many "$dataset" round2_scaling_comemnet_100epoch round2_scaling_current_retrained_100epoch
      done
      ;;
    r2_round2_scaling_prepare)
      prepare_scaling
      ;;
    r2_round2_scaling)
      prepare_scaling
      for dataset in ${SCALING_DATASETS}; do
        run_many "$dataset" round2_scaling_comemnet round2_scaling_current_retrained round2_scaling_all_history_retrained
      done
      ;;
    r2_round2_scaling_core)
      prepare_scaling
      for dataset in ${SCALING_DATASETS}; do
        run_many "$dataset" round2_scaling_comemnet round2_scaling_current_retrained
      done
      ;;
    r2_round2_scaling_stid)
      prepare_scaling
      for seed in ${SEEDS}; do
        GPU_ID="$GPU_ID" SEED="$seed" RESUME="$RESUME" MODELS="stid" PROTOCOLS="current" DATASETS="$SCALING_DATASETS" STID_BATCH_SIZE="${STID_BATCH_SIZE:-64}" \
          bash scripts/run_static_sota_retrained.sh
      done
      ;;
    r2_round2_d3_scaling_prepare)
      prepare_d3_scaling
      ;;
    r2_round2_d3_scaling)
      prepare_d3_scaling
      for dataset in ${D3_SCALING_DATASETS}; do
        run_many "$dataset" round2_scaling_comemnet round2_scaling_current_retrained round2_scaling_all_history_retrained
      done
      ;;
    r2_round2_d3_scaling_stid)
      prepare_d3_scaling
      for seed in ${SEEDS}; do
        GPU_ID="$GPU_ID" SEED="$seed" RESUME="$RESUME" MODELS="stid" PROTOCOLS="current" DATASETS="$D3_SCALING_DATASETS" STID_BATCH_SIZE="${STID_BATCH_SIZE:-64}" \
          bash scripts/run_static_sota_retrained.sh
      done
      ;;
    r2_round2_stid_efficiency)
      for seed in ${SEEDS}; do
        GPU_ID="$GPU_ID" SEED="$seed" RESUME="$RESUME" MODELS="stid" PROTOCOLS="current all_history" DATASETS="${STID_EFF_DATASETS:-PEMSD3-stream PEMSD4-large PEMSD8-mini}" STID_BATCH_SIZE="${STID_BATCH_SIZE:-64}" \
          bash scripts/run_static_sota_retrained.sh
      done
      ;;
    r2_round2_recent_static_baselines)
      for seed in ${SEEDS}; do
        GPU_ID="$GPU_ID" SEED="$seed" RESUME="$RESUME" MODELS="${RECENT_STATIC_MODELS:-dlinear patchtst}" PROTOCOLS="current" DATASETS="${RECENT_STATIC_DATASETS:-PEMSD3-stream PEMSD8-mini}" \
          DLINEAR_BATCH_SIZE="${DLINEAR_BATCH_SIZE:-128}" PATCHTST_BATCH_SIZE="${PATCHTST_BATCH_SIZE:-32}" ITRANSFORMER_BATCH_SIZE="${ITRANSFORMER_BATCH_SIZE:-16}" \
          bash scripts/run_static_sota_retrained.sh
      done
      ;;
    r2_round2_dataset_card)
      python scripts/build_dataset_card.py --datasets PEMSD3-stream PEMSD4-large PEMSD8-mini --output paper/reviewer/dataset_reproducibility_card.json
      ;;
    r2_round2_standard_benchmark_prepare)
      python scripts/prepare_pems08_standard.py
      ;;
    r2_round2_standard_benchmark_check)
      python scripts/check_standard_benchmark.py --datasets ${STANDARD_DATASETS:-PEMS08}
      ;;
    r2_round2_standard_benchmark)
      python scripts/prepare_pems08_standard.py
      python scripts/check_standard_benchmark.py --datasets ${STANDARD_DATASETS:-PEMS08}
      for seed in ${SEEDS}; do
        GPU_ID="$GPU_ID" SEED="$seed" RESUME="$RESUME" MODELS="${STANDARD_MODELS:-stid dlinear}" PROTOCOLS="current" DATASETS="${STANDARD_DATASETS:-PEMS08}" STID_BATCH_SIZE="${STID_BATCH_SIZE:-128}" DLINEAR_BATCH_SIZE="${DLINEAR_BATCH_SIZE:-128}" STATIC_RUN_TAG="${STATIC_RUN_TAG:-standard}" \
          bash scripts/run_static_sota_retrained.sh
      done
      ;;
    *)
      echo "Unknown round-2 group: $group" >&2
      echo "Use r2_round2_100epoch_prepare, r2_round2_pemsd3_100epoch_core, r2_round2_scaling_100epoch_core," >&2
      echo "r2_round2_scaling_prepare, r2_round2_scaling, r2_round2_scaling_core, r2_round2_scaling_stid," >&2
      echo "r2_round2_d3_scaling_prepare, r2_round2_d3_scaling, r2_round2_d3_scaling_stid," >&2
      echo "r2_round2_stid_efficiency, r2_round2_recent_static_baselines," >&2
      echo "r2_round2_dataset_card, r2_round2_standard_benchmark_prepare, r2_round2_standard_benchmark_check, or r2_round2_standard_benchmark" >&2
      exit 2
      ;;
  esac
done
