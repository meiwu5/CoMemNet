#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0}"
RESUME="${RESUME:-1}"
EXPERIMENT_GROUPS="${REVIEWER_GROUPS:-${1:-p0}}"

# Fail early when a Blackwell GPU is paired with an older CUDA wheel that has
# no sm_120 kernels (for example torch 2.5.1+cu124 on an RTX 5090).
python -c 'import sys, torch; cap=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None; arch=set(torch.cuda.get_arch_list()) if torch.cuda.is_available() else set(); bad=cap is not None and cap >= (12, 0) and "sm_120" not in arch; print("ERROR: this GPU requires a PyTorch CUDA 12.8+ wheel containing sm_120. Install with: pip install --upgrade torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128", file=sys.stderr) if bad else None; sys.exit(1 if bad else 0)'

is_complete() {
    local dataset="$1"
    local variant="$2"
    local seed="$3"
    local config_path="config/reviewer/${dataset}/${variant}.json"
    python -c 'import hashlib, json, pathlib, sys; dataset, variant, seed, root, config_path=sys.argv[1],sys.argv[2],int(sys.argv[3]),pathlib.Path(sys.argv[4]),pathlib.Path(sys.argv[5]); expected=hashlib.sha256(config_path.read_bytes()).hexdigest(); ok=False
for path in root.glob("**/metrics/summary.json"):
    try:
        record=json.loads(path.read_text())
        if record.get("dataset")==dataset and record.get("variant")==variant and int(record.get("seed",-1))==seed and record.get("config_hash")==expected:
            ok=True; break
    except (OSError, ValueError, TypeError):
        pass
sys.exit(0 if ok else 1)' "$dataset" "$variant" "$seed" "res/reviewer/$dataset" "$config_path"
}

run_many() {
    local dataset="$1"
    shift
    local variant
    local seed
    for variant in "$@"; do
        for seed in ${SEEDS}; do
            if [[ "${RESUME}" == "1" ]] && is_complete "${dataset}" "${variant}" "${seed}"; then
                echo "[resume] skip completed: dataset=${dataset} variant=${variant} seed=${seed}"
                continue
            fi
            echo "[run] dataset=${dataset} variant=${variant} seed=${seed}"
            python main.py \
                --conf "config/reviewer/${dataset}/${variant}.json" \
                --dataset "${dataset}" --gpuid "${GPU_ID}" --seed "${seed}"
        done
    done
}

for group in ${EXPERIMENT_GROUPS}; do
    case "${group}" in
        p0)
            # Core: selectors, target/momentum, replay. W1 repeats for diagnostics.
            run_many PEMSD3-stream sampler_feature sampler_random sampler_high_error \
                sampler_feature_l2 sampler_recency target_online_sampler \
                momentum_0_9 momentum_0_95 momentum_0_995 no_replay
            ;;
        p1)
            # Optional PEMSD3 distance and topology-scope ablations.
            run_many PEMSD3-stream sampler_feature_kl sampler_feature_js sampler_feature_mmd \
                graph_selected_nodes_only graph_1_hop graph_3_hop
            ;;
        p2)
            # Optional key sampler controls on the larger datasets.
            run_many PEMSD4-large sampler_random sampler_high_error sampler_feature_l2
            run_many PEMSD8-mini sampler_random sampler_high_error sampler_feature_l2
            ;;
        *) echo "Unknown group: ${group}; use p0, p1, or p2" >&2; exit 2 ;;
    esac
done
