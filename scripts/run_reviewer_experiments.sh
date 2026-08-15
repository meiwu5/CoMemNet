#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 1 2}"
RESUME="${RESUME:-1}"

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

# Priority 1: same-backbone sampler comparison on the smallest dataset.
run_many PEMSD3-stream sampler_feature sampler_random sampler_recency sampler_high_error \
    sampler_feature_l2 sampler_feature_kl sampler_feature_js sampler_feature_mmd

# Priority 2: target-branch ablations.
run_many PEMSD3-stream target_no_target target_online_sampler momentum_0_9 momentum_0_95 \
    momentum_0_99 momentum_0_995 loss_mae_only contrastive_weight_0_01 \
    contrastive_weight_0_05 contrastive_weight_0_1 contrastive_weight_0_2 \
    graph_selected_nodes_only graph_1_hop graph_2_hop graph_3_hop

# Priority 3: repeat the strongest reviewer-facing comparisons on larger datasets.
run_many PEMSD4-large sampler_feature sampler_random sampler_high_error sampler_feature_l2 \
    target_no_target target_online_sampler loss_mae_only graph_selected_nodes_only graph_2_hop

run_many PEMSD8-mini sampler_feature sampler_random sampler_high_error sampler_feature_l2 \
    target_no_target target_online_sampler loss_mae_only graph_selected_nodes_only graph_2_hop
