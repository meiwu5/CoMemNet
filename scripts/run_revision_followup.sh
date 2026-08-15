#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0}"
RESUME="${RESUME:-1}"
EXPERIMENT_GROUPS="${REVIEWER_GROUPS:-${1:-r1_forgetting}}"

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
        r1_forgetting)
            # Reviewer 1: distinguish adaptation from historical retention.
            run_many PEMSD3-stream forgetting_full forgetting_no_replay \
                forgetting_no_tmrb forgetting_static
            ;;
        r1_sampler_attribution)
            # Reviewer 1(b): test association with topology and ordinary flow changes.
            sampler_dir=$(find res/reviewer/PEMSD3-stream -maxdepth 2 -type d \
                -path '*/forgetting_full*/sampler' -print | sort | tail -1)
            if [[ -z "${sampler_dir}" ]]; then
                sampler_dir=$(find res/reviewer/PEMSD3-stream -maxdepth 2 -type d \
                    -path '*/sampler_feature*/sampler' -print | sort | tail -1)
            fi
            if [[ -z "${sampler_dir}" ]]; then
                echo "No W1 sampler diagnostics found; run r1_forgetting first." >&2
                exit 2
            fi
            output="res/reviewer/analysis/sampler_attribution_pemsd3_seed0.json"
            if [[ "${RESUME}" == "1" && -f "${output}" ]]; then
                echo "[resume] skip completed attribution: ${output}"
            else
                python scripts/analyze_sampler_attribution.py \
                    --dataset PEMSD3-stream --sampler-dir "${sampler_dir}" --output "${output}"
            fi
            ;;
        r2_retraining)
            run_many PEMSD3-stream forgetting_current_retrained all_history_retrained
            ;;
        r2_budget_sensitivity)
            run_many PEMSD3-stream budget_0_01 budget_0_02 budget_0_05 budget_0_1
            ;;
        r2_static_sota)
            # Reviewer 2: official STID, independently retrained per period.
            # STAEformer remains an optional direct runner because its spatial
            # attention is quadratic and cannot provide a uniform large-scale table.
            for seed in ${SEEDS}; do
                GPU_ID="${GPU_ID}" SEED="${seed}" RESUME="${RESUME}" \
                    MODELS="stid" DATASETS="${STID_DATASETS:-PEMSD3-stream PEMSD4-large PEMSD8-mini}" \
                    STID_BATCH_SIZE="${STID_BATCH_SIZE:-64}" \
                    bash scripts/run_static_sota_retrained.sh
            done
            ;;
        r3_forward_transfer)
            run_many PEMSD3-stream fwt_full fwt_no_replay fwt_static
            ;;
        r3_robustness)
            mkdir -p res/reviewer/analysis/robustness
            for spec in "forgetting_full:full" "forgetting_no_replay:no_replay" "forgetting_static:static"; do
                variant=${spec%%:*}; label=${spec##*:}
                run_root=$(find res/reviewer/PEMSD3-stream -maxdepth 1 -type d -name "${variant}*" | sort | tail -1)
                [[ -n "${run_root}" ]] || { echo "missing run: ${variant}" >&2; exit 2; }
                output="res/reviewer/analysis/robustness/${label}_seed0.json"
                if [[ "${RESUME}" == "1" && -f "${output}" ]]; then echo "[resume] skip ${output}"; else
                    python scripts/evaluate_robustness.py --run-root "${run_root}" \
                        --config "config/reviewer/PEMSD3-stream/${variant}.json" --output "${output}" --seed 0
                fi
            done
            ;;
        r1_efficiency_accounting)
            output="res/reviewer/analysis/efficiency_accounting_seed0.json"
            if [[ "${RESUME}" == "1" && -f "${output}" ]]; then echo "[resume] skip ${output}"; else
                python scripts/account_efficiency.py --output "${output}"
            fi
            ;;
        *)
            echo "Unknown group: ${group}" >&2
            echo "Use r1_forgetting, r1_sampler_attribution, r1_efficiency_accounting," >&2
            echo "r2_retraining, r2_budget_sensitivity, r2_static_sota," >&2
            echo "r3_forward_transfer, or r3_robustness" >&2
            exit 2
            ;;
    esac
done
