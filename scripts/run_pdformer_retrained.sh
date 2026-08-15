#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
DATASETS="${DATASETS:-PEMSD3-stream PEMSD8-mini PEMSD4-large}"
RESUME="${RESUME:-1}"
ALLOW_LARGE="${ALLOW_LARGE:-0}"
PREPROCESS="${PREPROCESS:-official}"
PATTERN_METHOD="${PATTERN_METHOD:-official}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-20}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -f "$ROOT/baseline/PDFormer-official/libcity/model/traffic_flow_prediction/PDFormer.py" ]]; then
  echo "ERROR: missing official PDFormer checkout at baseline/PDFormer-official" >&2
  echo "Clone: git clone --depth 1 https://github.com/BUAABIGSCity/PDFormer.git baseline/PDFormer-official" >&2
  exit 2
fi

python - <<'PY'
missing=[]
for module in ('fastdtw','tslearn'):
    try: __import__(module)
    except ImportError: missing.append(module)
if missing:
    raise SystemExit('Missing PDFormer preprocessing dependencies: '+', '.join(missing)+
                     '. Install with: pip install fastdtw tslearn')
PY

for dataset in $DATASETS; do
  cmd=(python "$ROOT/scripts/run_pdformer_retrained.py"
       --dataset "$dataset" --seed "$SEED" --gpu "$GPU_ID"
       --preprocess "$PREPROCESS" --pattern-method "$PATTERN_METHOD"
       --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --patience "$PATIENCE")
  [[ "$RESUME" == 1 ]] && cmd+=(--resume) || cmd+=(--no-resume)
  [[ "$ALLOW_LARGE" == 1 ]] && cmd+=(--allow-large)
  "${cmd[@]}"
done
