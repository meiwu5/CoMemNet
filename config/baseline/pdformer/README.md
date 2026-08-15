# PDFormer static retraining protocol

- Official source: `baseline/PDFormer-official`, commit `f8c8f6ad007a04fad3baee958b89504711852ce9`.
- Architecture/hyperparameters: official `PeMS04.json` defaults.
- Split: existing frozen `data/<dataset>/FastData/<year>_30day.npz`.
- Training: a fresh random initialization for every period; no checkpoint is carried across periods.
- Selection: current-period validation split only; evaluation uses the current-period test split.
- Preprocessing: adjacency, DTW and pattern keys are recomputed and cached separately for each period.
- Output: `res/baseline/PDFormer/<dataset>/pdformer-retrained-<seed>/metrics/summary.json`.

Official preprocessing requires:

```bash
pip install -r requirements-pdformer.txt
```

Run the two tractable datasets first:

```bash
GPU_ID=0 SEED=0 DATASETS="PEMSD3-stream PEMSD8-mini" \
  bash scripts/run_pdformer_retrained.sh
```

PEMSD4(L) reaches 2406 nodes. PDFormer uses quadratic spatial attention and pairwise DTW, so the runner stops above 1200 nodes by default. After confirming resources, explicitly allow the large periods:

```bash
GPU_ID=0 SEED=0 DATASETS="PEMSD4-large" ALLOW_LARGE=1 \
  BATCH_SIZE=1 bash scripts/run_pdformer_retrained.sh
```

The same experiment is registered as Reviewer 2 group `r2_static_sota`:

```bash
GPU_ID=0 SEEDS="0" REVIEWER_GROUPS="r2_static_sota" RESUME=1 \
  bash scripts/run_revision_followup.sh
```

The optional `PREPROCESS=euclidean PATTERN_METHOD=kmeans` mode is only a scalability diagnostic and is labelled `PDFormer-scalable-proxy`; it must not be reported as official PDFormer.
