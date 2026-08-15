# Static SOTA retraining baselines

The Reviewer 2 group uses official architectures under a common protocol:

- STAEformer (CIKM 2023), commit `fc49d39b2f1a8e3cf37b6289d7240680e1690f3f`;
- STID (CIKM 2022), commit `e8b313bc591bdd0101a1619962c9b503e75127c0`.

Every period starts from a fresh initialization, reads only that period's frozen
training split, selects a checkpoint on that period's validation split, and is
evaluated on that period's test split. Normalization statistics are fitted on
the current training split only.

Run the required workload (STID on all datasets):

```bash
GPU_ID=0 SEEDS="0" REVIEWER_GROUPS="r2_static_sota" RESUME=1 \
  bash scripts/run_revision_followup.sh
```

Run one model directly:

```bash
GPU_ID=0 SEED=0 MODELS="stid" DATASETS="PEMSD3-stream PEMSD4-large PEMSD8-mini" \
  RESUME=1 bash scripts/run_static_sota_retrained.sh
```

Outputs are stored under `res/baseline/STAEformer/` and `res/baseline/STID/`.

STAEformer is implemented and smoke-tested but optional. Its quadratic spatial
attention prevents a uniform comparison on the 2406-node PEMSD4 protocol.

STAEformer has quadratic spatial attention. PEMSD4 can be attempted explicitly
with `ALLOW_LARGE=1 BATCH_SIZE=1`, but an out-of-memory result should be reported
as a scalability boundary rather than silently changing the architecture.
