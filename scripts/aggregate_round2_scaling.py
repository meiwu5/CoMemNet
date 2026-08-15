#!/usr/bin/env python3
"""Aggregate round-2 scaling experiment summaries into JSON and CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import re


ROOT = Path(__file__).resolve().parents[1]


def latest_summary(root: Path, variant_prefix: str) -> Path | None:
    hits = sorted(root.glob(f"{variant_prefix}*/metrics/summary.json"))
    return hits[-1] if hits else None


def mean_metric(summary: dict, horizon: str, metric: str) -> float | None:
    key = metric.lower()
    h = str(horizon)
    if h in summary and key in summary[h]:
        vals = list(summary[h][key].values())
        return float(np.mean(vals)) if vals else None
    if h in summary.get("metrics", {}):
        return float(summary["metrics"][h][metric.upper()]["mean"])
    return None


def metric_from_log(summary_path: Path, horizon: int, metric: str) -> float | None:
    run_root = summary_path.parent.parent
    logs = sorted(run_root.glob("*.log"))
    if not logs:
        return None
    pat = re.compile(rf"\b{horizon}\s+{metric.lower()}\s+.*average:\s+([0-9.]+)")
    for line in logs[-1].read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            return float(m.group(1))
    return None


def efficiency(summary: dict) -> tuple[float | None, float | None]:
    rows = summary.get("efficiency_by_period") or {}
    if rows:
        return (
            float(sum(v.get("total_time", 0.0) for v in rows.values())),
            float(max(v.get("peak_memory_mb", 0.0) for v in rows.values())),
        )
    eff = summary.get("efficiency") or {}
    if eff:
        return (
            float(eff.get("cumulative_train_seconds")) if eff.get("cumulative_train_seconds") is not None else None,
            float(eff.get("peak_vram_mb")) if eff.get("peak_vram_mb") is not None else None,
        )
    return None, None


def nodes_for_dataset(dataset: str) -> int | None:
    manifest = ROOT / "data" / dataset / "scaling_manifest.json"
    if manifest.exists():
        return int(json.loads(manifest.read_text())["final_kept_nodes"])
    files = sorted((ROOT / "data" / dataset / "finaldata").glob("*.npz"))
    if not files:
        return None
    import numpy as np
    with np.load(files[-1]) as z:
        return int(z["x"].shape[1])


def add_row(rows: list[dict], dataset: str, method: str, summary: dict, summary_path: Path | None = None) -> None:
    train_s, vram = efficiency(summary)
    def mm(h: int, m: str) -> float | None:
        value = mean_metric(summary, str(h), m)
        if value is None and summary_path is not None:
            value = metric_from_log(summary_path, h, m)
        return value
    rows.append({
        "dataset": dataset,
        "final_nodes": nodes_for_dataset(dataset),
        "method": method,
        "seed": summary.get("seed"),
        "mae_3": mm(3, "MAE"),
        "rmse_3": mm(3, "RMSE"),
        "mae_6": mm(6, "MAE"),
        "rmse_6": mm(6, "RMSE"),
        "mae_12": mm(12, "MAE"),
        "rmse_12": mm(12, "RMSE"),
        "cumulative_train_seconds": train_s,
        "peak_vram_mb": vram,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["PEMSD4-scale25", "PEMSD4-scale50", "PEMSD4-scale75", "PEMSD4-large"])
    ap.add_argument("--output", default="res/reviewer/analysis/round2_scaling_summary.json")
    args = ap.parse_args()

    rows: list[dict] = []
    variants = {
        "CoMemNet": "round2_scaling_comemnet",
        "Current-only CoMemNet-backbone": "round2_scaling_current_retrained",
        "All-history CoMemNet-backbone": "round2_scaling_all_history_retrained",
    }
    for dataset in args.datasets:
        root = ROOT / "res" / "reviewer" / dataset
        for method, variant in variants.items():
            path = latest_summary(root, variant)
            if path:
                add_row(rows, dataset, method, json.loads(path.read_text()), path)
        extra = [
            ("STID-current-retrained", ROOT / "res" / "baseline" / "STID" / dataset / "stid-retrained-0" / "metrics" / "summary.json"),
            ("STID-all-history-retrained", ROOT / "res" / "baseline" / "STID" / dataset / "stid-all-history-retrained-0" / "metrics" / "summary.json"),
            ("DLinear-current-retrained", ROOT / "res" / "baseline" / "DLinear" / dataset / "dlinear-retrained-0" / "metrics" / "summary.json"),
            ("PatchTST-current-retrained", ROOT / "res" / "baseline" / "PatchTST-Lite" / dataset / "patchtst-retrained-0" / "metrics" / "summary.json"),
        ]
        for label, path in extra:
            if path.exists():
                add_row(rows, dataset, label, json.loads(path.read_text()), path)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print("saved", out)
    print("saved", csv_path)


if __name__ == "__main__":
    main()
