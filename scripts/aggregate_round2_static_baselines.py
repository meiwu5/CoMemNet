#!/usr/bin/env python3
"""Aggregate static baseline round-2 runs into JSON/CSV tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "STID": ("stid", ["retrained", "all-history-retrained"]),
    "DLinear": ("dlinear", ["retrained"]),
    "PatchTST-Lite": ("patchtst", ["retrained"]),
    "iTransformer-Lite": ("itransformer", ["retrained"]),
}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def row_from(dataset: str, family: str, suffix: str, seed: int, summary: dict) -> dict:
    proto = "current" if suffix == "retrained" else "all_history"
    def metric(h, m): return summary["metrics"][str(h)][m]["mean"]
    eff = summary.get("efficiency", {}) or {}
    periods = summary.get("periods", [])
    if not eff and periods:
        eff = {
            "parameters": periods[-1].get("parameters"),
            "cumulative_train_seconds": float(sum(x.get("train_seconds", 0.0) for x in periods)),
            "peak_vram_mb": float(max(x.get("peak_vram_mb", 0.0) for x in periods)),
        }
    return {
        "dataset": dataset,
        "method": f"{family}-{proto}",
        "seed": seed,
        "protocol": summary.get("protocol"),
        "periods": len(periods),
        "final_nodes": periods[-1].get("nodes") if periods else None,
        "parameters": eff.get("parameters"),
        "cumulative_train_seconds": eff.get("cumulative_train_seconds"),
        "peak_vram_mb": eff.get("peak_vram_mb"),
        "mae_3": metric(3, "MAE"),
        "rmse_3": metric(3, "RMSE"),
        "mae_6": metric(6, "MAE"),
        "rmse_6": metric(6, "RMSE"),
        "mae_12": metric(12, "MAE"),
        "rmse_12": metric(12, "RMSE"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["PEMSD3-stream", "PEMSD4-large", "PEMSD8-mini"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--output", default="res/reviewer/analysis/round2_static_baselines.json")
    args = ap.parse_args()
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for family, (slug, suffixes) in FAMILIES.items():
                for suffix in suffixes:
                    path = ROOT / "res" / "baseline" / family / dataset / f"{slug}-{suffix}-{seed}" / "metrics" / "summary.json"
                    summary = load(path)
                    if summary:
                        rows.append(row_from(dataset, family, suffix, seed, summary))
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        grouped.setdefault(key, []).append(row)
    mean_rows = []
    numeric = ["cumulative_train_seconds", "peak_vram_mb", "mae_3", "rmse_3", "mae_6", "rmse_6", "mae_12", "rmse_12"]
    for (dataset, method), items in grouped.items():
        out = {"dataset": dataset, "method": method, "n_seeds": len(items), "final_nodes": items[-1]["final_nodes"], "parameters": items[-1]["parameters"]}
        for k in numeric:
            vals = [x[k] for x in items if x[k] is not None]
            out[k + "_mean"] = float(np.mean(vals)) if vals else None
            out[k + "_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0 if vals else None
        mean_rows.append(out)
    payload = {"rows": rows, "mean_rows": mean_rows}
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(mean_rows[0].keys()) if mean_rows else ["dataset", "method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(mean_rows)
    print("saved", out_path)
    print("saved", csv_path)


if __name__ == "__main__":
    main()
