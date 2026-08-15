#!/usr/bin/env python3
"""Create node-scale variants for round-2 reviewer experiments.

The script constructs prefix-node subsets from an existing evolving PeMS
dataset.  It preserves the chronological periods and the expanding-node
property: for a target final-node budget B, period y keeps
min(N_y, B) nodes.  This makes the scaling study vary node count while keeping
the preprocessing, time splits, feature fields, and adjacency construction
format identical to the main benchmark.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def years_from_finaldata(dataset: str) -> list[int]:
    root = ROOT / "data" / dataset / "finaldata"
    years = sorted(int(p.stem) for p in root.glob("*.npz") if p.stem.isdigit())
    if not years:
        raise FileNotFoundError(f"No yearly npz files under {root}")
    return years


def subset_npz(src: Path, dst: Path, keep_nodes: int) -> dict[str, list[int] | str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    shapes = {}
    with np.load(src, allow_pickle=True) as z:
        for key in z.files:
            arr = z[key]
            original_shape = list(arr.shape)
            if src.parent.name == "finaldata" and arr.ndim >= 2 and arr.shape[1] >= keep_nodes:
                arr = arr[:, :keep_nodes, ...]
            elif src.parent.name == "FastData" and arr.ndim >= 3 and arr.shape[2] >= keep_nodes:
                arr = arr[:, :, :keep_nodes, ...]
            elif src.parent.name == "graph" and arr.ndim >= 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= keep_nodes:
                arr = arr[:keep_nodes, :keep_nodes]
            elif arr.ndim >= 2 and arr.shape[1] >= keep_nodes and key.lower().endswith("node"):
                arr = arr[:, :keep_nodes]
            payload[key] = arr
            shapes[key] = {"original": original_shape, "scaled": list(arr.shape)}
    np.savez_compressed(dst, **payload)
    return shapes


def copy_subset_dataset(source: str, ratio: float, target_prefix: str, force: bool) -> dict:
    years = years_from_finaldata(source)
    source_root = ROOT / "data" / source
    final_year = years[-1]
    with np.load(source_root / "finaldata" / f"{final_year}.npz") as z:
        final_nodes = int(z["x"].shape[1])
    final_keep = max(1, int(round(final_nodes * ratio)))
    target = f"{target_prefix}{int(round(ratio * 100)):02d}"
    target_root = ROOT / "data" / target
    if target_root.exists() and force:
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_dataset": source,
        "target_dataset": target,
        "ratio": ratio,
        "final_source_nodes": final_nodes,
        "final_kept_nodes": final_keep,
        "node_policy": "prefix subset of final-period node order; period y keeps min(N_y, final_kept_nodes)",
        "years": {},
    }
    for year in years:
        with np.load(source_root / "finaldata" / f"{year}.npz") as z:
            year_nodes = int(z["x"].shape[1])
        keep = min(year_nodes, final_keep)
        entry = {"source_nodes": year_nodes, "kept_nodes": keep}
        entry["finaldata_shapes"] = subset_npz(
            source_root / "finaldata" / f"{year}.npz",
            target_root / "finaldata" / f"{year}.npz",
            keep,
        )
        entry["fastdata_shapes"] = subset_npz(
            source_root / "FastData" / f"{year}_30day.npz",
            target_root / "FastData" / f"{year}_30day.npz",
            keep,
        )
        entry["graph_shapes"] = subset_npz(
            source_root / "graph" / f"{year}_adj.npz",
            target_root / "graph" / f"{year}_adj.npz",
            keep,
        )
        manifest["years"][str(year)] = entry

    manifest_path = target_root / "scaling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[scaling] wrote {target_root} final_keep={final_keep}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="PEMSD4-large")
    ap.add_argument("--target-prefix", default="PEMSD4-scale")
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for ratio in args.ratios:
        if not (0 < ratio <= 1):
            raise ValueError(f"ratio must be in (0,1], got {ratio}")
        copy_subset_dataset(args.source, ratio, args.target_prefix, args.force)


if __name__ == "__main__":
    main()
