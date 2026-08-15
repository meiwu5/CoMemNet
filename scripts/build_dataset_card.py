#!/usr/bin/env python3
"""Build dataset reproducibility cards for evolving PeMS datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_npz(path: Path) -> dict:
    item = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path), "arrays": {}}
    with np.load(path, allow_pickle=True) as z:
        for key in z.files:
            arr = z[key]
            item["arrays"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return item


def dataset_card(dataset: str) -> dict:
    root = ROOT / "data" / dataset
    years = sorted(int(p.stem) for p in (root / "finaldata").glob("*.npz") if p.stem.isdigit())
    if not years:
        raise FileNotFoundError(f"No finaldata for {dataset}")
    card = {
        "dataset": dataset,
        "source": "CalTrans PeMS-derived evolving traffic sensor benchmark; PEMSD3-stream follows the public PeMS-stream-style split when available, PEMSD4/8 are curated from raw PeMS records.",
        "sampling_interval": "5 minutes",
        "history_steps": 12,
        "prediction_steps": 12,
        "sensor_filtering": {
            "missing_rate_threshold": "< 10%",
            "state_postmile_threshold": "< 100",
            "persistence": "sensor in period tau is retained only if it appears in tau+1 for offline benchmark curation; no future traffic values/labels are used during training",
        },
        "graph_construction": {
            "distance": "absolute State Post-Mile difference",
            "route_direction_filter": "edges are constructed only for comparable route/highway and travel direction",
            "epsilon": 1,
            "delta": 100,
            "usage": "metadata adjacency for dataset construction and optional update-set expansion; not prediction-backbone input",
        },
        "splits": "chronological train/validation/test splits stored in FastData/<year>_30day.npz",
        "years": {},
    }
    for year in years:
        final = inspect_npz(root / "finaldata" / f"{year}.npz")
        fast = inspect_npz(root / "FastData" / f"{year}_30day.npz")
        graph = inspect_npz(root / "graph" / f"{year}_adj.npz")
        node_count = final["arrays"].get("x", {}).get("shape", [None, None])[1]
        card["years"][str(year)] = {"nodes": node_count, "finaldata": final, "fastdata": fast, "graph": graph}
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["PEMSD3-stream", "PEMSD4-large", "PEMSD8-mini"])
    ap.add_argument("--output", default="paper/reviewer/dataset_reproducibility_card.json")
    args = ap.parse_args()
    payload = {dataset: dataset_card(dataset) for dataset in args.datasets}
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
