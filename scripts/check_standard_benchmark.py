#!/usr/bin/env python3
"""Check whether standard PEMS04/PEMS08 data are ready for round-2 sanity runs."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ready(dataset: str) -> bool:
    root = ROOT / "data" / dataset
    return (root / "FastData").exists() and any((root / "FastData").glob("*_30day.npz")) and (root / "finaldata").exists() and (root / "graph").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["PEMS08", "PEMS04"])
    args = ap.parse_args()
    missing = []
    for dataset in args.datasets:
        if ready(dataset):
            print(f"[ready] {dataset}: data/{dataset}/FastData, finaldata, graph found")
        else:
            missing.append(dataset)
            print(f"[missing] {dataset}: expected CoMemNet-format directories data/{dataset}/FastData, data/{dataset}/finaldata, data/{dataset}/graph")
    if missing:
        raise SystemExit("Standard benchmark data are not prepared: " + ", ".join(missing))


if __name__ == "__main__":
    main()
