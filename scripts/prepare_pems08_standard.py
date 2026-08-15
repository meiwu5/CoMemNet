#!/usr/bin/env python3
"""Prepare the standard fixed-node PEMS08 benchmark in CoMemNet format.

This script is intended for the round-2 Reviewer-2 sanity check. PEMS08 is a
standard fixed-node benchmark, not an evolving-node benchmark. We therefore
create a single pseudo-period (2016) so existing static baseline runners can
reuse the CoMemNet FastData/finaldata/graph layout.

Default source: Zenodo record 7816008, files PEMS08.npz and PEMS08.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URLS = {
    "PEMS08.npz": "https://zenodo.org/records/7816008/files/PEMS08.npz?download=1",
    "PEMS08.csv": "https://zenodo.org/records/7816008/files/PEMS08.csv?download=1",
}
DEFAULT_MD5 = {
    "PEMS08.npz": "2a528d169c0d90294c9e24288a430132",
    "PEMS08.csv": "5e507b2b18d9235e95b63a20221cd4c5",
}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        print(f"[exists] {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"[download] {url} -> {path}")
    req = urllib.request.Request(url, headers={"User-Agent": "CoMemNet-PEMS08-prep/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(path)


def load_npz_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        if "data" in keys:
            arr = z["data"]
        elif "x" in keys:
            arr = z["x"]
        elif len(keys) == 1:
            arr = z[keys[0]]
        else:
            raise KeyError(f"Cannot infer data key from {path}; keys={keys}")
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected PEMS08 array with shape (T,N,F); got {arr.shape}")
    return arr


def parse_edges(csv_path: Path, num_nodes: int) -> tuple[np.ndarray, int]:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    distances = []
    rows = []
    with csv_path.open("r", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                i = int(float(row[0])); j = int(float(row[1]))
                dist = float(row[2]) if len(row) >= 3 and row[2] != "" else 1.0
            except ValueError:
                continue
            if 0 <= i < num_nodes and 0 <= j < num_nodes and i != j:
                rows.append((i, j, dist)); distances.append(dist)
    if not rows:
        # Fallback for datasets without adjacency: no spatial edges. Static sanity
        # baselines still run, and CoMemNet's prediction backbone is adjacency-free.
        return adj, 0
    delta = float(np.std(distances)) if np.std(distances) > 1e-6 else float(np.mean(distances) or 1.0)
    for i, j, dist in rows:
        weight = math.exp(-((dist / delta) ** 2)) if dist > 0 else 1.0
        adj[i, j] = max(adj[i, j], weight)
        adj[j, i] = max(adj[j, i], weight)
    return adj, len(rows)


def minmax_normalize(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    # input: B,T,N,1. Normalize traffic channel with training split only.
    train_bnft = np.transpose(train, (0, 2, 3, 1))
    val_bnft = np.transpose(val, (0, 2, 3, 1))
    test_bnft = np.transpose(test, (0, 2, 3, 1))
    mx = train_bnft.max(axis=(0, 1, 3), keepdims=True)
    mn = train_bnft.min(axis=(0, 1, 3), keepdims=True)
    denom = np.where((mx - mn) == 0, 1.0, mx - mn)
    def norm(a):
        b = np.transpose(a, (0, 2, 3, 1))
        b = 2.0 * ((b - mn) / denom) - 1.0
        return np.transpose(b, (0, 3, 1, 2)).astype(np.float32)
    return norm(train), norm(val), norm(test)


def make_fastdata(flow: np.ndarray, edge_index: np.ndarray, out_path: Path, x_len=12, y_len=12, train_rate=0.6, val_rate=0.2):
    # flow: T,N raw traffic feature.
    total_t, num_nodes = flow.shape
    time_ind = (np.arange(total_t) % 288) / 288.0
    day_ind = (np.arange(total_t) // 288) % 7
    data = np.stack([
        flow.astype(np.float32),
        np.tile(time_ind[:, None], (1, num_nodes)).astype(np.float32),
        np.tile(day_ind[:, None], (1, num_nodes)).astype(np.float32),
    ], axis=-1)
    xs, ys = [], []
    for t in range(x_len - 1, total_t - y_len):
        xs.append(data[t - x_len + 1:t + 1])
        ys.append(flow[t + 1:t + y_len + 1, :, None])
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.stack(ys, axis=0).astype(np.float32)
    n = x.shape[0]
    n_train = round(n * train_rate) - 1
    n_val = round(n * val_rate)
    n_test = n - n_train - n_val
    train_x, train_y = x[:n_train], y[:n_train]
    val_x, val_y = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    test_x, test_y = x[-n_test:], y[-n_test:]
    tx, vx, tex = minmax_normalize(train_x[..., :1], val_x[..., :1], test_x[..., :1])
    train_x = np.concatenate([tx, train_x[..., 1:]], axis=-1)
    val_x = np.concatenate([vx, val_x[..., 1:]], axis=-1)
    test_x = np.concatenate([tex, test_x[..., 1:]], axis=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
    return {
        "train_x": list(train_x.shape), "train_y": list(train_y.shape),
        "val_x": list(val_x.shape), "val_y": list(val_y.shape),
        "test_x": list(test_x.shape), "test_y": list(test_y.shape),
        "edge_index": list(edge_index.shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="PEMS08")
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw_standard" / "PEMS08")
    ap.add_argument("--output-root", type=Path, default=ROOT / "data")
    ap.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--skip-md5", action="store_true")
    ap.add_argument("--x-len", type=int, default=12)
    ap.add_argument("--y-len", type=int, default=12)
    args = ap.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DEFAULT_URLS.items():
        path = args.raw_dir / name
        if args.download:
            download(url, path, args.force_download)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Download it manually or rerun with --download.")
        if not args.skip_md5:
            got = md5sum(path)
            exp = DEFAULT_MD5[name]
            if got != exp:
                raise ValueError(f"MD5 mismatch for {path}: got {got}, expected {exp}")

    arr = load_npz_array(args.raw_dir / "PEMS08.npz")
    # Standard PEMS08 stores flow/occupancy/speed. Use flow as target feature.
    flow = arr[..., 0].astype(np.float32)
    num_nodes = flow.shape[1]
    adj, edge_rows = parse_edges(args.raw_dir / "PEMS08.csv", num_nodes)
    graph = nx.from_numpy_array(adj)
    edge_index = np.array(list(graph.edges), dtype=np.int64).T if graph.number_of_edges() else np.empty((2, 0), dtype=np.int64)

    out = args.output_root / args.dataset
    final_dir, fast_dir, graph_dir = out / "finaldata", out / "FastData", out / "graph"
    final_dir.mkdir(parents=True, exist_ok=True); fast_dir.mkdir(parents=True, exist_ok=True); graph_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(final_dir / f"{args.year}.npz", x=flow)
    np.savez_compressed(graph_dir / f"{args.year}_adj.npz", x=adj)
    shapes = make_fastdata(flow, edge_index, fast_dir / f"{args.year}_30day.npz", x_len=args.x_len, y_len=args.y_len)

    manifest = {
        "dataset": args.dataset,
        "source": "Zenodo record 7816008 TrafficDataSets; standard fixed-node PEMS08",
        "source_urls": DEFAULT_URLS,
        "source_md5": {k: md5sum(args.raw_dir / k) for k in DEFAULT_URLS},
        "year": args.year,
        "raw_shape": list(arr.shape),
        "used_feature": "feature 0 / traffic flow",
        "finaldata_shape": list(flow.shape),
        "nodes": int(num_nodes),
        "timesteps": int(flow.shape[0]),
        "edge_rows_in_csv": int(edge_rows),
        "undirected_edges_in_graph": int(edge_index.shape[1]),
        "fastdata_shapes": shapes,
        "protocol_note": "Fixed-node standard benchmark converted to one pseudo-period for sanity checks; not used as an evolving-node benchmark.",
    }
    (out / "standard_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
