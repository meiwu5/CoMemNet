#!/usr/bin/env python3
"""Run the official PDFormer model as a per-period static retraining baseline.

The adapter deliberately reuses CoMemNet's frozen NPZ splits.  Every period is
trained from a fresh random initialization; no previous-period weights or
future-period observations are available.  PDFormer's architecture and PeMS04
hyperparameters are retained, while preprocessing caches are scoped by
dataset/year to avoid continual-learning leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baseline" / "PDFormer-official"
OFFICIAL_COMMIT = "f8c8f6ad007a04fad3baee958b89504711852ce9"
DATASETS = {
    "PEMSD3-stream": (2011, 2017),
    "PEMSD4-large": (2009, 2015),
    "PEMSD8-mini": (2012, 2018),
}


class FrozenSplit(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).float(), torch.from_numpy(self.y[index]).float()


class TorchMinMax11:
    """PDFormer-compatible scaler supporting numpy arrays and torch tensors."""

    def __init__(self, minimum: float, maximum: float):
        self.min = float(minimum)
        self.max = float(maximum)
        if not self.max > self.min:
            raise ValueError(f"invalid training range: min={self.min}, max={self.max}")

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2.0 - 1.0

    def inverse_transform(self, data):
        return ((data + 1.0) / 2.0) * (self.max - self.min) + self.min


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_frozen_split(dataset: str, year: int):
    path = ROOT / "data" / dataset / "FastData" / f"{year}_30day.npz"
    with np.load(path) as z:
        arrays = {key: z[key] for key in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")}
    return path, arrays


def make_pdformer_features(x: np.ndarray) -> np.ndarray:
    """Convert CoMemNet [traffic,time-of-day,day-index] to PDFormer features."""
    traffic = x[..., :1].astype(np.float32, copy=False)
    tod = x[..., 1:2].astype(np.float32, copy=False)
    day_index = np.rint(x[..., 2]).astype(np.int64) % 7
    dow = np.eye(7, dtype=np.float32)[day_index]
    return np.concatenate((traffic, tod, dow), axis=-1)


def training_range(dataset: str, year: int, num_train_samples: int, input_window: int = 12):
    """Recover the exact train-X min/max used by CoMemNet MinMaxnormalization."""
    raw_path = ROOT / "data" / dataset / "finaldata" / f"{year}.npz"
    with np.load(raw_path) as z:
        raw = z["x"]
        # train sample t uses raw[t:t+input_window]; union is [0, num_train+input_window-1).
        train_raw = raw[: num_train_samples + input_window - 1]
        minimum = float(np.nanmin(train_raw))
        maximum = float(np.nanmax(train_raw))
    return raw_path, minimum, maximum


def hop_matrix(adj: np.ndarray, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    graph = csr_matrix((adj > 0).astype(np.float32))
    hops = shortest_path(graph, directed=False, unweighted=True)
    hops[~np.isfinite(hops)] = 511
    hops = np.minimum(hops, 511).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, hops)
    return hops


def daily_profile(dataset: str, year: int, nodes: int) -> np.ndarray:
    with np.load(ROOT / "data" / dataset / "finaldata" / f"{year}.npz") as z:
        raw = np.asarray(z["x"][: 30 * 288, :nodes], dtype=np.float32)
    days = raw.shape[0] // 288
    return raw[: days * 288].reshape(days, 288, nodes).mean(axis=0).T


def dtw_matrix(dataset: str, year: int, nodes: int, cache: Path, mode: str) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    profile = daily_profile(dataset, year, nodes)
    if mode == "official":
        try:
            from fastdtw import fastdtw
        except ImportError as exc:
            raise RuntimeError("official DTW requires `pip install fastdtw`") from exc
        matrix = np.zeros((nodes, nodes), dtype=np.float32)
        for i in range(nodes):
            if i % 25 == 0:
                print(f"[dtw] {dataset} {year}: {i}/{nodes}", flush=True)
            for j in range(i, nodes):
                matrix[i, j] = fastdtw(profile[i], profile[j], radius=6)[0]
                matrix[j, i] = matrix[i, j]
    else:
        # Explicit scalability fallback, never labelled as official PDFormer in outputs.
        # Squared Euclidean distances preserve full 288-step daily profiles.
        sq = np.sum(profile * profile, axis=1, keepdims=True)
        matrix = np.maximum(sq + sq.T - 2 * profile @ profile.T, 0).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, matrix)
    return matrix


def pattern_keys(x_train: np.ndarray, cache: Path, clusters: int, seed: int, method: str):
    if cache.exists():
        return np.load(cache)
    candidates = x_train[:, :3, :, :1].swapaxes(1, 2).reshape(-1, 3, 1)
    # Official code uses at most 14 days. Frozen split has sample windows rather than raw timestamps.
    candidates = candidates[: min(len(candidates), 14 * 288 * x_train.shape[2])]
    if method == "official":
        try:
            from tslearn.clustering import KShape
        except ImportError as exc:
            raise RuntimeError("official pattern clustering requires `pip install tslearn`") from exc
        centers = KShape(n_clusters=clusters, max_iter=5, random_state=seed).fit(candidates).cluster_centers_
    else:
        km = KMeans(n_clusters=clusters, max_iter=100, n_init=10, random_state=seed)
        centers = km.fit(candidates.reshape(len(candidates), -1)).cluster_centers_.reshape(clusters, 3, 1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, centers.astype(np.float32))
    return centers.astype(np.float32)


def laplacian_positional_encoding(adj: np.ndarray, dim: int, device):
    degree = np.sum(adj, axis=1)
    inv = np.zeros_like(degree, dtype=np.float64)
    inv[degree > 0] = degree[degree > 0] ** -0.5
    lap = np.eye(adj.shape[0]) - inv[:, None] * adj * inv[None, :]
    _, vectors = np.linalg.eigh(lap)
    vectors = vectors[:, 1 : dim + 1]
    if vectors.shape[1] < dim:
        vectors = np.pad(vectors, ((0, 0), (0, dim - vectors.shape[1])))
    return torch.from_numpy(vectors.astype(np.float32)).to(device)


def masked_metrics(truth: np.ndarray, pred: np.ndarray):
    result = {}
    for horizon in (3, 6, 12):
        y, p = truth[:, :horizon], pred[:, :horizon]
        mask = y != 0
        error = p - y
        mae = np.abs(error)[mask].mean()
        rmse = np.sqrt(np.square(error)[mask].mean())
        nz = mask & (np.abs(y) > 1e-5)
        mape = (np.abs(error[nz] / y[nz]).mean() * 100) if nz.any() else float("nan")
        result[str(horizon)] = {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}
    return result


def run_period(args, dataset: str, year: int, device, official_config: dict, root: Path):
    output = root / str(year) / "metrics.json"
    if args.resume and output.exists():
        print(f"[resume] skip PDFormer: dataset={dataset} year={year} seed={args.seed}")
        return json.loads(output.read_text())

    split_path, data = load_frozen_split(dataset, year)
    nodes = data["train_x"].shape[2]
    if nodes > args.max_nodes and not args.allow_large:
        raise RuntimeError(
            f"{dataset}/{year} has {nodes} nodes > --max-nodes={args.max_nodes}; "
            "PDFormer's O(N^2) attention is likely impractical. Re-run with --allow-large after checking memory."
        )
    raw_path, minimum, maximum = training_range(dataset, year, len(data["train_x"]))
    scaler = TorchMinMax11(minimum, maximum)
    for key in ("train_x", "val_x", "test_x"):
        data[key] = make_pdformer_features(data[key])
    for key in ("train_y", "val_y", "test_y"):
        data[key] = scaler.transform(data[key].astype(np.float32))

    adj_path = ROOT / "data" / dataset / "graph" / f"{year}_adj.npz"
    with np.load(adj_path) as z:
        adj = np.asarray(z["x"], dtype=np.float32)
    cache = ROOT / "res" / "baseline" / "PDFormer" / "cache" / dataset / str(year)
    sh_mx = hop_matrix(adj, cache / "hop.npy")
    dtw = dtw_matrix(dataset, year, nodes, cache / f"dtw_{args.preprocess}.npy", args.preprocess)
    keys = pattern_keys(data["train_x"], cache / f"patterns_{args.pattern_method}_seed{args.seed}.npy", 16, args.seed, args.pattern_method)

    sys.path.insert(0, str(OFFICIAL))
    from libcity.model.traffic_flow_prediction.PDFormer import PDFormer

    config = dict(official_config)
    config.update({
        "dataset": f"{dataset}-{year}", "device": device, "world_size": 1,
        "input_window": 12, "output_window": 12, "output_dim": 1,
        "add_time_in_day": True, "add_day_in_week": True,
        "max_epoch": args.epochs, "set_loss": args.loss,
        "use_curriculum_learning": False,
    })
    train_loader = DataLoader(FrozenSplit(data["train_x"], data["train_y"]), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(FrozenSplit(data["val_x"], data["val_y"]), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(FrozenSplit(data["test_x"], data["test_y"]), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    feature = {
        "scaler": scaler, "adj_mx": adj, "sd_mx": None, "sh_mx": sh_mx,
        "ext_dim": 8, "num_nodes": nodes, "feature_dim": 9, "output_dim": 1,
        "num_batches": len(train_loader), "dtw_matrix": dtw, "pattern_keys": keys,
    }
    seed_everything(args.seed)
    model = PDFormer(config, feature).to(device)
    lap = laplacian_positional_encoding(adj, config.get("lape_dim", 8), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    best, stale, elapsed, peak = math.inf, 0, 0.0, 0.0
    checkpoint = root / str(year) / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); started = time.time(); total = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model({"X": x}, lap)
            loss = model.calculate_loss_without_predict(y, pred, batches_seen=epoch * len(train_loader), set_loss=args.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); total += float(loss); count += 1
        elapsed += time.time() - started
        model.eval(); values = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model({"X": x}, lap)
                values.append(float(model.calculate_loss_without_predict(y, pred, batches_seen=0, set_loss="masked_mae")))
        val = float(np.mean(values)); scheduler.step()
        if device.type == "cuda": peak = max(peak, torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        print(f"[PDFormer] {dataset}/{year} epoch={epoch:03d} train={total/max(count,1):.4f} val_mae={val:.4f}", flush=True)
        if val < best - 1e-5:
            best, stale = val, 0
            torch.save({"model": model.state_dict(), "config": config}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience: break

    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    model.eval(); predictions, truths = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model({"X": x.to(device)}, lap)
            predictions.append(scaler.inverse_transform(pred).cpu().numpy())
            truths.append(scaler.inverse_transform(y.to(device)).cpu().numpy())
    metrics = masked_metrics(np.concatenate(truths), np.concatenate(predictions))
    payload = {
        "baseline": "PDFormer" if args.preprocess == args.pattern_method == "official" else "PDFormer-scalable-proxy",
        "official_commit": OFFICIAL_COMMIT, "protocol": "per-period static retraining; frozen CoMemNet split",
        "dataset": dataset, "year": year, "seed": args.seed, "nodes": nodes,
        "split": str(split_path.relative_to(ROOT)), "raw": str(raw_path.relative_to(ROOT)), "adjacency": str(adj_path.relative_to(ROOT)),
        "preprocess": args.preprocess, "pattern_method": args.pattern_method,
        "parameters": sum(p.numel() for p in model.parameters()), "train_seconds": elapsed,
        "peak_vram_mb": peak, "best_val_mae": best, "metrics": metrics,
    }
    atomic_json(output, payload)
    return payload


def aggregate(dataset: str, seed: int, periods: list[dict], root: Path):
    summary = {"baseline": periods[0]["baseline"], "dataset": dataset, "seed": seed,
               "protocol": periods[0]["protocol"], "periods": periods, "metrics": {}}
    for horizon in ("3", "6", "12"):
        summary["metrics"][horizon] = {}
        for metric in ("MAE", "RMSE", "MAPE"):
            values = [p["metrics"][horizon][metric] for p in periods]
            summary["metrics"][horizon][metric] = {"per_period": values, "mean": float(np.mean(values))}
    atomic_json(root / "metrics" / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", default="huber", choices=("huber", "masked_mae"))
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preprocess", choices=("official", "euclidean"), default="official")
    parser.add_argument("--pattern-method", choices=("official", "kmeans"), default="official")
    parser.add_argument("--max-nodes", type=int, default=1200)
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args()
    if not OFFICIAL.exists():
        raise SystemExit("missing baseline/PDFormer-official; run scripts/run_pdformer_retrained.sh after cloning")
    begin, end = DATASETS[args.dataset]
    years = args.years or list(range(begin, end + 1))
    invalid = [y for y in years if not begin <= y <= end]
    if invalid: raise SystemExit(f"invalid years for {args.dataset}: {invalid}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    official_config = json.loads((OFFICIAL / "PeMS04.json").read_text())
    root = ROOT / "res" / "baseline" / "PDFormer" / args.dataset / f"pdformer-retrained-{args.seed}"
    periods = [run_period(args, args.dataset, year, device, official_config, root) for year in years]
    summary = aggregate(args.dataset, args.seed, periods, root)
    print(json.dumps({"output": str(root / 'metrics' / 'summary.json'), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
