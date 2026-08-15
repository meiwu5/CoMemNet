#!/usr/bin/env python3
"""Relate saved W1 node scores to topology and raw-traffic changes."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return {"rho": None, "pvalue": None, "n": int(mask.sum())}
    rho, p = spearmanr(a[mask], b[mask])
    return {"rho": float(rho), "pvalue": float(p), "n": int(mask.sum())}


def neighbor_jaccard_distance(a, b, node):
    na = set(np.flatnonzero(a[node]))
    nb = set(np.flatnonzero(b[node]))
    union = na | nb
    return 1.0 - len(na & nb) / len(union) if union else 0.0


def zscore(x):
    x = np.asarray(x, float)
    std = np.std(x)
    return (x - np.mean(x)) / std if std > 1e-12 else x - np.mean(x)


def top_overlap(score, reference, budget):
    budget = min(int(budget), len(score))
    left = set(np.argsort(score)[-budget:])
    right = set(np.argsort(reference)[-budget:])
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='PEMSD3-stream')
    ap.add_argument('--sampler-dir', required=True)
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--window', type=int, default=2016)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.data_root) / args.dataset
    rows = []
    for diagnostic in sorted(Path(args.sampler_dir).glob('*.json')):
        year = int(diagnostic.stem); previous = year - 1
        record = json.loads(diagnostic.read_text())
        scores = np.asarray(record.get('all_scores'), dtype=float)
        if scores.size == 0:
            continue
        a0 = np.load(root/'graph'/f'{previous}_adj.npz')['x']
        a1 = np.load(root/'graph'/f'{year}_adj.npz')['x']
        x0 = np.load(root/'finaldata'/f'{previous}.npz')['x'][-args.window:]
        x1 = np.load(root/'finaldata'/f'{year}.npz')['x'][-args.window:]
        n = min(len(scores), a0.shape[0], a1.shape[0], x0.shape[1], x1.shape[1])
        score = scores[:n]; a0, a1 = a0[:n,:n], a1[:n,:n]
        x0, x1 = x0[-min(len(x0),len(x1)):,:n], x1[-min(len(x0),len(x1)):,:n]
        degree_change = np.abs((a1 != 0).sum(1) - (a0 != 0).sum(1)).astype(float)
        adjacency_l1 = np.abs(a1-a0).sum(1)
        neighbor_change = np.asarray([neighbor_jaccard_distance(a0,a1,j) for j in range(n)])
        traffic_mean_change = np.abs(x1.mean(0)-x0.mean(0))
        traffic_std_change = np.abs(x1.std(0)-x0.std(0))
        traffic_l1 = np.mean(np.abs(zscore(x1)-zscore(x0)),axis=0)
        metrics = {
            'topology_degree_change': degree_change,
            'topology_adjacency_l1': adjacency_l1,
            'topology_neighbor_change': neighbor_change,
            'traffic_mean_change': traffic_mean_change,
            'traffic_std_change': traffic_std_change,
            'traffic_normalized_l1': traffic_l1,
        }
        budget = int(record.get('budget', max(1, round(n*.05))))
        rows.append({'year':year,'nodes':n,'budget':budget,
            'correlations':{k:safe_corr(score,v) for k,v in metrics.items()},
            'top_set_jaccard':{k:float(top_overlap(score,v,budget)) for k,v in metrics.items()}})
    aggregate={}
    keys=rows[0]['correlations'].keys() if rows else []
    for key in keys:
        rhos=[r['correlations'][key]['rho'] for r in rows if r['correlations'][key]['rho'] is not None]
        overlaps=[r['top_set_jaccard'][key] for r in rows]
        aggregate[key]={'mean_spearman_rho':float(np.mean(rhos)) if rhos else None,
                        'mean_top_set_jaccard':float(np.mean(overlaps)) if overlaps else None}
    output={'dataset':args.dataset,'sampler_dir':args.sampler_dir,
            'interpretation':'association only; not causal topology attribution',
            'periods':rows,'aggregate':aggregate}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2))
    print('saved',out)
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__': main()
