import sys
sys.path.append('src/')
import numpy as np
from scipy.stats import entropy as kldiv
from utils.dataloader import Cotinual_learning_DataLoader
import torch
from scipy.spatial import distance
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction


def contrastive_loss(embedding1, embedding2, temperature=0.1):
    sim_matrix = F.cosine_similarity(embedding1.unsqueeze(1), embedding2.unsqueeze(0), dim=-1)
    sim_matrix = sim_matrix / temperature
    labels = torch.arange(sim_matrix.size(0)).to(embedding1.device)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    window = 288 * 7
    segment = data[-window-1:-1, :] if data.shape[0] > window else data[-window:, :]
    usable = (segment.shape[0] // args.x_len) * args.x_len
    data = np.reshape(segment[-usable:, :], (-1, args.x_len, node_size, 3))
    dataloader = Cotinual_learning_DataLoader(data, batch_size=data.shape[0], shuffle=True,pad_with_last_sample=True)
    for batch_idx, data in enumerate(dataloader.get_iterator()):
        if getattr(args, "sampler_branch", "target") == "online":
            feature = model.online_branch(data, args.year)
        else:
            feature = model.target_branch(data,args.year)
        return feature.cpu().detach().numpy()


def get_current(data, graph, args, model, adj):
    node_size = data.shape[1]
    window = 288 * 7
    segment = data[-window-1:-1, :] if data.shape[0] > window else data[-window:, :]
    usable = (segment.shape[0] // args.x_len) * args.x_len
    data = np.reshape(segment[-usable:, :], (-1, args.x_len, node_size, 3))
    dataloader = Cotinual_learning_DataLoader(data, batch_size=data.shape[0], shuffle=True,pad_with_last_sample=True)
    for batch_idx, data in enumerate(dataloader.get_iterator()):
        feature = model.online_branch(data,args.year)
        return feature.cpu().detach().numpy()


def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)


def _top_nodes(score, topm):
    score = np.asarray(score, dtype=np.float64)
    if score.size == 0:
        return []
    topm = min(int(topm), score.size)
    return np.argsort(score)[-topm:].tolist()


def _record_selection(args, scores, selected):
    args.last_sampler_scores = None if scores is None else np.asarray(scores, dtype=np.float64).tolist()
    args.last_sampler_nodes = list(map(int, selected))
    return args.last_sampler_nodes


def _safe_prob(values, bins=10, value_range=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if value_range is None:
        lo, hi = np.nanmin(values), np.nanmax(values)
    else:
        lo, hi = value_range
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones(bins, dtype=np.float64) / bins
    hist, _ = np.histogram(values, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64) + 1e-12
    return hist / hist.sum()


def _js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kldiv(p, m) + 0.5 * kldiv(q, m)


def _mmd_rbf(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if x.size == 0 or y.size == 0:
        return 0.0
    xy = np.vstack([x, y]).reshape(-1)
    sigma = np.std(xy)
    if sigma <= 1e-12 or not np.isfinite(sigma):
        sigma = 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    xx = np.exp(-gamma * (x - x.T) ** 2).mean()
    yy = np.exp(-gamma * (y - y.T) ** 2).mean()
    xy_kernel = np.exp(-gamma * (x - y.T) ** 2).mean()
    return float(xx + yy - 2.0 * xy_kernel)


def _node_vectors(data):
    # Input: [batch, horizon, node, channel] or [time, node, channel].
    if data.ndim == 4:
        return np.transpose(data, (2, 0, 1, 3)).reshape(data.shape[2], -1)
    if data.ndim == 3:
        return np.transpose(data, (1, 0, 2)).reshape(data.shape[1], -1)
    if data.ndim == 2:
        return data.T
    raise ValueError("Unsupported data shape for node vectors: {}".format(data.shape))


def _distribution_score(pre_vec, cur_vec, strategy):
    if strategy in ("feature_l2", "l2"):
        size = min(pre_vec.size, cur_vec.size)
        return float(np.linalg.norm(pre_vec[:size] - cur_vec[:size]))
    if strategy in ("feature_cosine", "cosine"):
        size = min(pre_vec.size, cur_vec.size)
        if size == 0:
            return 0.0
        denom = np.linalg.norm(pre_vec[:size]) * np.linalg.norm(cur_vec[:size])
        if denom <= 1e-12:
            return 0.0
        return float(1.0 - np.dot(pre_vec[:size], cur_vec[:size]) / denom)

    lo = min(np.nanmin(pre_vec), np.nanmin(cur_vec))
    hi = max(np.nanmax(pre_vec), np.nanmax(cur_vec))
    p = _safe_prob(pre_vec, value_range=(lo, hi))
    q = _safe_prob(cur_vec, value_range=(lo, hi))
    if strategy in ("feature_kl", "kl", "original"):
        return float(kldiv(p, q))
    if strategy in ("feature_js", "js"):
        return float(_js_divergence(p, q))
    if strategy in ("feature_mmd", "mmd"):
        return _mmd_rbf(pre_vec, cur_vec)
    support = np.linspace(lo, hi, num=p.size, endpoint=True)
    return float(wasserstein_distance(support, support, u_weights=p, v_weights=q))


def _feature_based_selection(model, args, pre_data, cur_data, pre_graph, cur_graph, strategy):
    model.eval()
    pre_feature = get_feature(pre_data, pre_graph, args, model, None)
    cur_feature = get_current(cur_data, cur_graph, args, model, None)
    num_nodes = min(pre_feature.shape[2], cur_feature.shape[2])
    pre_vectors = _node_vectors(pre_feature[:, :, :num_nodes, :])
    cur_vectors = _node_vectors(cur_feature[:, :, :num_nodes, :])
    score = [
        _distribution_score(pre_vectors[i], cur_vectors[i], strategy)
        for i in range(num_nodes)
    ]
    selected = _top_nodes(score, args.topm)
    return _record_selection(args, score, selected)


def _raw_distribution_selection(args, pre_data, cur_data, strategy):
    window = int(getattr(args, "replay_window_steps", 288 * 7))
    pre_recent = pre_data[-window:]
    cur_recent = cur_data[-window:]
    num_nodes = min(pre_recent.shape[1], cur_recent.shape[1])
    pre_vectors = _node_vectors(pre_recent[:, :num_nodes, :])
    cur_vectors = _node_vectors(cur_recent[:, :num_nodes, :])
    score = [
        _distribution_score(pre_vectors[i], cur_vectors[i], strategy)
        for i in range(num_nodes)
    ]
    selected = _top_nodes(score, args.topm)
    return _record_selection(args, score, selected)


def _random_selection(args, num_nodes):
    rng = np.random.default_rng(int(getattr(args, "seed", 0)) + int(args.year))
    topm = min(int(args.topm), int(num_nodes))
    score = rng.random(num_nodes)
    selected = _top_nodes(score, topm)
    return _record_selection(args, score, selected)


def _recency_selection(args, pre_data, cur_data):
    steps = int(getattr(args, "recency_steps", 288))
    pre_recent = pre_data[-steps:]
    cur_recent = cur_data[-steps:]
    num_nodes = min(pre_recent.shape[1], cur_recent.shape[1])
    size = min(pre_recent.shape[0], cur_recent.shape[0])
    score = np.mean(
        np.abs(cur_recent[-size:, :num_nodes, 0] - pre_recent[-size:, :num_nodes, 0]),
        axis=0,
    )
    selected = _top_nodes(score, args.topm)
    return _record_selection(args, score, selected)


def _high_error_selection(model, args, cur_data):
    model.eval()
    x_len = int(args.x_len)
    y_len = int(args.y_len)
    max_windows = int(getattr(args, "high_error_windows", 256))
    start_min = max(0, cur_data.shape[0] - (max_windows + x_len + y_len))
    starts = range(start_min, cur_data.shape[0] - x_len - y_len, max(1, y_len))
    xs, ys = [], []
    for start in starts:
        xs.append(cur_data[start:start + x_len])
        ys.append(cur_data[start + x_len:start + x_len + y_len, :, :1])
    if not xs:
        return []
    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)
    dataloader = Cotinual_learning_DataLoader(
        xs, batch_size=min(len(xs), int(args.batch_size)), shuffle=False, pad_with_last_sample=False
    )
    offset = 0
    errors = []
    with torch.no_grad():
        for data in dataloader.get_iterator():
            pred = model.online_branch(data, args.year).cpu().numpy()
            batch_size = pred.shape[0]
            truth = ys[offset:offset + batch_size]
            offset += batch_size
            errors.append(np.mean(np.abs(pred - truth), axis=(0, 1, 3)))
    score = np.mean(np.stack(errors, axis=0), axis=0)
    selected = _top_nodes(score, args.topm)
    return _record_selection(args, score, selected)


def score_func(pre_data, cur_data, args):
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))
    return np.argpartition(np.asarray(score), -args.topm)[-args.topm:]



def visualize_distributions(save_dis, top_nodes, args):
    years = getattr(args, 'years', [args.year])

    for year in years:
        plt.figure(figsize=(25, 10 * len(top_nodes)))

        for idx, node in enumerate(top_nodes):
            if (node, 0) in save_dis:
                pre_prob, cur_prob = save_dis[(node, 0)]
                bins = np.linspace(0, 1.0, 11)
                bin_centers = bins[:-1]

                pre_positions = bin_centers - 0.02
                cur_positions = bin_centers + 0.02
                gap = 0.05

                plt.subplot(len(top_nodes), 1, idx + 1)
                # plt.bar(pre_positions, pre_prob, width=0.04, alpha=0.5, label=f'Previous', color='blue')
                # plt.bar(cur_positions, cur_prob, width=0.04, alpha=0.5, label=f'Current', color='orange')
                # plt.bar(pre_positions, pre_prob, width=0.04, alpha=0.5, label=f'Previous', color='teal')
                # plt.bar(cur_positions, cur_prob, width=0.04, alpha=0.5, label=f'Current', color='salmon')
                plt.bar(pre_positions, pre_prob, width=0.04, alpha=1, label=f'Previous', color='peachpuff')
                plt.bar(cur_positions, cur_prob, width=0.04, alpha=1, label=f'Current', color='lightskyblue')
                plt.xlabel('Normalized Range', fontsize=30)
                plt.ylabel('Density', fontsize=30)
                plt.title(f'Distribution for Node {node} at {year} year', fontsize=30)
                plt.legend(fontsize=30)
                plt.grid(True, alpha=0.3)
                fraction_labels = [f"{int(x * 10)}/10" for x in bins]
                plt.xticks(bins, fraction_labels, rotation=0, fontsize=30)
                plt.yticks(fontsize=30)

        plt.tight_layout()
        plt.savefig(f'/root/autodl-fs/CoMemNet/figure/PEMSD8/node_distributions_{year}.png')
        plt.close()
        print(f"Adjusted distribution plot for year {year} saved as 'node_distributions_{year}.png'")


def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    save_dis = {}
    strategy = getattr(args, "replay_strategy", "feature")
    num_nodes = min(pre_data.shape[1], cur_data.shape[1])
    if strategy == "random":
        return _random_selection(args, num_nodes)
    if strategy == "recency":
        return _recency_selection(args, pre_data, cur_data)
    if strategy == "high_error":
        return _high_error_selection(model, args, cur_data)
    if strategy in ("raw_l2", "raw_cosine", "raw_kl", "raw_js", "raw_mmd", "raw_wasserstein"):
        return _raw_distribution_selection(args, pre_data, cur_data, strategy.replace("raw_", "feature_"))

    if strategy == 'original':
        pre_data = pre_data[-288*7-1:-1,:]
        cur_data = cur_data[-288*7-1:-1,:]
        node_size = pre_data.shape[1]
        score = []
        for node in range(node_size):
            max_val = max(np.max(pre_data[:,node,:]), np.max(cur_data[:,node,:]))
            min_val = min(np.min(pre_data[:,node,:]), np.min(cur_data[:,node,:]))
            pre_prob, _ = np.histogram(pre_data[:,node,:], bins=10, range=(min_val, max_val))
            pre_prob = pre_prob *1.0 / sum(pre_prob)
            cur_prob, _ = np.histogram(cur_data[:,node,:], bins=10, range=(min_val, max_val))
            cur_prob = cur_prob * 1.0 /sum(cur_prob)
            score.append(kldiv(pre_prob, cur_prob))
        selected = _top_nodes(score, args.topm)
        return _record_selection(args, score, selected)

    elif strategy in ('feature', 'feature_wasserstein', 'feature_l2', 'feature_cosine', 'feature_kl', 'feature_js', 'feature_mmd'):
        if strategy != "feature":
            return _feature_based_selection(model, args, pre_data, cur_data, pre_graph, cur_graph, strategy)

        model.eval()
        pre_data = get_feature(pre_data, pre_graph, args, model, None)
        cur_data = get_current(cur_data, cur_graph, args, model, None)


        num_nodes = min(pre_data.shape[2], cur_data.shape[2])
        pre_data = pre_data[:, :, :num_nodes, :]
        cur_data = cur_data[:, :, :num_nodes, :]
        print("Aligned num_nodes:", num_nodes)

        score = []
        save_dis = {}

        for i in range(num_nodes):
            score_ = 0.0
            for j in range(pre_data.shape[1]):
                try:
                    if np.max(pre_data[:, j, i, 0]) == np.min(pre_data[:, j, i, 0]):
                        print(f"Warning: Node {i}, timestep {j} has constant value")
                        continue
                    shared_min = min(np.min(pre_data[:, j, i, 0]), np.min(cur_data[:, j, i, 0]))
                    shared_max = max(np.max(pre_data[:, j, i, 0]), np.max(cur_data[:, j, i, 0]))
                    scale = max(shared_max - shared_min, 1e-12)
                    pre_data[:, j, i, 0] = (pre_data[:, j, i, 0] - shared_min) / scale
                    cur_data[:, j, i, 0] = (cur_data[:, j, i, 0] - shared_min) / scale

                    pre_prob, _ = np.histogram(pre_data[:, j, i, 0], bins=10, range=(0, 1), density=True)
                    cur_prob, _ = np.histogram(cur_data[:, j, i, 0], bins=10, range=(0, 1), density=True)

                    save_dis[(i, j)] = [pre_prob, cur_prob]
                    support = np.linspace(0.05, 0.95, num=10)
                    pre_mass = pre_prob / max(pre_prob.sum(), 1e-12)
                    cur_mass = cur_prob / max(cur_prob.sum(), 1e-12)
                    score_ += wasserstein_distance(support, support, u_weights=pre_mass, v_weights=cur_mass)
                except Exception as e:
                    print(f"Error for node {i}, timestep {j}: {e}")
                    continue
            score.append(score_)

        if getattr(args, "save_sampler_debug", False):
            with open('save_dis.pkl', 'wb') as f:
                pickle.dump(save_dis, f)

        args.topm = min(args.topm, len(score))
        top_nodes = np.argsort(score)[-args.topm:].tolist()

        if getattr(args, "save_sampler_plot", False):
            visualize_distributions(save_dis, top_nodes, args)

        selected = _top_nodes(score, args.topm)
        return _record_selection(args, score, selected)

    raise ValueError("Unknown replay_strategy: {}".format(strategy))
