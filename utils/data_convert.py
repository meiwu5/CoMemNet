import numpy as np
import networkx as nx
import torch
import os
import pandas as pd
from torch_geometric.utils import to_dense_batch

num_feat=1
def z_score(data):
    std = np.std(data)
    return (data - np.mean(data)) / std if std != 0 else data

def MinMaxnormalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same

    _max = train.max(axis=(0, 1, 3), keepdims=True)
    _min = train.min(axis=(0, 1, 3), keepdims=True)

    print('_max.shape:', _max.shape)
    print('_min.shape:', _min.shape)

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_max': _max, '_min': _min}, train_norm, val_norm, test_norm

def generate_dataset(
        data, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    num_samples, num_nodes, _ = data.shape
    feature_list = [data[..., 0:num_feat]]
    if add_time_in_day:
        time_ind = [i%288 / 288 for i in range(num_samples)]
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        day_in_week = [(i // 288)%7 for i in range(num_samples)]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_samples(days, savepath, data, graph,train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False, x_len=12, y_len=12, add_time_in_day=True, add_day_in_week=True):
    edge_index = np.array(list(graph.edges)).T
    del graph
    data = np.expand_dims(data[0:days * 288, :], axis=-1)
    # print('data.shape',data.shape)
    x_len, y_len = x_len, y_len
    y_start=1
    x_offsets = np.sort(np.concatenate((np.arange(-(x_len - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(y_start, (y_len + 1), 1))
    x, y = generate_dataset(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_time_in_day
    )

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_train = round(num_samples * train_rate) - 1
    num_test = round(num_samples * test_rate)
    num_val = num_samples - num_test - num_train
    train_x, train_y = x[:num_train], y[:num_train][..., 0:1]
    val_x, val_y = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val][..., 0:1],
    )
    test_x, test_y = x[-num_test:], y[-num_test:][..., 0:1]

    # =========== Do it for minmaxnorm ============ #
    train_x_norm = train_x[:, :, :, :num_feat]
    train_x_time = train_x[:, :, :, num_feat:]
    val_x_norm   = val_x[:, :, :, :num_feat]
    val_x_time   = val_x[:, :, :, num_feat:]
    test_x_norm  = test_x[:, :, :, :num_feat]
    test_x_time   = test_x[:, :, :, num_feat:]

    train_x_norm = np.transpose(train_x_norm, axes=[0, 2, 3, 1])
    val_x_norm = np.transpose(val_x_norm, axes=[0, 2, 3, 1])
    test_x_norm = np.transpose(test_x_norm, axes=[0, 2, 3, 1])

    stat, train_x_norm, val_x_norm, test_x_norm = MinMaxnormalization(train_x_norm, val_x_norm, test_x_norm)

    train_x_norm = np.transpose(train_x_norm, axes=[0, 3, 1, 2])
    val_x_norm = np.transpose(val_x_norm, axes=[0, 3, 1, 2])
    test_x_norm = np.transpose(test_x_norm, axes=[0, 3, 1, 2])
    _max = stat['_max']
    _min = stat['_min']

    train_x = np.concatenate([train_x_norm, train_x_time], axis=-1)
    val_x = np.concatenate([val_x_norm, val_x_time], axis=-1)
    test_x = np.concatenate([test_x_norm, test_x_time], axis=-1)

    np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
    data = {"train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y, "test_x": test_x, "test_y": test_y, "edge_index": edge_index}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, list):
            print(f"{key}: {[v.shape for v in value]}")
        else:
            print(f"{key}: Not a tensor or ndarray, type={type(value)}")
    return data
