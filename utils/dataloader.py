import numpy as np
import torch

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, shuffle=True, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x_i = torch.Tensor(x_i).to('cuda:0', non_blocking=True)
                y_i = torch.Tensor(y_i).to('cuda:0', non_blocking=True)
                yield {'x': x_i, 'y': y_i}
                self.current_ind += 1

        return _wrapper()

class Cotinual_learning_DataLoader(object):
    def __init__(self, xs, batch_size, shuffle=True, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs = self.xs[permutation]
        self.xs = xs

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                x_i = torch.Tensor(x_i).to('cuda:0', non_blocking=True)
                yield {'x': x_i}
                self.current_ind += 1

        return _wrapper()

class AllHistoryDataLoader(object):
    """Stream yearly NPZ training splits and pad unavailable old nodes per batch."""
    def __init__(self, paths, node_count, batch_size, shuffle=True, device='cuda:0'):
        self.paths = list(paths)
        self.node_count = int(node_count)
        self.batch_size = int(batch_size)
        self.shuffle_data = bool(shuffle)
        self.device = device
        self.num_batch = 0
        for path in self.paths:
            with np.load(path, allow_pickle=True) as data:
                self.num_batch += int(np.ceil(len(data['train_x']) / self.batch_size))

    def __len__(self):
        return self.num_batch

    def _pad_nodes(self, array):
        if array.shape[2] == self.node_count:
            return array
        if array.shape[2] > self.node_count:
            return array[:, :, :self.node_count, :]
        shape = list(array.shape)
        shape[2] = self.node_count - array.shape[2]
        return np.concatenate([array, np.zeros(shape, dtype=array.dtype)], axis=2)

    def get_iterator(self):
        def _wrapper():
            paths = list(self.paths)
            if self.shuffle_data:
                np.random.shuffle(paths)
            for path in paths:
                with np.load(path, allow_pickle=True) as data:
                    xs, ys = data['train_x'], data['train_y']
                    indices = np.arange(len(xs))
                    if self.shuffle_data:
                        np.random.shuffle(indices)
                    for start in range(0, len(indices), self.batch_size):
                        batch_indices = indices[start:start + self.batch_size]
                        if len(batch_indices) < self.batch_size:
                            batch_indices = np.pad(batch_indices, (0, self.batch_size-len(batch_indices)),
                                                   mode='edge')
                        x = self._pad_nodes(xs[batch_indices])
                        y = self._pad_nodes(ys[batch_indices])
                        yield {'x': torch.as_tensor(x, dtype=torch.float32, device=self.device),
                               'y': torch.as_tensor(y, dtype=torch.float32, device=self.device)}
        return _wrapper()
