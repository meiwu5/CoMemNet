import numpy as np
import os

folder_path = '/root/autodl-fs/DCS-MR/data/PEMSD4-large/graph'
# folder_path = '/root/autodl-fs/DCS-MR/data/PEMSD8-mini/graph'

for year in range(2009, 2016):  # 包括2009到2015
# for year in range(2012, 2019):  # 包括2012到2019
    file_name = f"{year}_adj.npz"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        data = np.load(file_path)
        print(f"File: {file_name}")
        for key in data.files:
            matrix = data[key]
            shape = matrix.shape
            edge_num = np.count_nonzero(matrix) // 2  # 无向图除以2
            print(f"  Shape of '{key}': {shape}, Edges: {edge_num}")
    else:
        print(f"File not found: {file_name}")

