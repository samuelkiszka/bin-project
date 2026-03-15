import os
import numpy as np






def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)


def load_data(data_path):
    with np.load(data_path) as data:
        x_train = data['x_train']
        y_train = data['y_train']
    return x_train, y_train