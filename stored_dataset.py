import os

import numpy as np
import torch
from torch.utils.data import Dataset
from simulation import Y_factory
from copy import deepcopy


def fix_levels(matrix, layers):
    """
    Outputs a matrix with given number of layers.
    :param matrix: dim 3 matrix
    :param layers: The desired dim1 size
    :return: dim 3 matrix
    """
    diff = layers - matrix.shape[0]
    out_matrix = deepcopy(matrix)
    idx = np.random.randint(0, np.shape(out_matrix)[0], diff)
    out_matrix = np.concatenate([out_matrix, out_matrix[idx]])
    return out_matrix


def fix_trans_size(matrix, size):
    raise NotImplementedError


class QPIDataSet(Dataset):
    def __init__(self, path2dataset):

        self.files_in_folder = os.listdir(path2dataset)
        self.length = len(self.files_in_folder) // 3
        self.files_in_folder.sort()

        self.activation_map = self.files_in_folder[:self.length]
        self.kernel = self.files_in_folder[self.length:2 * self.length]
        self.measurement = self.files_in_folder[2 * self.length:3 * self.length]


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        measurement = torch.load(self.measurement[idx])
        kernel = torch.load(self.kernel[idx])
        activation = torch.load(self.activation_map[idx])

        if torch.cuda.is_available():
            return measurement.cuda(), kernel.cuda(), activation.cuda()
        return measurement, kernel, activation
