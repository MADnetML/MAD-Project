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
    def __init__(self, number_of_samples, measurement_size, kernel_size, SNR=2, defect_density=0.01):
        """

        :param number_of_samples:
        :param measurement_size: 3 numbers (E, n1, n2)
        :param kernel_size: (m1, m2)
        :param SNR: A positive number
        :param defect_density: A fractional number
        """
        E, n1, n2 = measurement_size
        m1, m2 = kernel_size
        self.number_of_samples = number_of_samples
        self.measurement = torch.zeros([number_of_samples, E, n1, n2])
        self.kernel = np.zeros((number_of_samples, E, m1, m2))
        self.activation_map = np.zeros((number_of_samples, n1, n2))

        for i in range(number_of_samples):
            E_shift = np.random.randint(-4, 1)
            temp_measurement, temp_kernel, self.activation_map[i] = Y_factory(E + E_shift, (n1, n2), kernel_size, defect_density, SNR=2)

            self.kernel[i] = torch.tensor(fix_levels(temp_kernel, E))
            self.measurement[i] = torch.tensor(fix_levels(temp_measurement, E))
            self.activation_map[i] = torch.tensor(self.activation_map[i])

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        if torch.cuda.is_available():
            return self.measurement[idx].cuda(), self.kernel[idx].cuda(), self.activation_map[idx].cuda()
        return self.measurement[idx], self.kernel[idx], self.activation_map[idx]
