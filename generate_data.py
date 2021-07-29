from simulation import Y_factory
from dataset import fix_levels
import torch
import numpy as np
import os


def save_data(number_of_samples, measurement_size, kernel_size, SNR=2, training=False, validation=False):
    defect_density = np.random.uniform(low=-4, high=-1, size=(number_of_samples,))

    E, n1, n2 = measurement_size
    for i in range(number_of_samples):
        E_shift = np.random.randint(-4, 1)
        temp_measurement, temp_kernel, temp_activation_map = Y_factory(E + E_shift, (n1, n2),
                                                                       kernel_size,
                                                                       10 ** defect_density[i],
                                                                       SNR)
        if training:
            torch.save(torch.tensor(fix_levels(temp_kernel, E)), os.getcwd() + '/training/kernel_%d.pt' % i)
            torch.save(torch.tensor(fix_levels(temp_measurement, E)), os.getcwd() + '/training/measurement_%d.pt' % i)
            torch.save(torch.tensor(temp_activation_map), os.getcwd() + '/training/activation_%d.pt' % i)
        elif validation:
            torch.save(torch.tensor(fix_levels(temp_kernel, E)), os.getcwd() + '/validation/kernel_%d.pt' % i)
            torch.save(torch.tensor(fix_levels(temp_measurement, E)), os.getcwd() + '/validation/measurement_%d.pt' % i)
            torch.save(torch.tensor(temp_activation_map), os.getcwd() + '/validation/activation_%d.pt' % i)
        else:
            print("Specify validation or training to save files.")




number_of_samples = 2
measurement_size = (100, 200, 200)  # = (E, n1, n2)
kernel_size = (20, 20)  # = (m1, m2)
save_data(number_of_samples, measurement_size, kernel_size)