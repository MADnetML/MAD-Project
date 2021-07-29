from simulation import Y_factory
from streaming_dataset import fix_levels
import torch
import numpy as np
import os


def save_data(number_of_samples, measurement_size, kernel_size, SNR=2, training=False, validation=False):

    files_in_folder = os.listdir()
    if training and 'training_dataset' not in files_in_folder:
        os.system("mkdir training_dataset")
    if validation and 'validation_dataset' not in files_in_folder:
        os.system("mkdir validation_dataset")


    defect_density = np.random.uniform(low=-4, high=-1, size=(number_of_samples,))

    E, n1, n2 = measurement_size
    for i in range(number_of_samples):
        E_shift = np.random.randint(-4, 1)
        temp_measurement, temp_kernel, temp_activation_map = Y_factory(E + E_shift, (n1, n2),
                                                                       kernel_size,
                                                                       10 ** defect_density[i],
                                                                       SNR)
        if training:
            torch.save(torch.tensor(fix_levels(temp_kernel, E)), os.getcwd() + '/training_dataset/kernel_%d.pt' % i)
            torch.save(torch.tensor(fix_levels(temp_measurement, E)), os.getcwd() + '/training_dataset/measurement_%d.pt' % i)
            torch.save(torch.tensor(temp_activation_map), os.getcwd() + '/training_dataset/activation_%d.pt' % i)
        elif validation:
            torch.save(torch.tensor(fix_levels(temp_kernel, E)), os.getcwd() + '/validation_dataset/kernel_%d.pt' % i)
            torch.save(torch.tensor(fix_levels(temp_measurement, E)), os.getcwd() + '/validation_dataset/measurement_%d.pt' % i)
            torch.save(torch.tensor(temp_activation_map), os.getcwd() + '/validation_dataset/activation_%d.pt' % i)
    if not training and not validation:
        print("Specify validation or training to save files.")




number_of_samples = 5
measurement_size = (100, 200, 200)  # = (E, n1, n2)
kernel_size = (20, 20)  # = (m1, m2)
save_data(number_of_samples, measurement_size, kernel_size, training=True)
save_data(number_of_samples, measurement_size, kernel_size, validation=True)