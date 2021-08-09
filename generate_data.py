from simulation import Y_factory
from streaming_dataset import fix_levels
import torch
import numpy as np
import os


def save_data(number_of_samples, measurement_size, kernel_size, SNR=2, training=False, validation=False, testing=False):
    files_in_folder = os.listdir()
    if training and 'training_dataset' not in files_in_folder:
        os.system("mkdir training_dataset")
    if validation and 'validation_dataset' not in files_in_folder:
        os.system("mkdir validation_dataset")
    if testing and 'testing_dataset' not in files_in_folder:
        os.system("mkdir testing_dataset")

    defect_density = np.random.uniform(low=-4, high=-1, size=(number_of_samples,))

    E, n1, n2 = measurement_size
    for i in range(number_of_samples):
        temp_measurement, temp_kernel, temp_activation_map = Y_factory(E, (n1, n2),
                                                                       kernel_size,
                                                                       10 ** defect_density[i],
                                                                       SNR)
        if training:
            np.save(os.getcwd() + '/training_dataset/kernel_%d' % i, fix_levels(temp_kernel, E))
            np.save(os.getcwd() + '/training_dataset/measurement_%d' % i, fix_levels(temp_measurement, E))
            np.save(os.getcwd() + '/training_dataset/activation_%d' % i, temp_activation_map)
        elif validation:
            np.save(os.getcwd() + '/validation_dataset/kernel_%d' % i, fix_levels(temp_kernel, E))
            np.save(os.getcwd() + '/validation_dataset/measurement_%d' % i, fix_levels(temp_measurement, E))
            np.save(os.getcwd() + '/validation_dataset/activation_%d' % i, temp_activation_map)
        elif testing:
            np.save(os.getcwd() + '/testing_dataset/kernel_%d' % i, fix_levels(temp_kernel, E))
            np.save(os.getcwd() + '/testing_dataset/measurement_%d' % i, fix_levels(temp_measurement, E))
            np.save(os.getcwd() + '/testing_dataset/activation_%d' % i, temp_activation_map)
    if not training and not validation and not testing:
        print("Specify validation, training ot testing to save files.")


measurement_size = (1, 200, 200)  # = (E, n1, n2)
kernel_size = (20, 20)  # = (m1, m2)
save_data(15, measurement_size, kernel_size, training=True)
save_data(5, measurement_size, kernel_size, validation=True)
# save_data(5, measurement_size, kernel_size, testing=True)
