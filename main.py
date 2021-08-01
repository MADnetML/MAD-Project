import os

import numpy as np
from tqdm import tqdm
from scipy.signal import convolve
import matplotlib.pyplot as plt

import torch
from model import MADNet
from stored_dataset import QPIDataSet
from custom_losses import IndividualLoss
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader


# def regulated_loss(activation, kernel, target, rmagnitude):
#     mse_loss = nn.MSELoss()
#     conv = F.conv2d(activation, kernel, padding='same')
#     return mse_loss(conv, target) / 2 + rmagnitude * matrix_norm(activation)



def conv_per_layer(activation, kernel, requires_grad=False):
    """
    Convolves per-batch, per-layer
    :param requires_grad:
    :param activation:
    :param kernel:
    :return: torch tensor
    """
    activation = activation.cpu().detach().numpy()
    kernel = kernel.cpu().detach().numpy()
    batches, levels, m1, m2 = np.shape(kernel)
    _, _, n1, n2 = np.shape(activation)
    out = np.zeros([batches, levels, n1, n2])
    for batch in range(batches):
        for level in range(levels):
            out[batch, level] = convolve(activation[batch, 0], kernel[batch, level], mode='same')

    if torch.cuda.is_available() and requires_grad:
        return torch.tensor(out, requires_grad=True, dtype=torch.float).cuda()
    elif torch.cuda.is_available() and not requires_grad:
        return torch.tensor(out, requires_grad=False, dtype=torch.float).cuda()
        
    if requires_grad:
        return torch.tensor(out, requires_grad=True, dtype=torch.float)
    return torch.tensor(out, dtype=torch.float)


def compute_loss(dataloader, network, loss_function):
    loss = 0

    if torch.cuda.is_available():
        network.cuda()
    network.eval()

    n_batches = 0
    with torch.no_grad():
        for measurement, kernel, activation_map in dataloader:
            n_batches += 1

            if torch.cuda.is_available():
                measurement = measurement.cuda()
                kernel = kernel.cuda()
                activation_map = activation_map.to(torch.device('cuda'))
            predic_active, predic_kernel = network(measurement)

            # This line does convolution per energy level
            predic_measurement = conv_per_layer(predic_active, predic_kernel)

            loss += loss_function(predic_active, predic_kernel, activation_map, kernel)

    return loss / n_batches


train_ds = QPIDataSet(os.getcwd() + '/training_dataset')
valid_ds = QPIDataSet(os.getcwd() + '/validation_dataset')
training_dataloader = DataLoader(train_ds)
valid_dataloader = DataLoader(train_ds)

measurement_size = (100, 200, 200)
net = MADNet(measurement_size)
regulated_loss = IndividualLoss()
optimizer = Adam(net.parameters(), lr=1e-4)
train_val_loss_func = nn.MSELoss()

if torch.cuda.is_available():
    net.cuda()
    print('Using GPU.')

n_epochs = 2

measurement_training_loss_vs_epoch = []
activation_training_loss_loss_vs_epoch = []
kernel_training_loss_vs_epoch = []

measurement_val_loss_vs_epoch = []
activation_val_loss_loss_vs_epoch = []
kernel_val_loss_vs_epoch = []


pbar = tqdm(range(n_epochs))

for epoch in pbar:

    if len(measurement_val_loss_vs_epoch) > 1:
        pbar.set_description(f'Val loss: {round(100 * measurement_val_loss_vs_epoch[-1])},'
                             f' Best: {round(100 * min(measurement_val_loss_vs_epoch))}; '
                             f'Training loss:{round(100 * measurement_training_loss_vs_epoch[-1])},'
                             f' Best: {round(100 * min(measurement_training_loss_vs_epoch))}')

    net.train()  # put the net into "training mode"
    for target_measurement, target_kernel, target_activation in training_dataloader:
        if torch.cuda.is_available():
            target_measurement = target_measurement.cuda()

        optimizer.zero_grad()
        pred_active, pred_kernel = net(target_measurement)
        pred_measurement = conv_per_layer(pred_active, pred_kernel, requires_grad=True)
        # loss = loss_func(pred_measurement, target_measurement)  # Regularization term to be added?
        loss = regulated_loss(pred_active, pred_kernel, target_activation, target_kernel)
        loss.backward()
        optimizer.step()

    net.eval()  # put the net into evaluation mode
    measurement_training_loss, activation_training_loss, kernel_training_loss = compute_loss(training_dataloader, net,
                                                                                             train_val_loss_func)
    measurement_val_loss, activation_val_loss, kernel_val_loss = compute_loss(valid_dataloader, net, train_val_loss_func)

    measurement_training_loss_vs_epoch.append(measurement_training_loss)
    activation_training_loss_loss_vs_epoch.append(activation_training_loss)
    kernel_training_loss_vs_epoch.append(kernel_training_loss)

    measurement_val_loss_vs_epoch.append(measurement_val_loss)
    activation_val_loss_loss_vs_epoch.append(activation_val_loss)
    kernel_val_loss_vs_epoch.append(kernel_val_loss)

    if min(measurement_val_loss_vs_epoch) == measurement_val_loss_vs_epoch[-1]:
        torch.save(net.state_dict(), 'trained_model.pt')

# Plotting results
plt.plot(measurement_val_loss_vs_epoch)
plt.ylabel('Validation Loss')
plt.xlabel('Epoch number')

plt.show()
