import numpy as np
import torch
from model import MADNet
from dataset import QPIDataSet
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.signal import convolve
import matplotlib.pyplot as plt


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
    """
    Returns three losses, measurement, activation and kernel MSE loss
    :param dataloader:
    :param network:
    :param loss_function:
    :return:
    """
    measurement_loss, activation_loss, kernel_loss = 0, 0, 0

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

            measurement_loss += loss_function(predic_measurement, measurement).item()
            activation_loss += loss_function(predic_active[:, 0, :, :], activation_map).item()
            kernel_loss += loss_function(predic_kernel, kernel).item()

    return measurement_loss / n_batches, activation_loss / n_batches, kernel_loss / n_batches


number_of_samples = 2
measurement_size = (100, 200, 200)  # = (E, n1, n2)
kernel_size = (20, 20)  # = (m1, m2)
print('Starting to make the data.')
train_ds = QPIDataSet(number_of_samples, measurement_size, kernel_size)
valid_ds = QPIDataSet(1, measurement_size, kernel_size)
training_dataloader = DataLoader(train_ds, batch_size=100)
valid_dataloader = DataLoader(train_ds, batch_size=100)
print('Finished making the data.')

net = MADNet(measurement_size)
optimizer = Adam(net.parameters(), lr=1e-4)
loss_func = nn.MSELoss()

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
    for target_measurement, _, _ in training_dataloader:
        if torch.cuda.is_available():
            target_measurement = target_measurement.cuda()

        optimizer.zero_grad()
        pred_active, pred_kernel = net(target_measurement)
        pred_measurement = conv_per_layer(pred_active, pred_kernel, requires_grad=True)
        loss = loss_func(pred_measurement, target_measurement)  # Regularization term to be added?
        loss.backward()
        optimizer.step()

    net.eval()  # put the net into evaluation mode
    measurement_training_loss, activation_training_loss, kernel_training_loss = compute_loss(training_dataloader, net,
                                                                                             loss_func)
    measurement_val_loss, activation_val_loss, kernel_val_loss = compute_loss(valid_dataloader, net, loss_func)

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
