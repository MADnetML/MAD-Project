import os
import argparse

import numpy as np
from tqdm import tqdm
from scipy.signal import convolve
import matplotlib.pyplot as plt

import torch
from model import MADNet, MADNet2
from stored_dataset import QPIDataSet
from custom_losses import IndividualLoss
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader



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


def compute_mse_loss(dataloader, net):
    mse_loss = nn.MSELoss()
    activation_loss = 0
    kernel_loss = 0

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    n_batches = 0
    with torch.no_grad():
        for measurement, kernel, activation_map in dataloader:
            n_batches += 1
            if torch.cuda.is_available():
                measurement = measurement.cuda()
                kernel = kernel.cuda()
                activation_map = activation_map.cuda()

            predic_active, predic_kernel = net(measurement)
            activation_loss += mse_loss(activation_map, predic_active)
            if predic_kernel.shape[-1] > kernel.shape[-1]:
                diff = predic_kernel.shape[-1] - kernel.shape[-1]
                cropped = slice(diff // 2, diff // 2 + kernel.shape[-1])
                kernel_loss = mse_loss(predic_kernel[:, :, cropped, cropped], kernel)
            else:
                kernel_loss += mse_loss(kernel, predic_kernel)

    return activation_loss / n_batches, kernel_loss / n_batches


def plot_losses(total_train, total_val, ker_train, ker_val, act_train, act_val, folder_name):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(f'Madnet{model}')

    axs[0].set_title('Total Loss')
    axs[0].plot(total_train, label='Training')
    axs[0].plot(total_val, label='Validation')

    axs[1].set_title('Kernel Loss')
    axs[1].plot(ker_train, label='Training')
    axs[1].plot(ker_val, label='Validation')

    axs[2].set_title('Activation Loss')
    axs[2].plot(act_train, label='Training')
    axs[2].plot(act_val, label='Validation')
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    for i in range(3):
        axs[i].set_xlim(left=0)
        axs[i].set_ylim(bottom=0)
        for axis in ['top', 'bottom', 'left', 'right']:
            axs[i].spines[axis].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(folder_name + '/Losses.png', dpi=400)

# Adding CLA
parser = argparse.ArgumentParser(description=
                                 "Deep learning model for deconvolving Quasiparticle interference")
parser.add_argument("-m", "--model",
                    dest='model',
                    type=int,
                    choices=[0, 1],
                    help="Specify which model (MADNet1 of MADNet2) will be used",
                    required=True)
parser.add_argument("-e", "--epochs",
                    dest='epochs',
                    type=int,
                    help="Number of epochs",
                    required=True)
parser.add_argument("-lr", "--learning-rate",
                    dest='lr',
                    type=float,
                    help="Learning rate",
                    required=False)
parser.add_argument("-fn", "--folder-name",
                    dest='folder_name',
                    type=str,
                    help="Name of folder for output files to be saved in",
                    required=True)

args = parser.parse_args()
print('Using model {} with {} epochs.\n'
      'Learning rate = {}.\n'.format(args.model, args.epochs, args.lr))


model = args.model

train_ds = QPIDataSet(os.getcwd() + '/training_dataset')
valid_ds = QPIDataSet(os.getcwd() + '/validation_dataset')
training_dataloader = DataLoader(train_ds)
valid_dataloader = DataLoader(valid_ds)

measurement_size = (20, 200, 200)

if model == 1:
    net = MADNet(measurement_size)
else:
    net = MADNet2(measurement_size)

if os.path.isdir(args.folder_name):
    net.load_state_dict(torch.load(args.folder_name + '/trained_model.pt'))
    total_training_loss_vs_epoch = np.load(args.folder_name + '/total_training_loss_vs_epoch.npy').tolist()
    activation_training_loss_loss_vs_epoch = np.load(args.folder_name + '/activation_training_loss_loss_vs_epoch.npy').tolist()
    kernel_training_loss_vs_epoch = np.load(args.folder_name + '/kernel_training_loss_vs_epoch.npy').tolist()
    total_val_loss_vs_epoch = np.load(args.folder_name + '/total_val_loss_vs_epoch.npy').tolist()
    activation_val_loss_loss_vs_epoch = np.load(args.folder_name + '/activation_val_loss_loss_vs_epoch.npy').tolist()
    kernel_val_loss_vs_epoch = np.load(args.folder_name + '/kernel_val_loss_vs_epoch.npy').tolist()
else:
    os.system("mkdir {}".format(args.folder_name))


individual_loss = IndividualLoss()

lr = 1e-4
if args.lr:
    lr = float(args.lr)
optimizer = Adam(net.parameters(), lr=lr)

if torch.cuda.is_available():
    net.cuda()
    print('Using GPU.')

n_epochs = args.epochs

total_training_loss_vs_epoch = []
activation_training_loss_loss_vs_epoch = []
kernel_training_loss_vs_epoch = []

total_val_loss_vs_epoch = []
activation_val_loss_loss_vs_epoch = []
kernel_val_loss_vs_epoch = []

pbar = tqdm(range(n_epochs))

for epoch in pbar:

    if len(total_val_loss_vs_epoch) > 1:
        pbar.set_description('Val loss: %.3f, '
                             'Best: %.3f; '
                             'Training loss: %.3f '
                             'Best: %.3f ' % (1e6 * total_val_loss_vs_epoch[-1],
                                              1e6 * min(total_val_loss_vs_epoch),
                                              1e6 * total_training_loss_vs_epoch[-1],
                                              1e6 * min(total_training_loss_vs_epoch)
                                              ))

    net.train()  # put the net into "training mode"
    for target_measurement, target_kernel, target_activation in training_dataloader:
        if torch.cuda.is_available():
            target_measurement = target_measurement.cuda()

        optimizer.zero_grad()
        pred_active, pred_kernel = net(target_measurement)
        pred_measurement = conv_per_layer(pred_active, pred_kernel, requires_grad=True)
        loss = individual_loss(pred_active.squeeze(), pred_kernel, target_activation, target_kernel)
        loss.backward()
        optimizer.step()

    net.eval()  # put the net into evaluation mode
    total_training_loss = compute_loss(training_dataloader, net, individual_loss)
    total_validation_loss = compute_loss(valid_dataloader, net, individual_loss)
    activation_training_loss, kernel_training_loss = compute_mse_loss(training_dataloader, net)
    activation_val_loss, kernel_val_loss = compute_mse_loss(valid_dataloader, net)

    total_training_loss_vs_epoch.append(total_training_loss)
    activation_training_loss_loss_vs_epoch.append(activation_training_loss)
    kernel_training_loss_vs_epoch.append(kernel_training_loss)

    total_val_loss_vs_epoch.append(total_validation_loss.data.cpu().numpy())
    activation_val_loss_loss_vs_epoch.append(activation_val_loss.data.cpu().numpy())
    kernel_val_loss_vs_epoch.append(kernel_val_loss.data.cpu().numpy())

    if min(total_val_loss_vs_epoch) == total_val_loss_vs_epoch[-1]:
        torch.save(net.state_dict(), args.folder_name + '/trained_model.pt')

    np.save(args.folder_name + '/total_training_loss_vs_epoch', total_training_loss_vs_epoch)
    np.save(args.folder_name + '/activation_training_loss_loss_vs_epoch', activation_training_loss_loss_vs_epoch)
    np.save(args.folder_name + '/kernel_training_loss_vs_epoch', kernel_training_loss_vs_epoch)
    np.save(args.folder_name + '/total_val_loss_vs_epoch', total_val_loss_vs_epoch)
    np.save(args.folder_name + '/activation_val_loss_loss_vs_epoch', activation_val_loss_loss_vs_epoch)
    np.save(args.folder_name + '/kernel_val_loss_vs_epoch', kernel_val_loss_vs_epoch)


# Plotting results
plot_losses(total_training_loss_vs_epoch, total_val_loss_vs_epoch, kernel_training_loss_vs_epoch,
            kernel_val_loss_vs_epoch, activation_training_loss_loss_vs_epoch, activation_val_loss_loss_vs_epoch,
            args.folder_name)
