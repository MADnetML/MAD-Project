import os
import argparse
import sys

import numpy as np
from tqdm import tqdm
from scipy.signal import convolve
import matplotlib.pyplot as plt
import torch
from model2 import MADNet, MADNet2, MADNet3  # model -> model2
from stored_dataset import QPIDataSet
from custom_losses import SumIndividualLoss, RegulatedLoss
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
            predic_active, predic_active_classes, predic_kernel = network(measurement)

            # This line does convolution per energy level
            predic_measurement = conv_per_layer(predic_active, predic_kernel)

            # loss += loss_function(predic_active, predic_active_classes, predic_kernel, activation_map, kernel)
            loss += loss_function(predic_active, predic_kernel, measurement)

    return loss / n_batches


def compute_class_loss(dataloader, network):
    loss = 0
    loss_function = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        network.cuda()
    network.eval()
    n_batches = 0
    with torch.no_grad():
        for measurement, _, activation_map in dataloader:
            n_batches += 1

            if torch.cuda.is_available():
                measurement = measurement.cuda()
                activation_map = activation_map.to(torch.device('cuda'))
            _, predic_active_classes, _ = network(measurement)
            activation_class_target = 1*(activation_map > 0)
            loss += loss_function(predic_active_classes, activation_class_target)

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

            predic_active, _, predic_kernel = net(measurement)
            activation_loss += mse_loss(activation_map, predic_active)
            if predic_kernel.shape[-1] > kernel.shape[-1]:
                diff = predic_kernel.shape[-1] - kernel.shape[-1]
                cropped = slice(diff // 2, diff // 2 + kernel.shape[-1])
                kernel_loss = mse_loss(predic_kernel[:, :, cropped, cropped], kernel)
            else:
                kernel_loss += mse_loss(kernel, predic_kernel)

    return activation_loss / n_batches, kernel_loss / n_batches


def compute_regularized_loss(dataloader, network, loss_function, baseline=False):
    loss = 0
    lam = 0.005
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
            if baseline:
                loss += loss_function(lam, kernel, activation_map, measurement)
            else:
                predic_active, _, predic_kernel = network(measurement)
                # Maybe need to turn Torch tensor to np array
                loss += loss_function(lam, predic_kernel, predic_active, measurement)
    return loss / n_batches


def cost_fun(lambda_in, ker, act, meas):
    """
    The cost function = 0.5|A conv X - Y|**2 + lambda * r(X)
    """
    meas = np.array(meas.squeeze().cpu())
    meas_pred = np.array(conv_per_layer(act, ker).squeeze().cpu())
    phi = 0.5 * np.sum((meas_pred - meas) ** 2) + lambda_in * regulator(act)
    return phi


def regulator(X):
    """
    The pseudo-Huber regulator
    :param X: 2D matrix
    :return: A real number
    """
    X_array = np.array(X.cpu())
    mu = 10 ** -6  # A small positive number (chosen in the paper to be 10 ** -6)
    return np.sum(mu ** 2 * (np.sqrt(1 + (mu ** -2) * np.abs(X_array)) - 1))


def plot_loss(training, validation, fig_name, folder_name):
    fig, ax = plt.subplots()
    fig.suptitle(f'Madnet{model}')

    ax.set_title('Total Loss')
    ax.plot(training, label='Training')
    ax.plot(validation, label='Validation')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.set_title(fig_name)
    plt.tight_layout()
    plt.savefig(folder_name + '/' + fig_name + '.png', dpi=400)


# Adding CLA
parser = argparse.ArgumentParser(description=
                                 "Deep learning model for deconvolving Quasiparticle interference")
parser.add_argument("-m", "--model",
                    dest='model',
                    type=int,
                    choices=[1, 2, 3],
                    help="Specify which model (MADNet1 or MADNet2 or MADNet3) will be used",
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
parser.add_argument("-bs", "--batch-size",
                    dest='batch_size',
                    type=int,
                    help="Batch size",
                    required=False)
parser.add_argument("-fn", "--folder-name",
                    dest='folder_name',
                    type=str,
                    help="Name of folder for output files to be saved in",
                    required=True)

args = parser.parse_args()
if not args.batch_size:
    args.batch_size = 20
if sys.platform == 'linux':
    os.system('echo ==========================================================')
    os.system(
        'echo Using model {} with {} epochs and batch size of {}'.format(args.model, args.epochs, args.batch_size))
    if args.lr:
        os.system('echo Learning rate = {}.\n'.format(args.lr))
    else:
        os.system('echo Using default value of lr = 1e-4')
else:
    print('Using model {} with {} epochs and batch size of {}'.format(args.model, args.epochs, args.batch_size))
    if args.lr:
        print('Learning rate = {}.\n'.format(args.lr))
    else:
        print('Using default value of lr = 1e-4')

model = args.model

train_ds = QPIDataSet(os.getcwd() + '/training_dataset')
valid_ds = QPIDataSet(os.getcwd() + '/validation_dataset')
training_dataloader = DataLoader(train_ds, batch_size=args.batch_size)
valid_dataloader = DataLoader(valid_ds, batch_size=args.batch_size)

measurement_size = (1, 200, 200)

if model == 1:
    net = MADNet(measurement_size)
elif model == 2:
    net = MADNet2(measurement_size)
else:
    net = MADNet3(measurement_size)

if os.path.isdir(args.folder_name):
    try:
        net.load_state_dict(torch.load(args.folder_name + '/trained_model.pt'))
        if sys.platform == 'linux':
            os.system('echo Parameters were loaded successfully!')
        else:
            print('Parameters were loaded successfully!')
        total_training_loss_vs_epoch = np.load(args.folder_name + '/total_training_loss_vs_epoch.npy').tolist()
        total_val_loss_vs_epoch = np.load(args.folder_name + '/total_val_loss_vs_epoch.npy').tolist()
        training_regulated_loss_vs_epoch = np.load(args.folder_name + '/training_regulated_loss_vs_epoch.npy').tolist()
        validation_regulated_loss_vs_epoch = np.load(args.folder_name + '/validation_regulated_loss_vs_epoch.npy').tolist()
        activation_mse_training_loss_vs_epoch = np.load(args.folder_name + '/activation_mse_training_loss_vs_epoch.npy').tolist()
        activation_mse_val_loss_loss_vs_epoch = np.load(args.folder_name + '/activation_mse_val_loss_loss_vs_epoch.npy').tolist()
        kernel_mse_training_loss_vs_epoch = np.load(args.folder_name + '/kernel_mse_training_loss_vs_epoch.npy').tolist()
        kernel_mse_val_loss_vs_epoch = np.load(args.folder_name + '/kernel_mse_val_loss_vs_epoch.npy').tolist()
        training_class_loss_vs_epoch = np.load(args.folder_name + '/training_class_loss_vs_epoch.npy').tolist()
        validation_class_loss_vs_epoch = np.load(args.folder_name + '/validation_class_loss_vs_epoch.npy').tolist()
    except FileNotFoundError:
        total_training_loss_vs_epoch = []
        total_val_loss_vs_epoch = []
        training_regulated_loss_vs_epoch = []
        validation_regulated_loss_vs_epoch = []
        activation_mse_training_loss_vs_epoch = []
        activation_mse_val_loss_loss_vs_epoch = []
        kernel_mse_training_loss_vs_epoch = []
        kernel_mse_val_loss_vs_epoch = []
        training_class_loss_vs_epoch = []
        validation_class_loss_vs_epoch = []
else:  # The folder doesn't exist, so create it and initialize lists
    os.system("mkdir {}".format(args.folder_name))
    total_training_loss_vs_epoch = []
    total_val_loss_vs_epoch = []
    training_regulated_loss_vs_epoch = []
    validation_regulated_loss_vs_epoch = []
    activation_mse_training_loss_vs_epoch = []
    activation_mse_val_loss_loss_vs_epoch = []
    kernel_mse_training_loss_vs_epoch = []
    kernel_mse_val_loss_vs_epoch = []
    training_class_loss_vs_epoch = []
    validation_class_loss_vs_epoch = []

# individual_loss = SumIndividualLoss
regulated_loss = RegulatedLoss(0.01)  # Added

lr = 1e-4
if args.lr:
    lr = float(args.lr)
optimizer = Adam(net.parameters(), lr=lr)

if torch.cuda.is_available():
    net.cuda()
    if sys.platform == 'linux':
        os.system('echo Running on GPU.')
    else:
        print('Running on GPU.')

n_epochs = args.epochs

pbar = tqdm(range(n_epochs))

for epoch in pbar:

    if len(total_val_loss_vs_epoch) > 1:
        pbar.set_description('Val loss: %.3f, '
                             'Best: %.3f; '
                             'Training loss: %.3f '
                             'Best: %.3f ' % (total_val_loss_vs_epoch[-1],
                                              min(total_val_loss_vs_epoch),
                                              total_training_loss_vs_epoch[-1],
                                              min(total_training_loss_vs_epoch)
                                              ))

    net.train()  # put the net into "training mode"
    for target_measurement, target_kernel, target_activation in training_dataloader:
        if torch.cuda.is_available():
            target_measurement = target_measurement.cuda()

        optimizer.zero_grad()
        pred_active, pred_active_class, pred_kernel = net(target_measurement)
        pred_measurement = conv_per_layer(pred_active, pred_kernel, requires_grad=True)
        # loss = individual_loss(pred_active.squeeze(), pred_active_class, pred_kernel, target_activation, target_kernel)
        loss = regulated_loss(pred_active, pred_kernel, target_measurement)  # Added
        loss.backward()
        optimizer.step()

    net.eval()  # put the net into evaluation mode

    total_training_loss = compute_loss(training_dataloader, net, regulated_loss)  # individual_loss -> regulated_loss
    total_validation_loss = compute_loss(valid_dataloader, net, regulated_loss)  # individual_loss -> regulated_loss
    training_regulated_loss = compute_regularized_loss(training_dataloader, net, cost_fun)
    validation_regulated_loss = compute_regularized_loss(valid_dataloader, net, cost_fun)
    activation_training_mse_loss, kernel_training_mse_loss = compute_mse_loss(training_dataloader, net)
    activation_val_mse_loss, kernel_val_mse_loss = compute_mse_loss(valid_dataloader, net)
    activation_training_classLoss = compute_class_loss(training_dataloader, net)
    activation_validation_classLoss = compute_class_loss(training_dataloader, net)

    total_training_loss_vs_epoch.append(total_training_loss.data.cpu().numpy())
    total_val_loss_vs_epoch.append(total_validation_loss.data.cpu().numpy())
    training_regulated_loss_vs_epoch.append(training_regulated_loss)
    validation_regulated_loss_vs_epoch.append(validation_regulated_loss)
    activation_mse_training_loss_vs_epoch.append(activation_training_mse_loss.data.cpu().numpy())
    activation_mse_val_loss_loss_vs_epoch.append(activation_val_mse_loss.data.cpu().numpy())
    kernel_mse_training_loss_vs_epoch.append(kernel_training_mse_loss.data.cpu().numpy())
    kernel_mse_val_loss_vs_epoch.append(kernel_val_mse_loss.data.cpu().numpy())
    training_class_loss_vs_epoch.append(activation_training_classLoss.data.cpu().numpy())
    validation_class_loss_vs_epoch.append(activation_validation_classLoss.data.cpu().numpy())

    if min(total_val_loss_vs_epoch) == total_val_loss_vs_epoch[-1]:
        torch.save(net.state_dict(), args.folder_name + '/trained_model.pt')

    np.save(args.folder_name + '/total_training_loss_vs_epoch.npy', total_training_loss_vs_epoch)
    np.save(args.folder_name + '/total_val_loss_vs_epoch.npy', total_val_loss_vs_epoch)
    np.save(args.folder_name + '/training_regulated_loss_vs_epoch.npy', training_regulated_loss_vs_epoch)
    np.save(args.folder_name + '/validation_regulated_loss_vs_epoch.npy', validation_regulated_loss_vs_epoch)
    np.save(args.folder_name + '/activation_mse_training_loss_vs_epoch.npy', activation_mse_training_loss_vs_epoch)
    np.save(args.folder_name + '/activation_mse_val_loss_loss_vs_epoch.npy', activation_mse_val_loss_loss_vs_epoch)
    np.save(args.folder_name + '/kernel_mse_training_loss_vs_epoch.npy', kernel_mse_training_loss_vs_epoch)
    np.save(args.folder_name + '/kernel_mse_val_loss_vs_epoch.npy', kernel_mse_val_loss_vs_epoch)
    np.save(args.folder_name + '/training_class_loss_vs_epoch.npy', training_class_loss_vs_epoch)
    np.save(args.folder_name + '/validation_class_loss_vs_epoch.npy', validation_class_loss_vs_epoch)


# Plotting results
plot_loss(total_training_loss_vs_epoch, total_val_loss_vs_epoch, 'total loss', args.folder_name)
plot_loss(training_regulated_loss_vs_epoch, validation_regulated_loss_vs_epoch, 'regulated loss', args.folder_name)
plot_loss(activation_mse_training_loss_vs_epoch, activation_mse_val_loss_loss_vs_epoch, 'activation mse loss', args.folder_name)
plot_loss(kernel_mse_training_loss_vs_epoch, kernel_mse_val_loss_vs_epoch, 'kernel mse loss', args.folder_name)
plot_loss(training_class_loss_vs_epoch, validation_class_loss_vs_epoch, 'classification loss', args.folder_name)

