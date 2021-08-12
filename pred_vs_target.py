import argparse
import numpy as np
from tqdm import tqdm
from scipy.signal import convolve
import matplotlib.pyplot as plt
import torch
from model2 import MADNet, MADNet2, MADNet3
import torch.nn as nn
from stored_dataset import QPIDataSet
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


def conv_per_layer(activation, kernel):
    """
    Convolves, per-layer
    """
    kernel = np.array(torch.squeeze(kernel, dim=0).detach().cpu())
    activation = np.array(activation.squeeze().detach().cpu())
    levels, m1, m2 = np.shape(kernel)
    n1, n2 = np.shape(activation)
    out = np.zeros([levels, n1, n2])
    for level in range(levels):
        out[level] = convolve(activation, kernel[level], mode='same')

    return out


def regulator(X):
    """
    The pseudo-Huber regulator
    :param X: 2D matrix
    :return: A real number
    """
    X_array = np.array(X.cpu())
    mu = 10 ** -6  # A small positive number (chosen in the paper to be 10 ** -6)
    return np.sum(mu ** 2 * (np.sqrt(1 + (mu ** -2) * np.abs(X_array)) - 1))


def cost_fun(lambda_in, ker, act, meas):
    """
    The cost function = 0.5|A conv X - Y|**2 + lambda * r(X)
    """
    meas = np.array(torch.squeeze(meas, dim=0).cpu())
    s, n1, n2 = np.shape(meas)
    meas_pred = conv_per_layer(act, ker)
    phi = 0.5 * np.sum((meas_pred - meas) ** 2) + lambda_in * regulator(act)
    return phi


def plot_pics(ker_pred, act_pred, meas_pred, ker, act, meas, folder_name, losses='', title='Prediction', i=0):
    act = np.array(act.squeeze().detach().cpu())
    act_pred = np.array(act_pred.squeeze().detach().cpu())
    ker = np.array(ker.squeeze().detach().cpu())
    ker_pred = np.array(ker_pred.squeeze().detach().cpu())
    meas = np.array(meas.squeeze().detach().cpu())

    meas_max, meas_min = meas.max(), meas.min()
    ker_max, ker_min = ker.max(), ker.min()
    act_max, act_min = act.max(), act.min()

    pred_meas_max, pred_meas_min = pred_meas.max(), pred_meas.min()
    pred_ker_max, pred_ker_min = ker_pred.max(), ker_pred.min()
    pred_act_max, pred_act_min = act_pred.max(), act_pred.min()

    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle(title + '\n' + losses)
    plot_data = [[ker_pred, act_pred, meas_pred[0]],
                 [ker, act, meas]]
    maxes = [[pred_ker_max, pred_act_max, pred_meas_max],
             [ker_max, act_max, meas_max]]
    mins = [[pred_ker_min, pred_act_min, pred_meas_min],
            [ker_min, act_min, meas_min]]
    titles = [['Kernel Pred', 'Activation Pred', 'Measurement Pred'],
              ['Kernel Target', 'Activation Target', 'Measurement Target']]

    for row in range(2):
        for col in range(3):
            axs[row, col].set_title(titles[row][col])
            divider = make_axes_locatable(axs[row, col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = axs[row, col].imshow(plot_data[row][col], vmin=mins[row][col], vmax=maxes[row][col])
            axs[row, col].axis('off')
            fig.colorbar(im, ax=axs[row, col], cax=cax,
                         ticks=[mins[row][col], (mins[row][col] + maxes[row][col]) / 2, 0, maxes[row][col]])
    plt.tight_layout()

    plt.savefig(folder_name + '/' + title + '.png', dpi=400)


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


parser = argparse.ArgumentParser(description="Visualizing target and prediction")
parser.add_argument("-m", "--model",
                    dest='model',
                    type=int,
                    choices=[1, 2, 3],
                    help="Specify which model (MADNet1 of MADNet2) will be used",
                    required=True)
parser.add_argument("-fn", "--folder-name",
                    dest='folder_name',
                    type=str,
                    help="Name of folder for output files to be saved in",
                    required=True)
parser.add_argument("-pn", "--pic-num",
                    dest='pic_num',
                    type=int,
                    help="Number of pictures to save",
                    required=True)
# parser.add_argument("-nos", "--number-of-samples",
#                     dest='number_of_samples',
#                     type=int,
#                     help="Number of samples",
#                     required=True)

args = parser.parse_args()

number_of_samples = len(os.listdir(os.getcwd() + '/testing_dataset')) // 3

measurement_size = (1, 200, 200)
E, n1, n2 = measurement_size
kernel_size = (20, 20)
defect_density = np.random.uniform(low=-4, high=-1, size=number_of_samples)
SNR = 2

if args.model == 1:
    net = MADNet(measurement_size)
elif args.model == 2:
    net = MADNet2(measurement_size)
elif args.model == 3:
    net = MADNet3(measurement_size)

if torch.cuda.is_available():
    net.load_state_dict(torch.load(args.folder_name + '/trained_model.pt'))
else:
    net.load_state_dict(torch.load(args.folder_name + '/trained_model.pt', torch.device('cpu')))
net.eval()

test_ds = QPIDataSet(os.getcwd() + '/testing_dataset')
testing_dataloader = DataLoader(test_ds)

net_loss = compute_regularized_loss(testing_dataloader, net, cost_fun)
basline_loss = compute_regularized_loss(testing_dataloader, net, cost_fun, baseline=True)

losses = f'Loss = {round(net_loss, 2)}, basline = {round(basline_loss, 2)}'

i_max = args.pic_num
for i, (measurement, kernel, activation_map) in enumerate(testing_dataloader):
    if i == i_max:
        break
    pred_active, _, pred_kernel = net(measurement)
    pred_meas = conv_per_layer(pred_active, pred_kernel)
    plot_pics(pred_kernel, pred_active, pred_meas,
              kernel, activation_map, measurement, args.folder_name,
              losses, f'Model{args.model} Prediction{i}')
