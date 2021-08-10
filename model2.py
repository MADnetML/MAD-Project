from unet_models import *
from cnn_model import *
import numpy as np
from copy import deepcopy
import torch.nn as nn


def fix_levels(matrix, layers):
    """
    Outputs a matrix with given number of layers.
    :param matrix: dim 3 matrix
    :param layers: The desired dim1 size
    :return: dim 3 matrix
    """
    diff = layers - matrix.shape[1]
    out_matrix = deepcopy(matrix)
    idx = np.random.randint(0, np.shape(out_matrix)[1], diff)
    out_matrix = torch.cat([out_matrix, out_matrix[:, idx]], dim=1)
    return out_matrix


def fix_sizes(matrix_in, out_size):
    """
    Turns matrix_in into the desired out_size
    :param matrix_in: 3D tensor (E', n1', n2')
    :param out_size:
    :return:
    """
    _, n, x, y = matrix_in.shape
    diff_x, diff_y = out_size[1] - x, out_size[2] - y
    pad_it = nn.ZeroPad2d((diff_x // 2, int(np.ceil(diff_x / 2)), diff_y // 2, int(np.ceil(diff_y / 2))))
    out = fix_levels(pad_it(matrix_in), out_size[0])
    return out


class MADNet(nn.Module):
    def __init__(self, input_shape, look_in=False):
        """

        :param input_shape: (E, n1, n2)
        :param look_in: If true, returns activation, kernel AND active_cnn, kernel_cnn
        """
        super(MADNet, self).__init__()
        assert input_shape[1] >= 180 and input_shape[
            2] >= 180, 'MADError: n1 or n2 are smaller than 180. Remember: we chose n1=n2=200 and m1=m2=20'
        self.look_in = look_in

        self.in_active = input_shape[0]
        self.in_kernel = input_shape[0]
        self.out_active = 16
        self.out_kernel = input_shape[0]
        self.hidden_active = [input_shape[0], input_shape[0], input_shape[0]]  # not sure about it
        self.hidden_kernel = [input_shape[0], input_shape[0]]  # can also be changed

        self.cnn_active = CnnInit(self.in_active, self.out_active, self.hidden_active)
        self.cnn_kernel = CnnInit(self.in_kernel, self.out_kernel, self.hidden_kernel)

        self.unet_active = UNet_active(self.out_active, 1)
        self.unet_kernel = UNet_kernel(self.out_kernel, self.out_kernel)
        self.measurement_size = input_shape  # (100, 200, 200). (100, 20, 20)
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x):
        x = fix_sizes(x, self.measurement_size)
        activated_cnn = self.cnn_active(x)
        kerneled_cnn = self.cnn_kernel(x)

        active_out = self.unet_active(activated_cnn)

        kernel_out = self.unet_kernel(kerneled_cnn)
        active_class = self.classifier(active_out)
        if self.look_in:
            return active_out, active_class, kernel_out, activated_cnn, kerneled_cnn
        else:
            return active_out, active_class, kernel_out


class MADNet2(nn.Module):
    def __init__(self, input_shape, look_in=False):
        super(MADNet2, self).__init__()
        self.look_in = look_in
        self.hidden_active = [input_shape[0], input_shape[0], input_shape[0]]
        self.input_shape = input_shape
        self.unet = UNet_active(input_shape[0], input_shape[0])  # Input channels = Output channels = E
        self.cnn_active = CnnInit(input_shape[0], 1, self.hidden_active)
        self.cnn_kernel = CnnShrink(input_shape[0])
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x):
        x = fix_sizes(x, self.input_shape)
        x = self.unet(x)
        active_out = self.cnn_active(x)
        kernel_out = self.cnn_kernel(x)
        active_class = self.classifier(active_out)
        if self.look_in:
            return active_out, active_class, kernel_out, x
        return active_out, active_class, kernel_out


class MADNet3(nn.Module):
    def __init__(self, input_shape, look_in=False):
        super(MADNet3, self).__init__()
        self.look_in = look_in
        self.hidden_active = [input_shape[0], input_shape[0], input_shape[0]]
        self.input_shape = input_shape
        self.unet = UNet_active(input_shape[0], input_shape[0])  # Input channels = Output channels = E
        self.cnn_active = CnnInit(input_shape[0], 1, self.hidden_active)
        self.cnn_kernel = CnnShrink(input_shape[0])
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x):
        x = fix_sizes(x, self.input_shape)
        unet_ker = self.unet(x)
        active_out = self.cnn_active(x)
        kernel_out = self.cnn_kernel(unet_ker)
        active_class = self.classifier(active_out)
        if self.look_in:
            return active_out, active_class, kernel_out, x
        return active_out, active_class, kernel_out
