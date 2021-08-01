import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_norm
from torch.autograd import Variable


class RegulatedLoss(nn.Module):
    def __init__(self, rmagnitude):
        super().__init__()
        self.rmagnitude = rmagnitude

    def forward(self, activation, kernel, target):
        conv = torch.zeros(target.shape)
        for batch_idx, batch_kernel in enumerate(kernel):
            conv[batch_idx] = F.conv2d(activation, batch_kernel, padding='same')
        loss = F.mse_loss(conv, target) / 2 + self.rmagnitude * matrix_norm(activation)
        loss = Variable(loss, requires_grad=True)

        if torch.cuda.is_available():
            return loss.cuda()
        return loss


class IndividualLoss(nn.Module):
    def __init__(self, weights=(0.5, 0.5)):
        super().__init__()
        self.weights = weights

    def forward(self, activation, kernel, activation_target, kernel_target):
        activation_pixels = activation.shape[-1] ** 2
        kernel_pixels = kernel.shape[-1] ** 2

        activation_loss = F.mse_loss(activation, activation_target) / activation_pixels
        kernel_loss = F.mse_loss(kernel, kernel_target) / kernel_pixels

        loss = self.weights[0] * activation_loss + self.weights[1] * kernel_loss

        if torch.cuda.is_available():
            return loss.cuda()
        return loss