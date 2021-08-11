import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import matrix_norm
from torch.autograd import Variable
import numpy as np
from model import fix_sizes


class RegulatedLoss(nn.Module):
    def __init__(self, rmagnitude):
        super().__init__()
        self.rmagnitude = rmagnitude
        self.mu = 10 ** -6

    def regulator(self, activation):
        return torch.sum(self.mu ** 2 * (torch.sqrt(1 + (self.mu ** -2) * torch.abs(activation)) - 1))

    def forward(self, activation, kernel, target):
        conv = []
        for i in range(activation.shape[0]):
            batch_kernel = kernel[i].unsqueeze(dim=0)
            batch_activation = activation[i].unsqueeze(dim=0)
            conv.append(F.conv2d(batch_activation, batch_kernel, padding='same'))
        conv_stack = torch.stack(conv, dim=0).squeeze(dim=1)
        loss = F.mse_loss(conv_stack, target) / 2 + self.rmagnitude * self.regulator(activation)

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

        activation_loss = F.mse_loss(activation.squeeze(), activation_target.squeeze()) / activation_pixels
        if kernel.shape[-1] > kernel_target.shape[-1]:
            diff = kernel.shape[-1] - kernel_target.shape[-1]
            cropped = slice(diff // 2, diff // 2 + kernel_target.shape[-1])
            kernel_loss = F.mse_loss(kernel[:, :, cropped, cropped], kernel_target) / kernel_pixels
        else:
            kernel_loss = F.mse_loss(kernel, kernel_target) / kernel_pixels

        loss = self.weights[0] * activation_loss + self.weights[1] * kernel_loss

        if torch.cuda.is_available():
            return loss.cuda()
        return loss


# class PixelLl(nn.Module):
#     def __init__(self):
#         super(PixelLl, self).__init__()
#         self.ll = nn.Sequential(nn.Linear(1, 10),
#                                 nn.ReLU(),
#                                 nn.Linear(10, 10),
#                                 nn.ReLU(),
#                                 nn.Linear(10, 2),
#                                 nn.ReLU()
#                                 )
#     def forward(self, x):
#         return self.ll(x)


# class CrossPixelLoss(nn.Module):
#     """
#     Computes pixel-wise cross entropy loss for the activation map (only!)
#     """
#     def __init__(self):
#         super().__init__()
#         self.ll = PixelLl()
#         if torch.cuda.is_available():
#             self.ll.cuda()
#         self.cros_ent_loss = nn.CrossEntropyLoss()
#
#     def forward(self, pred, target):
#         cieled_target = torch.ceil(target)
#         if torch.cuda.is_available():
#             cieled_target.cuda()
#         loss = torch.tensor([], requires_grad=True)
#         for batch in range(pred.shape[0]):
#             for i, pixels in enumerate(pred[batch]):
#                 for j, pixel in enumerate(pixels):
#                     features = self.ll(pixel.unsqueeze(0)).unsqueeze(0)
#                     long_pixel_target = cieled_target[batch][i][j].unsqueeze(0).type(torch.LongTensor)
#                     if torch.cuda.is_available():
#                         long_pixel_target = long_pixel_target.cuda()
#                     new_loss = self.cros_ent_loss(features, long_pixel_target).unsqueeze(0)
#                     torch.cat([loss, new_loss.type(torch.FloatTensor)], dim=0)
#         loss = torch.sum(loss)
#         loss = (loss + F.mse_loss(pred.squeeze(), target.squeeze())) / len(cieled_target[0][0] ** 2)
#
#         if torch.cuda.is_available():
#             return loss.cuda()
#         return loss


class SumIndividualLoss(nn.Module):
    def __init__(self, weights=(0.5, 0.5)):
        super().__init__()
        self.weights = weights
        self.activation_class_loss = nn.CrossEntropyLoss()  # CrossPixelLoss() -> nn.CrossEntropyLoss()

    def forward(self, activation, activation_classes, kernel, activation_target, kernel_target):
        kernel_pixels = kernel.shape[-1] ** 2
        activation_classes_target = 1 * (activation_target > 1)  # Creates class labels
        activation_loss = self.activation_class_loss(activation_classes, activation_classes_target)  # Added
        activation_loss += F.mse_loss(activation.squeeze(), activation_target)  # Added
        activation_loss = activation_loss / (activation.shape[-1] ** 2)  # Added
        if kernel.shape[-1] > kernel_target.shape[-1]:
            diff = kernel.shape[-1] - kernel_target.shape[-1]
            cropped = slice(diff // 2, diff // 2 + kernel_target.shape[-1])
            kernel_loss = F.mse_loss(kernel[:, :, cropped, cropped], kernel_target) / kernel_pixels
        else:
            kernel_loss = F.mse_loss(kernel, kernel_target) / kernel_pixels

        loss = self.weights[0] * activation_loss + self.weights[1] * kernel_loss
        loss = 1e6 * loss
        if torch.cuda.is_available():
            return loss.cuda()
        return loss
