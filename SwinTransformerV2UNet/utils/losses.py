import torch
import torch.nn as nn
import numpy as np
from fastai.learner import Metric
from fastai.torch_core import flatten_check
import torch.nn.functional as F



class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #         inputs = nnF.sigmoid(inputs)

        # flatten label and prediction tensors

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        inputs = F.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class Dice_th_pred(Metric):
    def __init__(self, ths=np.arange(0.1, 0.9, 0.01), axis=1):
        self.axis = axis
        self.ths = ths
        self.reset()

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self, p, t):
        pred, targ = flatten_check(p, t)
        for i, th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p * targ).float().sum().item()
            self.union[i] += (p + targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 2.0 * self.inter / self.union, torch.zeros_like(self.union))
        return dices
