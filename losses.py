import torch
from torch import nn


class PseudoHuberLoss(nn.Module):
    def __init__(self, c, reduction="mean"):
        super().__init__()
        self.c = c
        self.reduction = reduction

        if reduction == "mean":
            self.reduction_func = torch.mean
        elif reduction == "sum":
            self.reduction_func = torch.sum

    def forward(self, output, target):
        loss = torch.sqrt((output - target) ** 2 + self.c ** 2) - self.c

        if self.reduction != "none":
            loss = self.reduction_func(loss)

        return loss
