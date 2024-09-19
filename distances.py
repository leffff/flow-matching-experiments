import math

import torch
from torch import linalg


def l2_dist(X: torch.Tensor, Y: torch.Tensor):
    flat_X = X.view(X.size(0), -1)
    flat_Y = Y.view(Y.size(0), -1)

    # Calculate the squared differences
    squared_diff = (flat_X.unsqueeze(1) - flat_Y.unsqueeze(0)).pow(2)

    # Sum along the channel dimension
    sum_squared_diff = squared_diff.sum(dim=2)

    # Take the square root to get Euclidean distances
    distance_matrix = sum_squared_diff.sqrt()

    return distance_matrix