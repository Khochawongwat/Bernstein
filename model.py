import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from functools import lru_cache

class BernLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degrees = 3):
        super(BernLayer, self).__init__()

        # Coefficients of the Bernstein polynomials
        self.bern_weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, degrees + 1, dtype = torch.float32)
        )

        # Control points of the Bezier curves
        self.control_points = nn.Parameter(
            torch.zeros(out_channels, degrees + 1, dtype=torch.float32)
        )

        self.sigmoid = nn.Sigmoid()
        self.degrees = degrees

        self.out_channels = out_channels

        self.init_weights()

    @staticmethod
    @lru_cache(maxsize=128)
    def get_basis(x, degrees):
        x_flat = x.view(x.size(0), -1)

        indices = torch.arange(0, degrees + 1).view(-1, 1).float()
        binom_coeff = torch.tensor([math.comb(degrees, i) for i in range(degrees + 1)]).view(-1, 1).float()

        x_powers = x_flat.unsqueeze(1) ** indices
        one_minus_x_powers = (1 - x_flat.unsqueeze(1)) ** (degrees - indices)

        basis = binom_coeff * x_powers * one_minus_x_powers
        return basis

    def init_weights(self):
        nn.init.normal_(self.bern_weights)
        nn.init.uniform_(self.control_points, -1.0, 1.0)

    def forward(self, x):
        # Bound the space to [0, 1]
        x = self.sigmoid(x)
        basis = self.get_basis(x, self.degrees)
        y = torch.einsum('b d l, l o d -> b o', basis, self.bern_weights)
        y = torch.einsum('b o, o d -> b o', y, self.control_points)
        return y.view(-1, self.out_channels)

class BERN(nn.Module):
    def __init__(self, degrees = 3):
        super(BERN, self).__init__()
        self.layers = nn.Sequential(
            BernLayer(1, 10, degrees),
            nn.LayerNorm(10),
            BernLayer(10, 1, degrees)
        )

    def forward(self, x):
        return self.layers(x)