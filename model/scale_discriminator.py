from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import (remove_spectral_norm, remove_weight_norm,
                            spectral_norm, weight_norm)

LRELU_SLOPE = 0.1


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        self.use_spectral_norm = use_spectral_norm

        conv1 = nn.Conv1d(1, 128, 15, 1, padding=7)
        conv2 = nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)
        conv3 = nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)
        conv4 = nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)
        conv5 = nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)
        conv6 = nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)
        conv7 = nn.Conv1d(1024, 1024, 5, 1, padding=2)

        conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)

        self.convs = nn.ModuleList([conv1, conv2, conv3, conv4, conv5, conv6, conv7])
        self.conv_post = conv_post

        self.weight_norms = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv_post]

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def weight_norm(self):
        weight_norm_func = spectral_norm if self.use_spectral_norm else weight_norm

        for i in self.weight_norms:
            weight_norm_func(i)

    def remove_weight_norm(self):
        remove_weight_norm_func = (
            remove_spectral_norm if self.use_spectral_norm else remove_weight_norm
        )

        for i in self.weight_norms:
            remove_weight_norm_func(i)


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        discriminator1 = ScaleDiscriminator(use_spectral_norm=True)
        discriminator2 = ScaleDiscriminator()
        discriminator3 = ScaleDiscriminator()

        self.discriminators = nn.ModuleList(
            [discriminator1, discriminator2, discriminator3]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

        self.weight_norm_calls = [discriminator1, discriminator2, discriminator3]

    def forward(self, y, y_hat):
        y = y.unsqueeze(1)
        y_hat = y_hat.unsqueeze(1)

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def weight_norm(self):
        for i in self.weight_norm_calls:
            i.weight_norm()
