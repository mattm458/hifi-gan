from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import (remove_spectral_norm, remove_weight_norm,
                            spectral_norm, weight_norm)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()

        self.period = period
        self.use_spectral_norm = use_spectral_norm

        kernel_size = (kernel_size, 1)
        stride = (stride, 1)
        padding = (get_padding(5, 1), 0)

        conv1 = nn.Conv2d(1, 32, kernel_size, stride, padding=padding)
        conv2 = nn.Conv2d(32, 128, kernel_size, stride, padding=padding)
        conv3 = nn.Conv2d(128, 512, kernel_size, stride, padding=padding)
        conv4 = nn.Conv2d(512, 1024, kernel_size, stride, padding=padding)
        conv5 = nn.Conv2d(1024, 1024, kernel_size, 1, padding=(2, 0))

        self.convs = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

        self.weight_norms = [conv1, conv2, conv3, conv4, conv5, self.conv_post]

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def weight_norm(self):
        weight_norm_fn = spectral_norm if self.use_spectral_norm else weight_norm
        for i in self.weight_norms:
            weight_norm_fn(i)

    def remove_weight_norm(self):
        remove_weight_norm_fn = (
            remove_spectral_norm if self.use_spectral_norm else remove_weight_norm
        )
        for i in self.weight_norms:
            remove_weight_norm_fn(i)


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()

        discriminators = [
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11),
        ]

        self.discriminators = nn.ModuleList(discriminators)

        self.weight_norm_calls = discriminators

    def forward(
        self, y: Tensor, y_hat: Tensor
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        y = y.unsqueeze(1)
        y_hat = y_hat.unsqueeze(1)

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
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

    def remove_weight_norm(self):
        for i in self.weight_norm_calls:
            i.remove_weight_norm()
