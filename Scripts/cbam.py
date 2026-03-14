# cbam.py
# Convolutional Block Attention Module (CBAM) implementation.
# Applies channel attention followed by spatial attention to refine feature maps.

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Applies channel-wise attention using shared MLP on pooled features."""

    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c // r, c, 1, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        return x * self.sig(a + m)


class SpatialAttention(nn.Module):
    """Applies spatial attention using concatenated avg/max pooled channel descriptors."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([avg, mx], dim=1)
        return x * self.sig(self.conv(x2))


class CBAM(nn.Module):
    """CBAM: sequential channel attention then spatial attention."""

    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x