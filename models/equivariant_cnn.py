"""Student-scale p4-style rotation-equivariant CNN (no e2cnn).

Shared kernels are applied at the four cardinal rotations (orbit convolution).
Orientation channels are max-pooled before the classifier, yielding approximate
rotation invariance—used as a literature-style baseline vs FasterNet+aug+TTA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrbitConv2d(nn.Module):
    """Z2→p4 lift (or p4→p4 if in_channels already includes orientations).

    Stores one base kernel bank; applies rot90(k) for k in {0,1,2,3} and
    concatenates outputs along the channel axis (×4).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.padding = padding
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bn = nn.BatchNorm2d(out_channels * 4)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for k in range(4):
            w = torch.rot90(self.weight, k, dims=(-2, -1))
            outs.append(F.conv2d(x, w, padding=self.padding))
        y = torch.cat(outs, dim=1)
        return F.relu(self.bn(y), inplace=True)


class OrientationPool(nn.Module):
    """Max over the 4 orientation groups: (B, 4*C, H, W) -> (B, C, H, W)."""

    def __init__(self, channels_per_orient: int):
        super().__init__()
        self.c = channels_per_orient

    def forward(self, x: Tensor) -> Tensor:
        b, c4, h, w = x.shape
        x = x.view(b, 4, self.c, h, w)
        return x.max(dim=1).values


class RotEquivariantCNN(nn.Module):
    """Compact p4 orbit-CNN for 28×28 greyscale digits."""

    def __init__(self, in_chans: int = 1, num_classes: int = 10, width: int = 16):
        super().__init__()
        if in_chans != 1:
            raise ValueError("RotEquivariantCNN is greyscale-only in this assignment")
        self.lift = OrbitConv2d(in_chans, width, 3)  # -> 4*width
        self.pool1 = nn.MaxPool2d(2)
        # input already 4*width; treat as flat channels for next orbit (simple stack)
        self.conv2 = OrbitConv2d(width * 4, width * 2, 3)  # -> 8*width
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = OrbitConv2d(width * 8, width * 4, 3)  # -> 16*width
        self.orient_pool = OrientationPool(width * 4)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.lift(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.orient_pool(x)
        return self.head(x)


if __name__ == "__main__":
    m = RotEquivariantCNN()
    y = m(torch.randn(2, 1, 28, 28))
    print(y.shape, sum(p.numel() for p in m.parameters()))
