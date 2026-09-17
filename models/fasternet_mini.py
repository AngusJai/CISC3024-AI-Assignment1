"""
FasterNet-mini for MNIST digit recognition.

Faithful reproduction of the core ideas in:
  Chen et al., "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks"
  CVPR 2023. arXiv:2303.03667
  https://github.com/JierunChen/FasterNet

Adapted for 1x28x28 grayscale inputs (MNIST) instead of ImageNet 224x224.
Core mechanism preserved: Partial Convolution (PConv) spatial mixing + pointwise MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PartialConv3(nn.Module):
    """Apply a 3x3 conv to only 1/n_div of channels; leave the rest untouched.

    This is the central FasterNet operator (PConv): lower FLOPs *and* memory
    access than depthwise conv, while still mixing spatial information.
    """

    def __init__(self, dim: int, n_div: int = 4):
        super().__init__()
        self.dim_conv = max(dim // n_div, 1)
        self.dim_untouched = dim - self.dim_conv
        self.partial_conv = nn.Conv2d(
            self.dim_conv, self.dim_conv, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        return torch.cat((x1, x2), dim=1)


def _make_act(act: str) -> nn.Module:
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported act={act}")


class MLPBlock(nn.Module):
    """FasterNet block: PConv spatial mixer -> residual pointwise MLP."""

    def __init__(self, dim: int, n_div: int = 4, mlp_ratio: float = 2.0, act: str = "relu"):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.spatial_mixing = PartialConv3(dim, n_div)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            _make_act(act),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        return shortcut + self.mlp(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.reduction(x))


class FasterNetMini(nn.Module):
    """Scaled FasterNet for MNIST (1x28x28 -> 10 classes).

    Default layout (embed_dim=32, depths=(1, 2, 2)):
      stem 2x2 patchify -> 14x14
      stage1 @ 32 ch
      merge -> 7x7 @ 64 ch
      stage2 @ 64 ch
      merge -> 3x3 @ 128 ch
      stage3 @ 128 ch
      GAP + linear head
    """

    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 10,
        embed_dim: int = 32,
        depths: tuple[int, ...] = (1, 2, 2),
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        feature_dim: int = 256,
        act: str = "relu",
    ):
        super().__init__()
        self.act = act
        self.num_stages = len(depths)
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size=2)

        stages: list[nn.Module] = []
        for i, depth in enumerate(depths):
            dim = embed_dim * (2**i)
            blocks = nn.Sequential(
                *[
                    MLPBlock(dim, n_div=n_div, mlp_ratio=mlp_ratio, act=act)
                    for _ in range(depth)
                ]
            )
            stages.append(blocks)
            if i < self.num_stages - 1:
                stages.append(PatchMerging(dim))
        self.stages = nn.Sequential(*stages)

        num_features = embed_dim * (2 ** (self.num_stages - 1))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, feature_dim),
            _make_act(act),
            nn.Linear(feature_dim, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_fasternet(variant: str = "mini", num_classes: int = 10) -> FasterNetMini:
    """Variants inspired by FasterNet-T0 scaling, adapted to 28x28 digits.

    - mini: embed 32, depths (1,2,2) — default student-scale model
    - t0lite: embed 40, depths (1,2,4) — closer to FasterNet-T0 stage ratios
      (paper T0 uses embed 40 and deeper stage-3 on ImageNet; we keep 3 stages
      because 28x28 cannot afford a 4-stride ImageNet stem)
    """
    if variant == "mini":
        return FasterNetMini(
            num_classes=num_classes, embed_dim=32, depths=(1, 2, 2), feature_dim=256
        )
    if variant == "t0lite":
        return FasterNetMini(
            num_classes=num_classes, embed_dim=40, depths=(1, 2, 4), feature_dim=320
        )
    raise ValueError(variant)


if __name__ == "__main__":
    for v in ("mini", "t0lite"):
        m = build_fasternet(v)
        x = torch.randn(2, 1, 28, 28)
        y = m(x)
        print(v, "out", y.shape, "params", count_params(m))
