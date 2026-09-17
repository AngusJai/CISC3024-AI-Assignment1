"""Image corruption utilities for robustness evaluation and training aug."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def rotate_batch(images: Tensor, degrees: float) -> Tensor:
    """Rotate a BCHW batch by a fixed angle (degrees, counterclockwise)."""
    angle = math.radians(degrees)
    theta = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0]],
        dtype=images.dtype,
        device=images.device,
    )
    theta = theta.unsqueeze(0).repeat(images.size(0), 1, 1)
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


def add_gaussian_noise(images: Tensor, std: float = 0.25) -> Tensor:
    noise = torch.randn_like(images) * std
    return (images + noise).clamp(0.0, 1.0)


def add_salt_pepper(images: Tensor, amount: float = 0.05) -> Tensor:
    out = images.clone()
    rnd = torch.rand_like(out)
    out = torch.where(rnd < amount / 2, torch.zeros_like(out), out)
    out = torch.where(rnd > 1.0 - amount / 2, torch.ones_like(out), out)
    return out


def gaussian_blur(images: Tensor, kernel_size: int = 5, sigma: float = 1.2) -> Tensor:
    # Build a separable-ish 2D Gaussian kernel and apply depthwise conv.
    coords = torch.arange(kernel_size, dtype=images.dtype, device=images.device) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(images.size(1), 1, 1, 1)
    pad = kernel_size // 2
    return F.conv2d(images, kernel, padding=pad, groups=images.size(1)).clamp(0.0, 1.0)


def shift_batch(images: Tensor, max_shift: int = 3) -> Tensor:
    """Apply a random integer translation per image within [-max_shift, max_shift]."""
    b, _, h, w = images.shape
    out = torch.zeros_like(images)
    for i in range(b):
        dy = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        dx = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        y1_src, y2_src = max(0, -dy), min(h, h - dy)
        x1_src, x2_src = max(0, -dx), min(w, w - dx)
        y1_dst, y2_dst = max(0, dy), min(h, h + dy)
        x1_dst, x2_dst = max(0, dx), min(w, w + dx)
        out[i, :, y1_dst:y2_dst, x1_dst:x2_dst] = images[i, :, y1_src:y2_src, x1_src:x2_src]
    return out


def mild_occlusion(images: Tensor, patch_frac: float = 0.25) -> Tensor:
    """Zero out a random square patch (fraction of side length)."""
    b, _, h, w = images.shape
    out = images.clone()
    ph = max(1, int(h * patch_frac))
    pw = max(1, int(w * patch_frac))
    for i in range(b):
        y0 = int(torch.randint(0, h - ph + 1, (1,)).item())
        x0 = int(torch.randint(0, w - pw + 1, (1,)).item())
        out[i, :, y0 : y0 + ph, x0 : x0 + pw] = 0.0
    return out


def elastic_distort(
    images: Tensor, alpha: float = 18.0, sigma: float = 4.0
) -> Tensor:
    """Lightweight elastic-style warp via smoothed random displacement field."""
    b, _, h, w = images.shape
    device = images.device
    dtype = images.dtype
    # Random displacement, blurred
    dx = torch.randn(b, 1, h, w, device=device, dtype=dtype)
    dy = torch.randn(b, 1, h, w, device=device, dtype=dtype)
    dx = gaussian_blur(dx, kernel_size=7, sigma=sigma) * alpha
    dy = gaussian_blur(dy, kernel_size=7, sigma=sigma) * alpha

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    # Convert pixel displacements to normalized grid offsets
    disp_x = (2.0 * dx.squeeze(1) / max(w - 1, 1)).unsqueeze(-1)
    disp_y = (2.0 * dy.squeeze(1) / max(h - 1, 1)).unsqueeze(-1)
    grid = base + torch.cat([disp_x, disp_y], dim=-1)
    return F.grid_sample(images, grid, align_corners=True, padding_mode="zeros").clamp(0, 1)


CORRUPTION_FN = {
    "clean": lambda x: x,
    "rotate_30": lambda x: rotate_batch(x, 30),
    "rotate_45": lambda x: rotate_batch(x, 45),
    "rotate_90": lambda x: rotate_batch(x, 90),
    "rotate_m45": lambda x: rotate_batch(x, -45),
    "gaussian_noise": lambda x: add_gaussian_noise(x, 0.25),
    "salt_pepper": lambda x: add_salt_pepper(x, 0.05),
    "blur": lambda x: gaussian_blur(x, 5, 1.2),
    "shift": lambda x: shift_batch(x, 3),
    "occlusion": lambda x: mild_occlusion(x, 0.25),
    "elastic": lambda x: elastic_distort(x, 18.0, 4.0),
}


def apply_train_augmentations(images: Tensor) -> Tensor:
    """Stochastic mix used for the robust-aug training run."""
    out = images
    r = torch.rand(1).item()
    if r < 0.35:
        # Explicit cardinal rotations — critical for the Rot90 test target.
        angle = float(torch.tensor([0.0, 90.0, 180.0, 270.0, -90.0])[torch.randint(0, 5, (1,))])
        out = rotate_batch(out, angle)
    elif r < 0.85:
        angle = float(torch.empty(1).uniform_(-90, 90).item())
        out = rotate_batch(out, angle)
    if torch.rand(1).item() < 0.25:
        out = add_salt_pepper(out, amount=0.04)
    if torch.rand(1).item() < 0.2:
        out = elastic_distort(out, alpha=12.0, sigma=3.5)
    if torch.rand(1).item() < 0.4:
        out = add_gaussian_noise(out, std=float(torch.empty(1).uniform_(0.05, 0.25).item()))
    if torch.rand(1).item() < 0.3:
        out = shift_batch(out, max_shift=2)
    if torch.rand(1).item() < 0.2:
        out = gaussian_blur(out, 3, 0.8)
    if torch.rand(1).item() < 0.15:
        out = mild_occlusion(out, 0.2)
    return out.clamp(0.0, 1.0)
