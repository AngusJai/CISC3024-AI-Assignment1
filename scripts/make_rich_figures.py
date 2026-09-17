#!/usr/bin/env python3
"""Rich figure pack: multi-angle rotations + multi-corruption for digits 0-9."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from analyze_extra import grad_cam_maps
from corruptions import CORRUPTION_FN, rotate_batch
from evaluate import get_device, load_model, predict_with_rotation_tta

OUT = ROOT / "outputs"


def pick_one_per_digit(dataset):
    by = {i: None for i in range(10)}
    for img, y in dataset:
        y = int(y)
        if y in by and by[y] is None:
            by[y] = img
        if all(v is not None for v in by.values()):
            break
    return torch.stack([by[i] for i in range(10)])


@torch.no_grad()
def predict_row(model, x):
    probs = F.softmax(model(x), dim=1)
    conf, pred = probs.max(1)
    return pred.cpu(), conf.cpu()


def panel_grid(images_rows, titles_rows, row_labels, path: Path, suptitle: str):
    nrows = len(images_rows)
    ncols = images_rows[0].shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.15, nrows * 1.55))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            img = images_rows[r][c]
            if torch.is_tensor(img):
                img = img.detach().cpu()
                if img.ndim == 3:
                    img = img[0]
                img = img.numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(titles_rows[r][c], fontsize=7)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    device = get_device()
    # Prefer strongest MNIST aug ckpt; fall back
    ckpt = OUT / "fasternet_aug.pt"
    model, _ = load_model(ckpt, device)
    ds = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    batch = pick_one_per_digit(ds).to(device)

    # --- Multi-angle rotations ---
    angles = [0, 30, 45, 90, 180, -45]
    imgs, titles, labels = [], [], []
    for a in angles:
        x = batch if a == 0 else rotate_batch(batch, float(a))
        pred, conf = predict_row(model, x)
        imgs.append(x.cpu())
        titles.append(
            [f"T:{t}\nP:{int(pred[t])} ({conf[t]:.2f})" for t in range(10)]
        )
        labels.append(f"rot {a}°" if a != 0 else "clean")
    panel_grid(
        imgs,
        titles,
        labels,
        OUT / "fig_mnist_multangle_0to9.png",
        "MNIST digits 0-9 under multiple rotation angles (FasterNet-aug)",
    )

    # --- Multi-corruption on 0-9 (selected set) ---
    corr_names = [
        "clean",
        "rotate_45",
        "rotate_90",
        "gaussian_noise",
        "blur",
        "shift",
        "occlusion",
        "elastic",
    ]
    imgs, titles, labels = [], [], []
    for name in corr_names:
        x = CORRUPTION_FN[name](batch)
        pred, conf = predict_row(model, x)
        imgs.append(x.cpu())
        titles.append(
            [f"T:{t}\nP:{int(pred[t])} ({conf[t]:.2f})" for t in range(10)]
        )
        labels.append(name)
    panel_grid(
        imgs,
        titles,
        labels,
        OUT / "fig_mnist_multicorr_0to9.png",
        "MNIST digits 0-9 under multiple corruptions (FasterNet-aug)",
    )

    # --- Rot90 with / without TTA ---
    rot90 = rotate_batch(batch, 90)
    pred, conf = predict_row(model, rot90)
    with torch.no_grad():
        probs = predict_with_rotation_tta(model, rot90)
        conf_t, pred_t = probs.max(1)
    panel_grid(
        [rot90.cpu(), rot90.cpu()],
        [
            [f"T:{t}\nP:{int(pred[t])} ({conf[t]:.2f})" for t in range(10)],
            [f"T:{t}\nTTA P:{int(pred_t[t])} ({conf_t[t]:.2f})" for t in range(10)],
        ],
        ["rot90", "rot90+TTA"],
        OUT / "fig_mnist_rot90_vs_tta_0to9.png",
        "Single-pass vs rotation TTA on 90-degree digits 0-9",
    )

    # --- Grad-CAM multi-angle for a few digits ---
    show_digits = [2, 3, 6, 8, 9]
    sub = batch[show_digits]
    variants = [("clean", sub), ("rot45", rotate_batch(sub, 45)), ("rot90", rotate_batch(sub, 90))]
    fig, axes = plt.subplots(len(variants), len(show_digits), figsize=(10, 4.2))
    for r, (name, xb) in enumerate(variants):
        cam, logits = grad_cam_maps(model, xb)
        preds = logits.argmax(1)
        for c, d in enumerate(show_digits):
            ax = axes[r, c]
            ax.imshow(xb[c, 0].detach().cpu().numpy(), cmap="gray")
            ax.imshow(cam[c, 0].detach().cpu().numpy(), cmap="jet", alpha=0.45)
            ax.set_title(f"{name} T:{d} P:{int(preds[c])}", fontsize=7)
            ax.axis("off")
    fig.suptitle("Grad-CAM across angles (selected digits)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig_gradcam_multiangle.png", dpi=150)
    plt.close(fig)
    print("saved", OUT / "fig_gradcam_multiangle.png")

    # EMNIST figures if checkpoint exists
    emnist_ckpt = OUT / "fasternet_aug_emnist.pt"
    if emnist_ckpt.exists():
        make_emnist_figures(emnist_ckpt, device)
    else:
        print("EMNIST ckpt not ready yet; skip EMNIST figures")


def make_emnist_figures(ckpt: Path, device):
    model, _ = load_model(ckpt, device)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda t: t.transpose(1, 2))]
    )
    ds = datasets.EMNIST("data", split="digits", train=False, download=True, transform=tfm)
    # subsample indices already random in train; here just iterate
    batch = pick_one_per_digit(ds).to(device)
    angles = [0, 45, 90]
    imgs, titles, labels = [], [], []
    for a in angles:
        x = batch if a == 0 else rotate_batch(batch, float(a))
        pred, conf = predict_row(model, x)
        imgs.append(x.cpu())
        titles.append(
            [f"T:{t}\nP:{int(pred[t])} ({conf[t]:.2f})" for t in range(10)]
        )
        labels.append("clean" if a == 0 else f"rot {a}°")
    panel_grid(
        imgs,
        titles,
        labels,
        OUT / "fig_emnist_multangle_0to9.png",
        "EMNIST Digits 0-9 under rotations (FasterNet-aug)",
    )
    corr_names = ["clean", "rotate_90", "gaussian_noise", "blur", "occlusion", "elastic"]
    imgs, titles, labels = [], [], []
    for name in corr_names:
        x = CORRUPTION_FN[name](batch)
        pred, conf = predict_row(model, x)
        imgs.append(x.cpu())
        titles.append(
            [f"T:{t}\nP:{int(pred[t])} ({conf[t]:.2f})" for t in range(10)]
        )
        labels.append(name)
    panel_grid(
        imgs,
        titles,
        labels,
        OUT / "fig_emnist_multicorr_0to9.png",
        "EMNIST Digits 0-9 under corruptions (FasterNet-aug)",
    )


if __name__ == "__main__":
    main()
