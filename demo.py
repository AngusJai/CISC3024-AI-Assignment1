"""Batch demo: predict labels + confidence, with optional reject."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from corruptions import CORRUPTION_FN
from evaluate import load_model, get_device, save_gallery


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("outputs/fasternet_aug.pt"))
    p.add_argument("--corruption", default="rotate_90", choices=list(CORRUPTION_FN))
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--reject-threshold", type=float, default=0.7)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("outputs/demo_batch.png"))
    args = p.parse_args()

    device = get_device()
    model, _ = load_model(args.ckpt, device)
    ds = datasets.MNIST(str(args.data_dir), train=False, download=True, transform=transforms.ToTensor())
    imgs, labels = [], []
    for i in range(args.n):
        x, y = ds[i]
        imgs.append(x)
        labels.append(y)
    batch = torch.stack(imgs).to(device)
    batch = CORRUPTION_FN[args.corruption](batch)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)
        conf, pred = probs.max(1)

    titles = []
    for i in range(args.n):
        if conf[i].item() < args.reject_threshold:
            titles.append(f"T:{labels[i]} REJECT\np={conf[i].item():.2f}")
        else:
            titles.append(f"T:{labels[i]} P:{pred[i].item()}\np={conf[i].item():.2f}")
        print(titles[-1].replace("\n", " "))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gallery([batch[i].cpu() for i in range(args.n)], titles, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
