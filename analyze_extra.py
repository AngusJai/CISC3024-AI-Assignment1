"""Extra analyses: ECE, rotated per-class confusion, Grad-CAM galleries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from corruptions import CORRUPTION_FN
from evaluate import get_device, load_model


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            rows.append({"bin": i, "confidence": None, "accuracy": None, "count": 0})
            continue
        acc = correct[m].mean().item()
        avg_conf = conf[m].mean().item()
        ece += (m.float().mean().item()) * abs(acc - avg_conf)
        rows.append({"bin": i, "confidence": avg_conf, "accuracy": acc, "count": int(m.sum())})
    return ece, rows, conf.numpy(), correct.numpy()


@torch.no_grad()
def collect_logits(model, loader, device, corruption="clean"):
    fn = CORRUPTION_FN[corruption]
    probs_all, labels_all, preds_all = [], [], []
    for x, y in loader:
        x = fn(x.to(device))
        y = y.to(device)
        p = F.softmax(model(x), dim=1)
        probs_all.append(p.cpu())
        labels_all.append(y.cpu())
        preds_all.append(p.argmax(1).cpu())
    return torch.cat(probs_all), torch.cat(labels_all), torch.cat(preds_all)


def confusion(preds, labels, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def pair_stats(cm, a, b):
    return {
        f"{a}_as_{b}": int(cm[a, b]),
        f"{b}_as_{a}": int(cm[b, a]),
        f"{a}_total": int(cm[a].sum()),
        f"{b}_total": int(cm[b].sum()),
        f"{a}_recall": float(cm[a, a] / max(cm[a].sum(), 1)),
        f"{b}_recall": float(cm[b, b] / max(cm[b].sum(), 1)),
    }


def plot_reliability(bin_rows, ece, path: Path, title: str):
    xs, ys_acc, ys_conf = [], [], []
    for r in bin_rows:
        if r["count"] == 0:
            continue
        xs.append(r["confidence"])
        ys_acc.append(r["accuracy"])
        ys_conf.append(r["confidence"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.bar(
        [r["confidence"] for r in bin_rows if r["count"]],
        [r["accuracy"] for r in bin_rows if r["count"]],
        width=1 / 15,
        alpha=0.4,
        align="center",
        label="bin accuracy",
    )
    ax.plot(xs, ys_acc, "o-", color="#1d3557", label="accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE={ece:.4f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cm(cm, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grad_cam_maps(model, images, target_classes=None):
    """Grad-CAM on the last stage feature map before GAP head."""
    model.eval()
    feats = {}
    grads = {}

    # Last MLPBlock output lives inside stages; hook the last module of stages
    last = list(model.stages.children())[-1]
    # If last is Sequential of blocks, hook the sequential itself
    target_layer = last

    def fwd_hook(_m, _i, o):
        feats["value"] = o

    def bwd_hook(_m, _gi, go):
        grads["value"] = go[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    images = images.requires_grad_(True)
    logits = model(images)
    if target_classes is None:
        target_classes = logits.argmax(1)
    loss = logits.gather(1, target_classes.view(-1, 1)).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()

    h1.remove()
    h2.remove()

    f = feats["value"]  # B,C,H,W
    g = grads["value"]
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = (weights * f).sum(1)
    cam = F.relu(cam)
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    denom = cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    cam = cam / denom
    cam = F.interpolate(cam.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False)
    return cam.detach(), logits.detach()


def save_gradcam_gallery(model, dataset, device, out_path: Path, n=8):
    # Prefer digit 3 examples
    picks = []
    for img, y in dataset:
        if y == 3:
            picks.append(img)
        if len(picks) >= n:
            break
    if not picks:
        picks = [dataset[i][0] for i in range(n)]
    batch = torch.stack(picks).to(device)
    # show clean + rotated
    from corruptions import rotate_batch

    variants = [
        ("clean", batch),
        ("rot90", rotate_batch(batch, 90)),
    ]
    fig, axes = plt.subplots(len(variants), n, figsize=(n * 1.3, len(variants) * 2.2))
    for r, (name, xb) in enumerate(variants):
        cam, logits = grad_cam_maps(model, xb)
        preds = logits.argmax(1)
        for c in range(n):
            ax = axes[r, c] if len(variants) > 1 else axes[c]
            img = xb[c, 0].detach().cpu().numpy()
            heat = cam[c, 0].cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.imshow(heat, cmap="jet", alpha=0.45)
            ax.set_title(f"{name}\nP:{preds[c].item()}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("outputs/fasternet_aug.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    device = get_device()
    model, _ = load_model(args.ckpt, device)
    tag = args.ckpt.stem
    ds = datasets.MNIST(str(args.data_dir), train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    summary = {"tag": tag}

    for corr in ["clean", "rotate_90"]:
        probs, labels, preds = collect_logits(model, loader, device, corr)
        ece, bins, _, _ = expected_calibration_error(probs, labels)
        cm = confusion(preds, labels)
        plot_cm(cm, args.out_dir / f"{tag}_{corr}_perclass_cm.png", f"{tag} CM ({corr})")
        plot_reliability(bins, ece, args.out_dir / f"{tag}_{corr}_reliability.png", f"{tag} {corr}")
        per_class = {
            str(i): {
                "support": int(cm[i].sum()),
                "correct": int(cm[i, i]),
                "recall": float(cm[i, i] / max(cm[i].sum(), 1)),
            }
            for i in range(10)
        }
        summary[corr] = {
            "ece": ece,
            "accuracy": float((preds == labels).float().mean()),
            "per_class": per_class,
            "confusable_pairs": {
                "3_8": pair_stats(cm, 3, 8),
                "4_9": pair_stats(cm, 4, 9),
                "6_9": pair_stats(cm, 6, 9),
            },
            "confusion_matrix": cm.tolist(),
        }
        print(f"{tag} {corr}: acc={summary[corr]['accuracy']:.4f} ECE={ece:.4f}")
        print("  pairs", summary[corr]["confusable_pairs"])

    save_gradcam_gallery(model, ds, device, args.out_dir / f"{tag}_gradcam.png")
    path = args.out_dir / f"{tag}_extra_analysis.json"
    path.write_text(json.dumps(summary, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
