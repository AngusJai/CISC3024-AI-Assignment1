"""Full robustness evaluation + galleries + reject analysis."""

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

from corruptions import CORRUPTION_FN, rotate_batch
from models import FasterNetMini, SimpleCNN, count_params


@torch.no_grad()
def predict_with_rotation_tta(model, x: torch.Tensor, angles=(0, 90, 180, 270)) -> torch.Tensor:
    """Average softmax over cardinal rotations (test-time augmentation)."""
    probs = None
    for a in angles:
        xi = x if a == 0 else rotate_batch(x, float(a))
        p = F.softmax(model(xi), dim=1)
        probs = p if probs is None else probs + p
    return probs / len(angles)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = state["model_name"]
    variant = state.get("variant", "mini")
    n_div = int(state.get("n_div", 4))
    in_chans = int(state.get("in_chans", 1))
    act = state.get("act", "relu")
    if name == "fasternet":
        from models.fasternet_mini import FasterNetMini

        if variant == "t0lite":
            model = FasterNetMini(
                in_chans=in_chans,
                embed_dim=40,
                depths=(1, 2, 4),
                n_div=n_div,
                feature_dim=320,
                act=act,
            )
        else:
            model = FasterNetMini(
                in_chans=in_chans,
                embed_dim=32,
                depths=(1, 2, 2),
                n_div=n_div,
                feature_dim=256,
                act=act,
            )
    elif name == "equivariant":
        from models.equivariant_cnn import RotEquivariantCNN

        model = RotEquivariantCNN(in_chans=in_chans)
    else:
        model = SimpleCNN()
        if in_chans != 1:
            model.features[0] = __import__("torch").nn.Conv2d(in_chans, 32, 3, padding=1)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, state


@torch.no_grad()
def eval_corruption(
    model,
    loader,
    device,
    corruption: str,
    max_batches: int | None = None,
    use_tta: bool = False,
):
    fn = CORRUPTION_FN[corruption]
    correct, n = 0, 0
    all_probs, all_preds, all_labels = [], [], []
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = fn(x.to(device))
        y = y.to(device)
        if use_tta:
            probs = predict_with_rotation_tta(model, x)
        else:
            probs = F.softmax(model(x), dim=1)
        preds = probs.argmax(1)
        correct += (preds == y).sum().item()
        n += y.numel()
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    conf = probs.max(1).values
    return {
        "accuracy": correct / n,
        "n": n,
        "preds": preds,
        "labels": labels,
        "confidence": conf,
        "probs": probs,
    }


def reject_curve(confidence: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor, thresholds):
    rows = []
    for t in thresholds:
        keep = confidence >= t
        covered = keep.float().mean().item()
        if keep.sum() == 0:
            acc = None
        else:
            acc = (preds[keep] == labels[keep]).float().mean().item()
        rows.append({"threshold": t, "coverage": covered, "accuracy_on_accepted": acc})
    return rows


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def save_gallery(images, titles, path: Path, nrows=2, ncols=8, cmap="gray"):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.6))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i >= len(images):
            ax.axis("off")
            continue
        img = images[i]
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        if img.ndim == 3:
            img = img.squeeze(0)
        ax.imshow(img, cmap=cmap)
        ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def make_demo_galleries(model, test_set, device, out_dir: Path, tag: str):
    # Pick examples of digit 3 and rotate them
    threes = [(img, y) for img, y in test_set if y == 3][:16]
    imgs = torch.stack([img for img, _ in threes]).to(device)
    rotated = CORRUPTION_FN["rotate_90"](imgs)
    logits = model(rotated)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(1)
    conf = probs.max(1).values
    titles = [
        f"T:3 P:{p.item()}\np={c.item():.2f}"
        for p, c in zip(preds, conf)
    ]
    colors_ok = (preds.cpu() == 3)
    # save rotated gallery
    save_gallery(
        [rotated[i].cpu() for i in range(min(16, len(rotated)))],
        titles,
        out_dir / f"{tag}_rotated3_gallery.png",
    )

    # Hard cases: find mistakes on clean
    loader = DataLoader(test_set, batch_size=512, shuffle=False)
    hard_imgs, hard_titles = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(1)
        conf = probs.max(1).values
        wrong = preds != y
        for i in torch.where(wrong)[0][:]:
            hard_imgs.append(x[i].cpu())
            hard_titles.append(
                f"T:{y[i].item()} P:{preds[i].item()}\np={conf[i].item():.2f}"
            )
            if len(hard_imgs) >= 16:
                break
        if len(hard_imgs) >= 16:
            break
    if hard_imgs:
        save_gallery(hard_imgs, hard_titles, out_dir / f"{tag}_hard_cases.png")

    # Multi-corruption showcase for one digit-3
    base = threes[0][0].unsqueeze(0).to(device)
    show_names = ["clean", "rotate_45", "rotate_90", "gaussian_noise", "blur", "shift", "occlusion", "elastic"]
    show_imgs, show_titles = [], []
    for name in show_names:
        xi = CORRUPTION_FN[name](base)
        logits = model(xi)
        prob = F.softmax(logits, dim=1)[0]
        pred = int(prob.argmax().item())
        p = float(prob.max().item())
        show_imgs.append(xi[0].cpu())
        show_titles.append(f"{name}\nP:{pred} ({p:.2f})")
    save_gallery(show_imgs, show_titles, out_dir / f"{tag}_corruption_showcase.png", nrows=2, ncols=4)

    return bool(colors_ok.float().mean().item() > 0.5)


def plot_corruption_bars(table: dict, path: Path, title: str):
    names = list(table.keys())
    accs = [table[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), accs, color="#2a6f97")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history: dict, path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history["epoch"], history["train_loss"], color="tab:red")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["test_acc"], label="test")
    axes[1].set_title(title)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--reject-threshold", type=float, default=0.7)
    args = p.parse_args()

    device = get_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, meta = load_model(args.ckpt, device)
    tag = args.ckpt.stem
    dataset_name = meta.get("dataset", "mnist")

    if dataset_name == "emnist":
        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda t: t.transpose(1, 2))]
        )
        test_set = datasets.EMNIST(
            str(args.data_dir), split="digits", train=False, download=True, transform=tfm
        )
        g = torch.Generator().manual_seed(0)
        test_idx = torch.randperm(len(test_set), generator=g)[:10000]
        test_set = torch.utils.data.Subset(test_set, test_idx.tolist())
    elif dataset_name == "svhn":
        tfm = transforms.ToTensor()
        test_set = datasets.SVHN(str(args.data_dir), split="test", download=True, transform=tfm)
        g = torch.Generator().manual_seed(0)
        test_idx = torch.randperm(len(test_set), generator=g)[:10000]
        test_set = torch.utils.data.Subset(test_set, test_idx.tolist())
    else:
        tfm = transforms.ToTensor()
        test_set = datasets.MNIST(str(args.data_dir), train=False, download=True, transform=tfm)
    loader = DataLoader(test_set, batch_size=256, shuffle=False)

    table = {}
    clean_pack = None
    for name in CORRUPTION_FN:
        pack = eval_corruption(model, loader, device, name)
        table[name] = {"accuracy": pack["accuracy"], "n": pack["n"]}
        print(f"{tag} | {name:16s} acc={pack['accuracy']:.4f}")
        if name == "clean":
            clean_pack = pack

    # Extra: rotation TTA metrics (reported separately; does not replace single-pass table)
    tta_rotate_90 = eval_corruption(model, loader, device, "rotate_90", use_tta=True)
    tta_clean = eval_corruption(model, loader, device, "clean", use_tta=True)
    table["rotate_90_tta"] = {"accuracy": tta_rotate_90["accuracy"], "n": tta_rotate_90["n"]}
    table["clean_tta"] = {"accuracy": tta_clean["accuracy"], "n": tta_clean["n"]}
    print(f"{tag} | {'rotate_90_tta':16s} acc={tta_rotate_90['accuracy']:.4f}")
    print(f"{tag} | {'clean_tta':16s} acc={tta_clean['accuracy']:.4f}")

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    reject = reject_curve(
        clean_pack["confidence"], clean_pack["preds"], clean_pack["labels"], thresholds
    )
    # operating point
    t = args.reject_threshold
    keep = clean_pack["confidence"] >= t
    reject_op = {
        "threshold": t,
        "coverage": keep.float().mean().item(),
        "accuracy_on_accepted": (clean_pack["preds"][keep] == clean_pack["labels"][keep])
        .float()
        .mean()
        .item()
        if keep.any()
        else None,
        "num_rejected": int((~keep).sum().item()),
    }

    cm = confusion_matrix(clean_pack["preds"], clean_pack["labels"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion ({tag})")
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
    fig.savefig(args.out_dir / f"{tag}_confusion.png", dpi=150)
    plt.close(fig)

    plot_corruption_bars(table, args.out_dir / f"{tag}_corruption_bars.png", f"Robustness: {tag}")
    if meta.get("dataset", "mnist") in ("mnist", "emnist"):
        make_demo_galleries(model, test_set, device, args.out_dir, tag)

    # train curves if json exists
    train_json = args.out_dir / f"{tag}_train.json"
    if train_json.exists():
        with open(train_json) as f:
            tr = json.load(f)
        plot_curves(tr["history"], args.out_dir / f"{tag}_curves.png", f"Accuracy ({tag})")

    summary = {
        "tag": tag,
        "params": count_params(model),
        "corruption_table": table,
        "reject_curve": reject,
        "reject_operating_point": reject_op,
        "confusion_matrix": cm.tolist(),
        "mean_confidence_clean": float(clean_pack["confidence"].mean()),
    }
    with open(args.out_dir / f"{tag}_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"tag": tag, "clean": table["clean"], "rotate_90": table["rotate_90"], "reject": reject_op}, indent=2))


if __name__ == "__main__":
    main()
