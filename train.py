"""Train FasterNet-mini and SimpleCNN on MNIST (clean and robust-aug)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from corruptions import apply_train_augmentations
from models import FasterNetMini, RotEquivariantCNN, SimpleCNN, count_params
from models.fasternet_mini import build_fasternet


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_aug: bool,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_aug:
            x = apply_train_augmentations(x)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        n += bs
    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return correct / n


def build_model(
    name: str,
    variant: str = "mini",
    in_chans: int = 1,
    n_div: int = 4,
    act: str = "relu",
) -> nn.Module:
    if name == "fasternet":
        if variant == "mini":
            return FasterNetMini(
                in_chans=in_chans,
                embed_dim=32,
                depths=(1, 2, 2),
                n_div=n_div,
                feature_dim=256,
                act=act,
            )
        if variant == "t0lite":
            return FasterNetMini(
                in_chans=in_chans,
                embed_dim=40,
                depths=(1, 2, 4),
                n_div=n_div,
                feature_dim=320,
                act=act,
            )
        raise ValueError(variant)
    if name == "baseline":
        if in_chans != 1:
            # lightweight RGB baseline
            m = SimpleCNN()
            # replace first conv
            m.features[0] = nn.Conv2d(in_chans, 32, 3, padding=1)
            return m
        return SimpleCNN()
    if name == "equivariant":
        return RotEquivariantCNN(in_chans=in_chans)
    raise ValueError(name)


def load_digit_datasets(dataset: str, data_dir: Path, tfm):
    if dataset == "mnist":
        train_set = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
        test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
        return train_set, test_set, 1
    if dataset == "emnist":
        emnist_tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.transpose(1, 2)),
            ]
        )
        train_set = datasets.EMNIST(
            str(data_dir), split="digits", train=True, download=True, transform=emnist_tfm
        )
        test_set = datasets.EMNIST(
            str(data_dir), split="digits", train=False, download=True, transform=emnist_tfm
        )
        g = torch.Generator().manual_seed(0)
        train_idx = torch.randperm(len(train_set), generator=g)[:60000]
        test_idx = torch.randperm(len(test_set), generator=g)[:10000]
        train_set = torch.utils.data.Subset(train_set, train_idx.tolist())
        test_set = torch.utils.data.Subset(test_set, test_idx.tolist())
        return train_set, test_set, 1
    if dataset == "svhn":
        # Real-world street-view digits (RGB 32x32). Core train set.
        svhn_tfm = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.SVHN(str(data_dir), split="train", download=True, transform=svhn_tfm)
        test_set = datasets.SVHN(str(data_dir), split="test", download=True, transform=svhn_tfm)
        # subsample for student runtime while keeping scene diversity
        g = torch.Generator().manual_seed(0)
        train_idx = torch.randperm(len(train_set), generator=g)[:60000]
        test_idx = torch.randperm(len(test_set), generator=g)[:10000]
        train_set = torch.utils.data.Subset(train_set, train_idx.tolist())
        test_set = torch.utils.data.Subset(test_set, test_idx.tolist())
        return train_set, test_set, 3
    raise ValueError(dataset)


def run_training(
    model_name: str,
    use_aug: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    data_dir: Path,
    out_dir: Path,
    seed: int = 0,
    dataset: str = "mnist",
    variant: str = "mini",
    tag_suffix: str = "",
    n_div: int = 4,
    act: str = "relu",
) -> dict:
    torch.manual_seed(seed)
    np_seed = seed
    try:
        import numpy as np

        np.random.seed(np_seed)
    except Exception:
        pass
    device = get_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_set, test_set, in_chans = load_digit_datasets(dataset, data_dir, tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    model = build_model(
        model_name, variant=variant, in_chans=in_chans, n_div=n_div, act=act
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0
    tag = f"{model_name}_{'aug' if use_aug else 'clean'}"
    if dataset != "mnist":
        tag += f"_{dataset}"
    if variant != "mini":
        tag += f"_{variant}"
    if n_div != 4:
        tag += f"_ndiv{n_div}"
    if act != "relu":
        tag += f"_{act}"
    if tag_suffix:
        tag += tag_suffix
    ckpt_path = out_dir / f"{tag}.pt"

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device, use_aug)
        te_acc = evaluate(model, test_loader, device)
        sched.step()
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "use_aug": use_aug,
                    "variant": variant,
                    "dataset": dataset,
                    "seed": seed,
                    "n_div": n_div,
                    "in_chans": in_chans,
                    "act": act,
                },
                ckpt_path,
            )
        print(
            f"[{tag}] seed={seed} epoch {epoch:03d}/{epochs} "
            f"loss={tr_loss:.4f} train={tr_acc:.4f} test={te_acc:.4f}"
        )

    elapsed = time.perf_counter() - t0
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    final_acc = evaluate(model, test_loader, device)

    model.eval()
    xb, _ = next(iter(test_loader))
    xb = xb.to(device)
    with torch.no_grad():
        for _ in range(10):
            model(xb)
        if device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        reps = 50
        for _ in range(reps):
            model(xb)
        if device.type == "mps":
            torch.mps.synchronize()
        latency_ms = (time.perf_counter() - t1) / reps * 1000

    # quick corruption probe on test loader (subset)
    from corruptions import CORRUPTION_FN

    corr_acc = {}
    model.eval()
    with torch.no_grad():
        for cname in ["clean", "rotate_45", "rotate_90", "gaussian_noise"]:
            correct, n = 0, 0
            fn = CORRUPTION_FN[cname]
            for bi, (x, y) in enumerate(test_loader):
                if bi >= 20:  # ~10k with bs512 is fine; 20*512 covers full MNIST roughly
                    break
                x, y = fn(x.to(device)), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                n += y.numel()
            corr_acc[cname] = correct / max(n, 1)

    result = {
        "tag": tag,
        "model_name": model_name,
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "use_aug": use_aug,
        "epochs": epochs,
        "best_test_acc": best_acc,
        "final_test_acc": final_acc,
        "corruption_probe": corr_acc,
        "params": count_params(model),
        "n_div": n_div,
        "act": act,
        "in_chans": in_chans,
        "latency_ms_per_batch512": latency_ms,
        "device": str(device),
        "train_seconds": elapsed,
        "history": history,
        "checkpoint": str(ckpt_path),
    }
    with open(out_dir / f"{tag}_train.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["fasternet", "baseline", "equivariant"], required=True)
    p.add_argument("--aug", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", choices=["mnist", "emnist", "svhn"], default="mnist")
    p.add_argument("--variant", choices=["mini", "t0lite"], default="mini")
    p.add_argument("--n-div", type=int, default=4)
    p.add_argument("--act", choices=["relu", "gelu"], default="relu")
    p.add_argument("--tag-suffix", type=str, default="")
    args = p.parse_args()
    run_training(
        args.model,
        args.aug,
        args.epochs,
        args.batch_size,
        args.lr,
        args.data_dir,
        args.out_dir,
        seed=args.seed,
        dataset=args.dataset,
        variant=args.variant,
        tag_suffix=args.tag_suffix,
        n_div=args.n_div,
        act=args.act,
    )


if __name__ == "__main__":
    main()
