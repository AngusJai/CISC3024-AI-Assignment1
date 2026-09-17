#!/usr/bin/env python3
"""Run multi-seed MNIST aug, t0lite, and EMNIST experiments; aggregate stats."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs"
DATA = ROOT / "data"


def run(args: list[str]) -> None:
    print("\n>>>", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=str(ROOT))


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)


def main():
    OUT.mkdir(exist_ok=True)
    # 1) multi-seed FasterNet-aug on MNIST
    for seed in (0, 1, 2):
        run(
            [
                PY,
                "train.py",
                "--model",
                "fasternet",
                "--aug",
                "--epochs",
                "12",
                "--seed",
                str(seed),
                "--dataset",
                "mnist",
                "--variant",
                "mini",
                "--tag-suffix",
                f"_seed{seed}",
                "--data-dir",
                str(DATA),
                "--out-dir",
                str(OUT),
            ]
        )

    # 2) FasterNet-T0-lite (scaled) on MNIST
    run(
        [
            PY,
            "train.py",
            "--model",
            "fasternet",
            "--aug",
            "--epochs",
            "12",
            "--seed",
            "0",
            "--dataset",
            "mnist",
            "--variant",
            "t0lite",
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )

    # 3) EMNIST digits (more writer variability)
    run(
        [
            PY,
            "train.py",
            "--model",
            "fasternet",
            "--aug",
            "--epochs",
            "6",
            "--seed",
            "0",
            "--dataset",
            "emnist",
            "--variant",
            "mini",
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )

    # Aggregate multi-seed
    seeds = []
    for seed in (0, 1, 2):
        tag = f"fasternet_aug_seed{seed}"
        with open(OUT / f"{tag}_train.json") as f:
            seeds.append(json.load(f))

    metrics = {
        "clean": [s["corruption_probe"]["clean"] for s in seeds],
        "rotate_45": [s["corruption_probe"]["rotate_45"] for s in seeds],
        "rotate_90": [s["corruption_probe"]["rotate_90"] for s in seeds],
        "gaussian_noise": [s["corruption_probe"]["gaussian_noise"] for s in seeds],
        "final_test_acc": [s["final_test_acc"] for s in seeds],
    }
    summary = {
        k: {"mean": mean_std(v)[0], "std": mean_std(v)[1], "values": v} for k, v in metrics.items()
    }
    with open(OUT / "multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
