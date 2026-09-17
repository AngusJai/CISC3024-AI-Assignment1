#!/usr/bin/env python3
"""Run SVHN, EMNIST multi-seed, and full n_div ablation; aggregate JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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

    # 1) Full-ish n_div ablation on MNIST (12 epochs, full train)
    # Always use tag-suffix so we never overwrite the main fasternet_aug.pt checkpoint.
    for n_div in (2, 4, 8):
        run(
            [
                PY,
                "train.py",
                "--model",
                "fasternet",
                "--aug",
                "--epochs",
                "12",
                "--n-div",
                str(n_div),
                "--seed",
                "0",
                "--dataset",
                "mnist",
                "--tag-suffix",
                "_full",
                "--data-dir",
                str(DATA),
                "--out-dir",
                str(OUT),
            ]
        )

    # 2) EMNIST multi-seed
    for seed in (0, 1, 2):
        # seed0 already exists as fasternet_aug_emnist.pt - still retrain tagged seeds for clean stats
        run(
            [
                PY,
                "train.py",
                "--model",
                "fasternet",
                "--aug",
                "--epochs",
                "10",
                "--seed",
                str(seed),
                "--dataset",
                "emnist",
                "--tag-suffix",
                f"_seed{seed}",
                "--data-dir",
                str(DATA),
                "--out-dir",
                str(OUT),
            ]
        )

    # 3) SVHN scene digits
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
            "svhn",
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )
    run([PY, "evaluate.py", "--ckpt", str(OUT / "fasternet_aug_svhn.pt"), "--data-dir", str(DATA), "--out-dir", str(OUT)])

    # Aggregate
    ndiv_rows = []
    for n_div in (2, 4, 8):
        if n_div == 4:
            path = OUT / "fasternet_aug_full_train.json"
        else:
            path = OUT / f"fasternet_aug_ndiv{n_div}_full_train.json"
        tr = json.loads(path.read_text())
        ndiv_rows.append(
            {
                "n_div": n_div,
                "params": tr["params"],
                "clean": tr["corruption_probe"]["clean"],
                "rotate_45": tr["corruption_probe"]["rotate_45"],
                "rotate_90": tr["corruption_probe"]["rotate_90"],
                "gaussian_noise": tr["corruption_probe"]["gaussian_noise"],
                "final_test_acc": tr["final_test_acc"],
                "tag": tr["tag"],
            }
        )

    em_seeds = []
    for seed in (0, 1, 2):
        tr = json.loads((OUT / f"fasternet_aug_emnist_seed{seed}_train.json").read_text())
        em_seeds.append(tr)

    def pack(keys):
        out = {}
        for k in keys:
            vals = [s["corruption_probe"][k] for s in em_seeds]
            m, sd = mean_std(vals)
            out[k] = {"mean": m, "std": sd, "values": vals}
        return out

    svhn_eval = None
    if (OUT / "fasternet_aug_svhn_eval.json").exists():
        svhn_eval = json.loads((OUT / "fasternet_aug_svhn_eval.json").read_text())["corruption_table"]

    summary = {
        "ndiv_full12ep": ndiv_rows,
        "emnist_multiseed": pack(["clean", "rotate_45", "rotate_90", "gaussian_noise"]),
        "svhn": svhn_eval,
        "literature_note": {
            "fasternet_default_r": "1/4 (n_div=4) best accuracy/throughput on ImageNet (Chen et al. CVPR 2023)",
            "robustness_approaches": "augmentation+TTA (this work) vs group-equivariant/invariant-integration CNNs (Rath et al.; Rot-MNIST/SVHN literature)",
        },
    }
    (OUT / "upgrade15_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
