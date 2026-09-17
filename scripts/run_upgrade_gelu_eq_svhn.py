#!/usr/bin/env python3
"""Train GELU FasterNet, p4 equivariant baseline, and long SVHN; aggregate JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
DATA = ROOT / "data"


def run(args: list[str]) -> None:
    print("\n>>>", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=str(ROOT))


def flat_table(path: Path) -> dict:
    data = json.loads(path.read_text())
    table = data.get("corruption_table", data)
    out = {}
    for k in ("clean", "rotate_45", "rotate_90", "rotate_90_tta", "gaussian_noise"):
        if k not in table:
            continue
        v = table[k]
        out[k] = float(v["accuracy"] if isinstance(v, dict) else v)
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # 1) FasterNet + GELU on MNIST (12 ep, do not overwrite main ReLU ckpt)
    run(
        [
            PY,
            "train.py",
            "--model",
            "fasternet",
            "--aug",
            "--epochs",
            "12",
            "--act",
            "gelu",
            "--seed",
            "0",
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )
    run(
        [
            PY,
            "evaluate.py",
            "--ckpt",
            str(OUT / "fasternet_aug_gelu.pt"),
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )

    # 2) Equivariant baseline on MNIST
    run(
        [
            PY,
            "train.py",
            "--model",
            "equivariant",
            "--aug",
            "--epochs",
            "12",
            "--seed",
            "0",
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )
    run(
        [
            PY,
            "evaluate.py",
            "--ckpt",
            str(OUT / "equivariant_aug.pt"),
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )

    # 3) SVHN longer training
    run(
        [
            PY,
            "train.py",
            "--model",
            "fasternet",
            "--aug",
            "--epochs",
            "24",
            "--seed",
            "0",
            "--dataset",
            "svhn",
            "--tag-suffix",
            "_long",
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )
    run(
        [
            PY,
            "evaluate.py",
            "--ckpt",
            str(OUT / "fasternet_aug_svhn_long.pt"),
            "--data-dir",
            str(DATA),
            "--out-dir",
            str(OUT),
        ]
    )

    summary = {
        "fasternet_relu_ref": flat_table(OUT / "fasternet_aug_eval.json")
        if (OUT / "fasternet_aug_eval.json").exists()
        else None,
        "fasternet_gelu": flat_table(OUT / "fasternet_aug_gelu_eval.json"),
        "equivariant_aug": flat_table(OUT / "equivariant_aug_eval.json"),
        "svhn_12ep": flat_table(OUT / "fasternet_aug_svhn_eval.json")
        if (OUT / "fasternet_aug_svhn_eval.json").exists()
        else None,
        "svhn_24ep_long": flat_table(OUT / "fasternet_aug_svhn_long_eval.json"),
    }
    (OUT / "upgrade_gelu_eq_svhn.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
