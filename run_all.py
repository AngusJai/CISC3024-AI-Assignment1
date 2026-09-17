#!/usr/bin/env python3
"""End-to-end pipeline: train all variants, evaluate, write comparison tables."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs"
DATA = ROOT / "data"


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)

    jobs = [
        # FasterNet clean + aug
        ["train.py", "--model", "fasternet", "--epochs", "8"],
        ["train.py", "--model", "fasternet", "--aug", "--epochs", "8"],
        # Baseline clean + aug
        ["train.py", "--model", "baseline", "--epochs", "8"],
        ["train.py", "--model", "baseline", "--aug", "--epochs", "8"],
    ]
    for args in jobs:
        run([PY, *args, "--data-dir", str(DATA), "--out-dir", str(OUT)])

    ckpts = [
        OUT / "fasternet_clean.pt",
        OUT / "fasternet_aug.pt",
        OUT / "baseline_clean.pt",
        OUT / "baseline_aug.pt",
    ]
    for ckpt in ckpts:
        run([PY, "evaluate.py", "--ckpt", str(ckpt), "--data-dir", str(DATA), "--out-dir", str(OUT)])

    # Build comparison markdown/json
    rows = []
    for ckpt in ckpts:
        tag = ckpt.stem
        with open(OUT / f"{tag}_eval.json") as f:
            ev = json.load(f)
        with open(OUT / f"{tag}_train.json") as f:
            tr = json.load(f)
        ct = ev["corruption_table"]
        rows.append(
            {
                "tag": tag,
                "params": ev["params"],
                "latency_ms": tr.get("latency_ms_per_batch512"),
                "clean": ct["clean"]["accuracy"],
                "rotate_45": ct["rotate_45"]["accuracy"],
                "rotate_90": ct["rotate_90"]["accuracy"],
                "gaussian_noise": ct["gaussian_noise"]["accuracy"],
                "blur": ct["blur"]["accuracy"],
                "shift": ct["shift"]["accuracy"],
                "occlusion": ct["occlusion"]["accuracy"],
                "elastic": ct["elastic"]["accuracy"],
                "reject": ev["reject_operating_point"],
            }
        )

    with open(OUT / "comparison.json", "w") as f:
        json.dump(rows, f, indent=2)

    # markdown table
    headers = [
        "model",
        "clean",
        "rot45",
        "rot90",
        "noise",
        "blur",
        "shift",
        "occ",
        "elastic",
        "params",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["tag"],
                    f"{r['clean']:.3f}",
                    f"{r['rotate_45']:.3f}",
                    f"{r['rotate_90']:.3f}",
                    f"{r['gaussian_noise']:.3f}",
                    f"{r['blur']:.3f}",
                    f"{r['shift']:.3f}",
                    f"{r['occlusion']:.3f}",
                    f"{r['elastic']:.3f}",
                    str(r["params"]),
                ]
            )
            + " |"
        )
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n=== COMPARISON ===")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
