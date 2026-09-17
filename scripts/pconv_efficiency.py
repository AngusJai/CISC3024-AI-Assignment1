"""PConv mechanism / efficiency verification (analogous to a fusion check).

Compares PartialConv3 against depthwise 3x3 and full dense 3x3 on the same
activation shape: FLOPs estimate, parameter count, and wall-clock latency.
Also checks that PConv leaves untouched channels unchanged (exact identity).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from models.fasternet_mini import PartialConv3


def flops_conv2d(cin, cout, k, h, w, groups=1) -> int:
    # MAC count ≈ cout * (cin/groups) * k * k * h * w
    return cout * (cin // groups) * k * k * h * w


@torch.no_grad()
def bench(module: nn.Module, x: torch.Tensor, reps: int = 200) -> float:
    module.eval()
    for _ in range(20):
        module(x)
    if x.device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        module(x)
    if x.device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dim, h, w, n_div = 128, 14, 14, 4
    x = torch.randn(64, dim, h, w, device=device)

    pconv = PartialConv3(dim, n_div=n_div).to(device)
    dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False).to(device)
    dense = nn.Conv2d(dim, dim, 3, padding=1, bias=False).to(device)

    # Identity check: untouched channels must equal input
    y = pconv(x)
    untouched = dim - pconv.dim_conv
    max_untouched_err = (y[:, pconv.dim_conv :] - x[:, pconv.dim_conv :]).abs().max().item()

    rows = []
    for name, mod, flops in [
        (
            "PConv (n_div=4)",
            pconv,
            flops_conv2d(pconv.dim_conv, pconv.dim_conv, 3, h, w, groups=1) * x.size(0),
        ),
        (
            "Depthwise 3x3",
            dw,
            flops_conv2d(dim, dim, 3, h, w, groups=dim) * x.size(0),
        ),
        (
            "Dense 3x3",
            dense,
            flops_conv2d(dim, dim, 3, h, w, groups=1) * x.size(0),
        ),
    ]:
        params = sum(p.numel() for p in mod.parameters())
        ms = bench(mod, x)
        rows.append(
            {
                "op": name,
                "params": params,
                "flops_per_batch64": flops,
                "latency_ms": ms,
                "relative_flops_vs_dense": flops
                / (flops_conv2d(dim, dim, 3, h, w, 1) * x.size(0)),
            }
        )

    result = {
        "setting": {"B": 64, "C": dim, "H": h, "W": w, "n_div": n_div, "device": str(device)},
        "pconv_untouched_channel_max_abs_err": max_untouched_err,
        "operators": rows,
    }
    path = out_dir / "pconv_efficiency.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
