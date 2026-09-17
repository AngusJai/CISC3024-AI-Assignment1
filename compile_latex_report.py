#!/usr/bin/env python3
"""Fill SVHN row in report.tex from upgrade15_summary.json and compile with tectonic."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TEX = ROOT / "report.tex"
TECTONIC = Path("/tmp/tectonic")


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    summary_path = OUT / "upgrade15_summary.json"
    if not summary_path.exists():
        summary_path = OUT / "upgrade15_partial.json"
    summary = json.loads(summary_path.read_text())

    tex = TEX.read_text()

    # Fill SVHN table if present
    svhn = summary.get("svhn")
    if isinstance(svhn, dict):
        # corruption_table may be list of dicts or nested dict
        def extract(table: dict, name: str) -> float | None:
            if name not in table:
                return None
            v = table[name]
            if isinstance(v, dict):
                for k in ("accuracy", "acc"):
                    if k in v:
                        return float(v[k])
                return None
            if isinstance(v, (int, float)):
                return float(v)
            return None

        table = svhn.get("corruption_table", svhn) if isinstance(svhn, dict) else {}
        if not isinstance(table, dict):
            table = {}
        keys = {
            "clean": extract(table, "clean"),
            "rotate_45": extract(table, "rotate_45"),
            "rotate_90": extract(table, "rotate_90"),
            "rotate_90_tta": extract(table, "rotate_90_tta"),
            "gaussian_noise": extract(table, "gaussian_noise"),
        }
        ev = OUT / "fasternet_aug_svhn_eval.json"
        if ev.exists() and any(v is None for v in keys.values()):
            data = json.loads(ev.read_text())
            table = data.get("corruption_table", data)
            if isinstance(table, dict):
                for n in list(keys):
                    if keys[n] is None:
                        keys[n] = extract(table, n)

        if all(v is not None for v in keys.values()):
            row = (
                f"{fmt(keys['clean'])} & {fmt(keys['rotate_45'])} & "
                f"{fmt(keys['rotate_90'])} & {fmt(keys['rotate_90_tta'])} & "
                f"{fmt(keys['gaussian_noise'])} \\\\"
            )
            tex = re.sub(
                r"(\\label\{tab:svhn\}.*?\\midrule\n)(.*?)(\\bottomrule)",
                r"\1" + row + "\n" + r"\3",
                tex,
                count=1,
                flags=re.S,
            )
            TEX.write_text(tex)
            print("Filled SVHN table:", row)
        else:
            print("SVHN keys incomplete:", keys, file=sys.stderr)
    else:
        print("No SVHN results yet; compiling with placeholders.", file=sys.stderr)

    engine = TECTONIC if TECTONIC.exists() else shutil.which("tectonic")
    if not engine:
        raise SystemExit("tectonic not found at /tmp/tectonic or PATH")

    subprocess.check_call(
        [str(engine), "-X", "compile", str(TEX), "--outdir", str(ROOT)],
        cwd=str(ROOT),
    )
    src = ROOT / "report.pdf"
    dst = ROOT / "AIAssignment1_Report.pdf"
    if src.exists():
        shutil.copy2(src, dst)
        print("Wrote", dst)
    else:
        raise SystemExit("report.pdf missing after tectonic")


if __name__ == "__main__":
    main()
