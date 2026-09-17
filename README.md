# CISC3024 AI Assignment #1

Robust handwritten digit recognition with **FasterNet (CVPR 2023)** — implemented entirely by an AI agent.

## Quick start

```bash
source .venv/bin/activate   # or: python -m venv .venv && pip install -r requirements.txt
python run_all.py
python demo.py --ckpt outputs/fasternet_aug.pt --corruption rotate_90
```

## Deliverables

- `AIAssignment1_Report.md` — English report (fill in Name / Student ID)
- `process_log.md` — AI workflow diary
- `outputs/` — metrics, figures, checkpoints

Paper: Chen et al., FasterNet, CVPR 2023 (arXiv:2303.03667).
