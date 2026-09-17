# CISC3024 AI Assignment #1

**FasterNet for Robust Digit Pattern Recognition**  
CHE CHI HIN, Angus · UC325182

Source code: **https://github.com/AngusJai/CISC3024-AI-Assignment1**

Implemented entirely by a Cursor AI agent (no hand-written student code).

## Report (required 6 parts)

1. How AI was asked to find the algorithm  
2. Algorithm description (FasterNet / PConv)  
3. How AI implemented the algorithm  
4. Experiment settings and results  
5. What was learnt  
6. Source-code webpage link (this repo)

See `AIAssignment1_Report.pdf` / `report.tex` and `process_log.md`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
python demo.py --ckpt outputs/fasternet_aug.pt --corruption rotate_90
```

Paper: Chen et al., FasterNet, CVPR 2023 (arXiv:2303.03667).
