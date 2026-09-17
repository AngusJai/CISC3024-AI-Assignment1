# CISC3024 AI Assignment #1

**FasterNet for Robust Digit Pattern Recognition**  
CHE CHI HIN, Angus · UC325182

**Report (submit):** [`AIAssignment1_Report.pdf`](./AIAssignment1_Report.pdf)  
**Source:** https://github.com/AngusJai/CISC3024-AI-Assignment1

## Layout

```
├── AIAssignment1_Report.pdf   # submit this
├── report.tex / process_log.md
├── models/                    # FasterNet, baseline, equivariant
├── train.py / evaluate.py / corruptions.py / demo.py / run_all.py
├── scripts/                   # extras & demos
│   ├── run_upgrade_*.py
│   ├── analyze_extra.py / make_rich_figures.py / pconv_efficiency.py
│   └── app_gradio.py
└── outputs/                   # figures + metrics JSON
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py                          # train + eval MNIST variants
python demo.py --ckpt outputs/fasternet_aug.pt --corruption rotate_90
python scripts/app_gradio.py               # optional UI (needs checkpoint)
```

Paper: Chen et al., FasterNet, CVPR 2023 (arXiv:2303.03667).
