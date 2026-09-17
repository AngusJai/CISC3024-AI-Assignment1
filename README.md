# CISC3024 AI Assignment #1

**FasterNet for Robust Digit Pattern Recognition**  
CHE CHI HIN, Angus · UC325182

**Report (submit):** [`AIAssignment1_Report.pdf`](./AIAssignment1_Report.pdf)  
**Source:** https://github.com/AngusJai/CISC3024-AI-Assignment1  
**Checkpoint release:** https://github.com/AngusJai/CISC3024-AI-Assignment1/releases/tag/v1.1-fasternet-aug (`fasternet_aug.pt`)

## Layout

```
├── AIAssignment1_Report.pdf
├── report.tex / process_log.md
├── models/
├── train.py / evaluate.py / corruptions.py / demo.py / run_all.py
├── scripts/          # upgrades, analysis, Gradio
└── outputs/          # figures + metrics JSON
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
# or download fasternet_aug.pt from Releases into outputs/
python demo.py --ckpt outputs/fasternet_aug.pt --corruption rotate_90
python scripts/app_gradio.py
```

## Expected results (approx.)

| Setting | Clean | Rot90 | Rot90+TTA |
|--------|------:|------:|----------:|
| FasterNet-aug (MNIST) | ~0.99 | ~0.89 | ~0.95 |
| FasterNet-clean | ~0.99 | ~0.13 | ~0.66 |
| Equivariant-clean | ~0.99 | ~0.17 | ~0.70 |

Paper: Chen et al., FasterNet, CVPR 2023 (arXiv:2303.03667).
