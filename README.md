# CISC3024 AI Assignment #1

**FasterNet for Robust Digit Pattern Recognition**  
CHE CHI HIN, Angus · UC325182

**Source code:** https://github.com/AngusJai/CISC3024-AI-Assignment1  
**Report:** `AIAssignment1_Report.pdf` (from `report.tex`)

## Folder layout

| Path | Purpose |
|------|---------|
| `AIAssignment1_Report.pdf` | Submit this PDF |
| `report.tex` | LaTeX source |
| `process_log.md` | AI workflow diary |
| `models/` | FasterNet, baseline, equivariant |
| `train.py` / `evaluate.py` / `corruptions.py` | Core pipeline |
| `run_all.py` | Main MNIST train+eval |
| `run_upgrade_15.py` / `run_upgrade_gelu_eq_svhn.py` | Extra experiments |
| `outputs/` | Report figures + key metrics JSON only |
| `app_gradio.py` / `demo.py` | Optional demos |
| `data/` | Datasets (not in git; auto-downloaded) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Paper: Chen et al., FasterNet, CVPR 2023 (arXiv:2303.03667).
