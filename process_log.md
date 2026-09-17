# AI Assignment #1 — Process Log

**Course:** CISC3024 Pattern Recognition  
**Student:** CHE CHI HIN, Angus (UC325182)  
**Rule:** no hand-written code; AI agent (Cursor) performs search / implementation / experiments / drafting.  
**Human role:** set goals, approve direction, verify outputs, request revisions.

---

## How I directed the AI agent (personal summary)

I started by asking the agent to **plan professionally and ask clarifying questions** before coding. I locked English reporting, full robustness features, and told the agent **not to copy** a peer RepViT report—only to learn what looks strong academically. Later I requested EMNIST/SVHN, richer figures, LaTeX polish, GELU/equivariant/SVHN-long upgrades, and a cleaner repo layout (`scripts/`).

---

## Chronological technical log (summary)

- Selected **FasterNet** (CVPR 2023, PConv); avoided peer RepViT narrative.
- Built core pipeline: `models/`, `corruptions.py`, `train.py`, `evaluate.py`, `run_all.py`.
- MNIST FasterNet-aug (retrained): Clean 98.7%, Rot90 88.7%, Rot90+TTA 94.8%.
- Clean-only vs equivariant: FasterNet-clean Rot90 12.7% / TTA 65.9%; p4 Equivariant-clean Rot90 17.4% / TTA 69.8% — equivariance helps slightly, aug still essential.
- Analysis/figures via `scripts/pconv_efficiency.py`, `scripts/analyze_extra.py`, `scripts/make_rich_figures.py`.
- EMNIST + SVHN; multi-seed; `n_div` ablation; GELU; p4 equivariant; SVHN 24-epoch long run.
- Report: `report.tex` → `AIAssignment1_Report.pdf` (required 6 sections).
- Repo cleanup: obsolete tools removed; helpers under `scripts/`; `outputs/` keeps report assets only.
- Checkpoint `fasternet_aug.pt` attached as GitHub Release asset.

## Key commands

```bash
python run_all.py
python scripts/run_upgrade_15.py
python scripts/run_upgrade_gelu_eq_svhn.py
python scripts/app_gradio.py   # needs checkpoint from train
```

## Submission checklist

- [x] Recent Deep CNN (FasterNet CVPR 2023)
- [x] CV/PR application (robust digits, MNIST+EMNIST+SVHN)
- [x] AI-only coding/search/experiments
- [x] English LaTeX report PDF with required 6 sections
- [x] Name: CHE CHI HIN, Angus / ID: UC325182
- [x] Source: https://github.com/AngusJai/CISC3024-AI-Assignment1
- [ ] Upload PDF (+ GitHub link) to UMMoodle
