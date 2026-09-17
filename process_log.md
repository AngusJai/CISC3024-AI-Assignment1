# AI Assignment #1 — Process Log

**Course:** CISC3024 Pattern Recognition  
**Student:** CHE CHI HIN, Angus (UC325182)  
**Rule:** no hand-written code; AI agent (Cursor) performs search / implementation / experiments / drafting.  
**Human role:** set goals, approve direction, verify outputs, request revisions.

---

## How I directed the AI agent (personal summary)

I started by asking the agent to **plan professionally and ask clarifying questions** before coding, because the Moodle brief had no separate rubric. I locked English reporting, full robustness features (rotation/noise/blur/occlusion/reject/etc.), and told the agent **not to copy** a peer RepViT report I shared—only to learn what looks strong academically (mechanism checks, honesty about limits).

When the plan felt slow, I told the agent to **skip the one-week calendar and implement immediately**. After the first MNIST results, I asked for higher Rot90 performance and richer demos because galleries looked like “only digit 3 / only 90°,” which felt thin. I then requested **EMNIST**, more multi-angle/multi-corruption figures, and finally a **PDF polish pass (items 1–8)**: denser layout, abstract, summary table, figure interpretations, prose, architecture diagram, `n_div` ablation, and this personalized process note.

---

## Chronological technical log (agent-executed)

### Kickoff
- Brief-only constraints; choose Deep CNN + robust digits; differentiate from peer RepViT + 8×8 digits.

### Algorithm selection
- Web search → FasterNet (CVPR 2023, PConv). Rejected RepViT to avoid peer duplication.

### Implementation & MNIST
- Built `models/`, `corruptions.py`, `train.py`, `evaluate.py`, `demo.py`, `run_all.py`.
- Clean vs aug; baseline; Rot90 improved via cardinal-angle aug + TTA.
- Main MNIST FasterNet-aug: Clean 98.9%, Rot90 80.5%, Rot90+TTA 93.9%.

### Analysis pack
- `pconv_efficiency.py` (identity err=0; FLOPs/latency table)
- `analyze_extra.py` (ECE, per-class Rot90 pairs, Grad-CAM)
- Multi-seed n=3 → Rot90 84.1% ± 10.0%
- T0-lite variant; Gradio UI

### EMNIST + rich figures
- Downloaded EMNIST Digits; trained robust FasterNet-aug (60k/10k subsample).
- EMNIST: Clean 98.8%, Rot90 83.5%, TTA 94.1%.
- `make_rich_figures.py`: multi-angle (0/30/45/90/180/−45), multi-corruption, MNIST vs EMNIST bars.

### PDF polish (1–8)
- Architecture diagram `fig_architecture.png`
- `n_div` ablation `{2,4,8}` → `ndiv_ablation.json` / `fig_ndiv_ablation.png`
- Rewrote `build_report_pdf.py` for dense academic layout (abstract, tables, figure interpretations, prose)
- Synced `AIAssignment1_Report.md` to match PDF

### Upgrade items 1–5 (web-informed)
1. **LaTeX professional PDF:** `report.tex` + Tectonic → `AIAssignment1_Report.pdf` / `report.pdf`
2. **SVHN:** trained FasterNet-aug RGB mini (60k/10k); Clean 0.789, Rot90 0.572, Rot90+TTA 0.720, Noise 0.641
3. **Formal Related Work:** efficient CNNs (FasterNet/MobileOne), corruption/equivariance literature, MNIST/EMNIST/SVHN
4. **Stronger stats:** EMNIST 3-seed mean±std (Rot90 0.812±0.032); keep MNIST multi-seed 84.1%±10.0%
5. **Full `n_div` ablation:** 12-epoch robust-aug MNIST for `{2,4,8}` → `outputs/upgrade15_summary.json`; `n_div=4` best latency/clean trade-off
- Driver script: `run_upgrade_15.py`; compile helper: `compile_latex_report.py`

### Upgrade: GELU + equivariant + SVHN-long
- FasterNet `--act gelu` → Rot90 **0.866** (vs ReLU 0.805); clean 0.988
- p4 `RotEquivariantCNN` → Clean 0.964, Rot90 0.823, TTA 0.927
- SVHN 24ep `_long` → Clean **0.858** (was 0.789), Rot90+TTA **0.792** (was 0.720)
- Driver: `python run_upgrade_gelu_eq_svhn.py`; summary `outputs/upgrade_gelu_eq_svhn.json`

---

## Key commands

```bash
python run_upgrade_15.py
python run_upgrade_gelu_eq_svhn.py
./.tectonic -X compile report.tex --outdir .
python app_gradio.py
```

## Submission checklist

- [x] Recent Deep CNN (FasterNet CVPR 2023)
- [x] CV/PR application (robust digits, MNIST+EMNIST+SVHN)
- [x] AI-only coding/search/experiments
- [x] English LaTeX report PDF with Related Work + process documentation
- [x] GELU / equivariant baseline / longer SVHN
- [x] Name / Student ID filled
- [ ] Upload `AIAssignment1_Report.pdf` (+ optional code) to UMMoodle
