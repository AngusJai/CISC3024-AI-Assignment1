# AI Assignment #1

**FasterNet for Robust Digit Pattern Recognition under Geometric and Noise Corruptions**

**CISC3024 Pattern Recognition**  
Name: CHE CHI HIN, Angus  
Student ID: UC325182  

> Canonical deliverable: `AIAssignment1_Report.pdf` (compiled from `report.tex` via Tectonic). This Markdown mirrors the PDF.

## Abstract

This report presents an AI-agent-driven study of **FasterNet (CVPR 2023)**, a recent Deep CNN based on Partial Convolution (PConv). We apply it to **robust handwritten and scene digit recognition** (0–9) under rotation, noise, blur, shift, occlusion and elastic distortion. Beyond upright MNIST, we evaluate writer-varied **EMNIST Digits** and real-world **SVHN** house-number digits. Clean-only training yields ~99% upright accuracy but collapses under 90° rotation (~13%); robust augmentation recovers Rot90 to ~80–84% (TTA ~94%) on handwriting benchmarks. We verify PConv mechanistically, ablate `n_div ∈ {2,4,8}` aligned with the paper’s partial-ratio study, report multi-seed mean±std, ECE, and confusable pairs. All search, coding and drafting were performed by an AI agent (Cursor) under human direction.

## 1 Task

Recent Deep CNN/Autoencoder for CV/PR; AI agent for search/coding/writing. Application: robust digit recognition across MNIST, EMNIST Digits, and SVHN.

## 2 AI-Assisted Workflow

Human set goals and approved iterations; Cursor agent performed search, implementation, experiments, figures, and drafting. See `process_log.md`.

## 3 Related Work

- **Efficient CNNs:** FasterNet (PConv, default `r=1/4`); MobileOne (re-parameterization—different mechanism; not duplicated from peer fusion-style reports).
- **Robustness:** Hendrycks & Dietterich corruption benchmarks; rotation via (i) augmentation/TTA (this work) vs (ii) group-equivariant / invariant-integration networks (Rath et al.).
- **Digits:** MNIST, EMNIST Digits, SVHN (scene digits).

## 4 Method: FasterNet-mini

Adapted to 28×28 grayscale (MNIST/EMNIST) and 32×32 RGB (SVHN). PConv verification: untouched-channel error = 0; ~6.25% of dense 3×3 FLOPs at `n_div=4`. Main act = ReLU; **GELU** ablation; **p4 equivariant** orbit-CNN baseline.

### Full 12-epoch `n_div` ablation (robust-aug MNIST)

| n_div | Params | Clean | Rot45 | Rot90 | Noise | Lat. ms/512 |
|------:|-------:|------:|------:|------:|------:|------------:|
| 2 | 342k | 0.988 | 0.967 | 0.834 | 0.935 | 14.5 |
| 4 | 271k | 0.987 | 0.966 | 0.887 | 0.885 | 12.1 |
| 8 | 253k | 0.977 | 0.965 | 0.930 | 0.838 | 15.5 |

Default `n_div=4` remains the best latency–accuracy compromise (matches FasterNet paper).

## 5 Experiments & Results

| Setting | Clean | Rot45 | Rot90 | Rot90+TTA | Noise |
|--------|------:|------:|------:|----------:|------:|
| MNIST FasterNet-aug | 0.989 | 0.965 | 0.805 | 0.939 | 0.940 |
| EMNIST FasterNet-aug | 0.988 | 0.968 | 0.835 | 0.941 | 0.801 |
| MNIST clean-only | 0.992 | 0.548 | 0.130 | — | 0.425 |
| MNIST Baseline-aug | 0.958 | 0.735 | 0.267 | — | 0.934 |
| **SVHN 12ep** | **0.789** | **0.576** | **0.572** | **0.720** | **0.641** |
| **SVHN 24ep long** | **0.858** | **0.711** | **0.671** | **0.792** | **0.693** |
| FasterNet GELU (MNIST) | 0.988 | 0.973 | 0.866 | 0.935 | 0.857 |
| p4 Equivariant-aug | 0.964 | 0.912 | 0.823 | 0.927 | 0.831 |

- Multi-seed MNIST Rot90: **84.1% ± 10.0%**
- EMNIST Digits multi-seed (n=3): Clean **0.981±0.007**, Rot90 **0.812±0.032**, Noise **0.817±0.159**
- Clean ECE ≈ 0.0076; hardest Rot90 pair **6↔9**
- EMNIST/SVHN use 60k/10k student-scale subsamples

## 6 Discussion

GELU improves Rot90 vs ReLU; p4 equivariant is a strong Rot90 baseline but trails FasterNet on clean/noise. SVHN 24ep closes much of the under-training gap (clean 0.789→0.858).

## 7 Conclusion

AI agent end-to-end searched, implemented and evaluated FasterNet for robust digit recognition across MNIST, EMNIST Digits and SVHN, with mechanism checks, ablations and multi-seed statistics.

## References

1. Chen et al. (2023). FasterNet. CVPR.  
2. Vasu et al. (2023). MobileOne. CVPR.  
3. Hendrycks & Dietterich (2019). ICLR.  
4. LeCun et al. (1998). Proc. IEEE.  
5. Cohen et al. (2017). EMNIST.  
6. Netzer et al. (2011). SVHN.  
7. Rath & Condurache (2023). AISTATS; (2022) arXiv:2202.03967.

## Appendix

```bash
python run_upgrade_15.py
python run_upgrade_gelu_eq_svhn.py
./.tectonic -X compile report.tex
python app_gradio.py
```
