# AI Assignment #1

**FasterNet for Robust Digit Pattern Recognition under Geometric and Noise Corruptions**

**CISC3024 Pattern Recognition**  
Name: CHE CHI HIN, Angus  
Student ID: UC325182  

> Deliverable: `AIAssignment1_Report.pdf`. Source: https://github.com/AngusJai/CISC3024-AI-Assignment1

## Abstract

**Thesis:** FasterNet (PConv) + robust aug + rotation TTA recognizes digits under corruptions far better than clean-only training, and stays competitive with a student-scale equivariant baseline while remaining easy for an AI agent to implement and verify. Report covers the required six sections. Cursor AI under my direction (no hand-written code).

## 1 How I Asked AI Tools to Find the Algorithm

I used Cursor only. I asked it to plan/clarify first, forbid copying a peer RepViT story, require a verifiable operator, and frame robust digit recognition. After thin “digit-3 / 90° only” demos, I demanded multi-digit multi-angle figures and EMNIST/SVHN. We locked FasterNet (CVPR 2023, PConv). See `process_log.md`.

## 2 Algorithm Description

FasterNet Partial Convolution: spatially convolve `1/n_div` channels (default 1/4), leave rest untouched, residual pointwise MLP. FasterNet-mini for 28×28 / 32×32; PConv identity error = 0; ablations on `n_div`, GELU; p4 equivariant baseline.

## 3 How AI Implemented the Algorithm

Agent built `models/`, `corruptions.py`, `train.py`, `evaluate.py`, `run_all.py`, `scripts/*`. I set goals and approved iterations.

## 4 Experiment Settings and Results

AdamW + cosine, batch 128, MPS; robust aug; EMNIST/SVHN 60k/10k subsample.

| Setting | Clean | Rot45 | Rot90 | Rot90+TTA | Noise |
|--------|------:|------:|------:|----------:|------:|
| MNIST FasterNet-aug | 0.987 | 0.966 | 0.887 | 0.948 | 0.891 |
| EMNIST FasterNet-aug | 0.988 | 0.968 | 0.835 | 0.941 | 0.801 |
| MNIST clean-only | 0.993 | 0.588 | 0.127 | 0.659 | 0.377 |
| MNIST Baseline-aug | 0.958 | 0.735 | 0.267 | — | 0.934 |
| SVHN 12ep | 0.789 | 0.576 | 0.572 | 0.720 | 0.641 |
| SVHN 24ep long | 0.858 | 0.711 | 0.671 | 0.792 | 0.693 |
| GELU (MNIST) | 0.988 | 0.973 | 0.866 | 0.935 | 0.857 |
| p4 Equivariant-aug | 0.964 | 0.912 | 0.823 | 0.927 | 0.831 |

**Clean-only (no rotation aug):** FasterNet Rot90 0.127 vs Equivariant Rot90 0.174 (TTA 0.659 vs 0.698)—equivariant helps a little, but aug remains essential.

Multi-seed MNIST Rot90 **84.1%±10.0%**; EMNIST Rot90 **81.2%±3.2%**. Clean ECE ≈ 0.0071.

## 5 What I Have Learnt

Prompting/constraints matter more than typing code; clean accuracy lies; architecture vs aug comparison; mechanism checks + multi-seed; EMNIST/SVHN domain shift; process logs help.

## 6 Source Code Webpage Link

https://github.com/AngusJai/CISC3024-AI-Assignment1  
Checkpoint: https://github.com/AngusJai/CISC3024-AI-Assignment1/releases/tag/v1.1-fasternet-aug (`fasternet_aug.pt`).
