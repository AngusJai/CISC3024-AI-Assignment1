# AI Assignment #1

**FasterNet for Robust Digit Pattern Recognition under Geometric and Noise Corruptions**

**CISC3024 Pattern Recognition**  
Name: CHE CHI HIN, Angus  
Student ID: UC325182  

> Deliverable: `AIAssignment1_Report.pdf`. This Markdown mirrors the required six sections.

## Abstract

Report structure: (1) how AI found the algorithm, (2) algorithm description, (3) how AI implemented it, (4) experiment settings/results, (5) learning outcomes, (6) source-code link. Selected Deep CNN: **FasterNet (CVPR 2023)** with PConv for robust digit recognition on MNIST / EMNIST Digits / SVHN. All work by Cursor AI agent under human direction.

## 1 How I Asked AI Tools to Find the Algorithm

Used **Cursor** only. Prompts asked for a recent (2023–2024) Deep CNN/Autoencoder for CV/PR; avoid peer RepViT duplication; prefer a verifiable operator; frame as robust digit recognition. Agent searched and we locked **FasterNet** (PConv). Diary: `process_log.md`.

## 2 Algorithm Description

FasterNet / Partial Convolution: spatially convolve `1/n_div` channels (default `r=1/4`), leave rest untouched, then residual pointwise MLP. Adapted FasterNet-mini for 28×28 / 32×32 digits; verified PConv identity error = 0 and FLOPs; ablated `n_div`, GELU; compared p4 equivariant baseline.

## 3 How AI Implemented the Algorithm

Agent wrote `models/`, `corruptions.py`, `train.py`, `evaluate.py`, analysis/drivers, Gradio demo, LaTeX report. Human set goals and approved iterations; no hand-written student code.

## 4 Experiment Settings and Results

AdamW + cosine, batch 128, MPS; robust aug; EMNIST/SVHN 60k/10k subsample.

| Setting | Clean | Rot45 | Rot90 | Rot90+TTA | Noise |
|--------|------:|------:|------:|----------:|------:|
| MNIST FasterNet-aug | 0.989 | 0.965 | 0.805 | 0.939 | 0.940 |
| EMNIST FasterNet-aug | 0.988 | 0.968 | 0.835 | 0.941 | 0.801 |
| MNIST clean-only | 0.992 | 0.548 | 0.130 | — | 0.425 |
| MNIST Baseline-aug | 0.958 | 0.735 | 0.267 | — | 0.934 |
| SVHN 12ep | 0.789 | 0.576 | 0.572 | 0.720 | 0.641 |
| SVHN 24ep long | 0.858 | 0.711 | 0.671 | 0.792 | 0.693 |
| GELU (MNIST) | 0.988 | 0.973 | 0.866 | 0.935 | 0.857 |
| p4 Equivariant-aug | 0.964 | 0.912 | 0.823 | 0.927 | 0.831 |

Multi-seed MNIST Rot90 **84.1%±10.0%**; EMNIST Rot90 **81.2%±3.2%**.

## 5 What I Have Learnt

Directing AI with clear constraints; clean accuracy insufficient under corruptions; mechanism checks + multi-seed stats; equivariant baselines vs aug+TTA; EMNIST/SVHN harder than MNIST; process logging matters.

## 6 Source Code Webpage Link

**https://github.com/AngusJai/CISC3024-AI-Assignment1**
