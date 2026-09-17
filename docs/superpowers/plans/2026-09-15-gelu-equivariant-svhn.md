# GELU + Equivariant + SVHN-Long Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship GELU FasterNet, p4 equivariant baseline, and 24-epoch SVHN; refresh LaTeX PDF.

**Files:**
- Modify: `models/fasternet_mini.py`, `models/__init__.py`, `train.py`, `evaluate.py`, `report.tex`, `AIAssignment1_Report.md`, `process_log.md`
- Create: `models/equivariant_cnn.py`, `run_upgrade_gelu_eq_svhn.py`

## Task 1: GELU in FasterNet
- Add `act: str = "relu"` → `nn.ReLU` / `nn.GELU` in MLPBlock + head
- Persist `act` in checkpoint; rebuild in `evaluate.load_model`
- CLI `--act {relu,gelu}`; auto tag `_gelu` when gelu

## Task 2: Equivariant CNN
- Implement orbit conv (rotate shared weight ×4, concat) + orientation max-pool head
- `--model equivariant` in train/eval

## Task 3: Runner
- Train: FasterNet-aug GELU 12ep MNIST; equivariant-aug 12ep MNIST; FasterNet-aug SVHN 24ep `_long`
- Evaluate all three; write `outputs/upgrade_gelu_eq_svhn.json`

## Task 4: Report
- Tables + discussion; compile with `./.tectonic`
