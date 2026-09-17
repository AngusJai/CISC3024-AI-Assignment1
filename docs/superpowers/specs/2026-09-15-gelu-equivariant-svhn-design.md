# GELU + Equivariant Baseline + SVHN-Long Design

**Date:** 2026-09-15  
**Status:** Approved by user (“ok”)

## Goal
Strengthen the report with three literature-aligned upgrades without replacing FasterNet as the primary Deep CNN.

## Decisions
1. **Activation:** GELU in FasterNet MLP/head (FasterNet paper small-model practice). Keep ReLU checkpoint as control; new run tagged `_gelu`.
2. **Equivariant baseline:** Student-scale **p4 orbit-convolution CNN** (shared kernels at 0/90/180/270°, orientation max-pool before head). No `e2cnn` dependency. MNIST, same robust-aug protocol, 12 epochs. Greyscale only.
3. **SVHN longer training:** 24 epochs, tag `_long`, same 60k/10k subsample; evaluate full corruption table.

## Non-goals
- Replacing FasterNet backbone with equivariant net
- ImageNet-scale / full e2cnn stack
- Changing main `fasternet_aug.pt` unless GELU clearly wins (report both)

## Deliverables
- Code: `act` flag, `models/equivariant_cnn.py`, `run_upgrade_gelu_eq_svhn.py`
- Metrics JSON under `outputs/`
- Updated `report.tex` / PDF / `process_log.md` / Markdown mirror
