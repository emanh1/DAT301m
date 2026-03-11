# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for package management.

- Install dependencies: `uv sync`
- Run entry point: `uv run main.py`
- Add a dependency: `uv add <package>`

## Project Overview

This repository implements **SSMD** (Semi-Supervised Medical image Detector), a student-teacher detection framework for medical imaging described in `PAPER.md`. The project targets two tasks:
- **Nuclei detection** — 2018 Data Science Bowl dataset (~27,000 cells, 522 training images)
- **Lesion detection** — NIH DeepLesion dataset (32,120 CT slices, 32,735 lesions)

Datasets are stored in `dataset/` (gitignored) and are not committed to version control.

## Architecture (from PAPER.md)

SSMD uses a **student-teacher framework** where both networks share a RetinaNet backbone (ResNet-50 + FPN). Teacher weights are updated via exponential moving average (EMA) of student weights:

```
θ_t^n = α·θ_t^{n-1} + (1-α)·θ_s^n
```

Three novel components:

1. **Adaptive Consistency Cost** — down-weights background proposals by scaling consistency loss by `W(p_s, p_t) = ((1-p_s[0])² + (1-p_t[0])²) / 2`, where `p[0]` is background probability.

2. **Noisy Residual Block** — adds channel-wise attention-gated Gaussian noise to intermediate feature maps: `X^q = (X^n ⊗ sigmoid(γ·X^p)) ⊕ X^l`

3. **Instance-level Adversarial Perturbation** — applied only to teacher input; only high-confidence foreground proposals (above threshold τ) contribute gradients for computing r_adv.

**Training loss**: `loss = loss_sup + λ·loss_cont`
- `loss_sup`: CE + SmoothL1 (labeled data only)
- `loss_cont`: W ⊗ (KL + MSE) (labeled and unlabeled)
- λ ramps up in first quarter of training, stays at 1, then ramps down in last quarter

### DSB Hyperparameters
- Input: 448×448, batch size 8, 100 epochs
- Optimizer: Adam, lr=1e-5, decayed ×0.1 at epoch 75
- γ=0.9, rotation=10°

### DeepLesion Preprocessing
- Resize to 512×512, clip HU to [-1100, 1100], normalize to [-1, 1]
- Further normalize by dataset mean/std
