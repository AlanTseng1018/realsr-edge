# RealSR-Edge

A pre-handoff quantization analysis & optimization pipeline for super-resolution on edge AI accelerators. EDSR-baseline is used as the substrate; the deliverable is a decision package (recipes + reasoning + scope boundary) that lets a downstream NPU/SoC team adapt the work to their silicon, not a single tuned config.

The project is structured as four investigation tracks layered on a single trained model:

1. **Model training** — EDSR-baseline on DIV2K with realistic degradation (this section)
2. **Quantization recipes** — PTQ vs QAT, calibration ablation, per-layer sensitivity, mixed-precision sweep
3. **Cross-stack deployment** — ONNX × 3 EP (CPU / CUDA / TensorRT), Native TRT, C++ ORT runner, roofline analysis
4. **Findings & scope** — QDQ paradox, perceptual triangulation (PSNR/SSIM/LPIPS), HW utilization, honest verified / hypothesized / cannot-verify boundary

This README currently covers section 1; subsequent sections will be added incrementally.

---

## 1. Model Training

### 1.1 Setup

| Item | Value |
|---|---|
| Backbone | EDSR-baseline ([Lim et al., CVPRW 2017](https://arxiv.org/abs/1707.02921)) |
| Residual blocks × feature width | 16 × 64 |
| Upsample scale | ×2 |
| Parameters | ~1.37 M |
| Loss | L1 (rationale in [docs/adr/004_loss_function_choice.md](docs/adr/004_loss_function_choice.md)) |
| Optimizer | Adam, lr = 1e-4 |
| LR schedule | StepLR, step = 100 epochs, γ = 0.5 |
| Epochs | 200 |
| Batch / patch (LR) | 16 / 96 (HR patch = 192) |
| Dataset | DIV2K — 800 train HR / 100 val HR |
| Degradation | Realistic — Gaussian blur (σ 0.1–2.0) → bicubic ↓×2 → AWGN (σ 0–25) → JPEG (Q 60–95) → optional banding (4–6 bit) |
| Hardware | Single RTX 3090 24GB, ~3–4 GB VRAM at this config |
| Framework | PyTorch 2.x, optional `torch.compile` |

The realistic degradation pipeline is intentionally TV-content-leaning rather than the academic bicubic-only setup; details and per-step parameter ranges are in [src/data/degradation.py](src/data/degradation.py).

### 1.2 Reproduce

```bash
# Default Track B: realistic degradation, 200 epochs, batch 16
python -m src.training.train --compile --compile-mode default

# Quick smoke test (2 epochs, batch 2) — verifies the loop end-to-end
python -m src.training.train --quick

# Resume from a checkpoint
python -m src.training.train --resume results/runs/<run_id>/checkpoints/best.pt
```

Each run writes a timestamped folder under `results/runs/`:

```
results/runs/<YYYYMMDD_HHMMSS>_ep200_b16_scale2_realistic/
├── checkpoints/         # best.pt, periodic epoch_NNN.pt, final.pt
├── val_samples/         # LR | Bicubic | SR | HR comparison panels
├── metrics.csv          # epoch, train_loss, val_psnr_db, val_ssim
└── curves.png           # loss / PSNR / SSIM curves
```

### 1.3 Training Curve

![training curves](results/runs/20260427_143542_ep200_b16_scale2_realistic/curves.png)

Source: [results/runs/20260427_143542_ep200_b16_scale2_realistic/curves.png](results/runs/20260427_143542_ep200_b16_scale2_realistic/curves.png) — full per-epoch metrics in [metrics.csv](results/runs/20260427_143542_ep200_b16_scale2_realistic/metrics.csv).

### 1.4 Result

| Checkpoint | Val PSNR (dB) | Val SSIM |
|---|---|---|
| Bicubic baseline (val set) | ~25.6 | ~0.72 |
| EDSR-baseline FP32, epoch 200 | **27.44** | **0.7907** |

PSNR plateaus around epoch 150 with a long, low-amplitude tail to epoch 200; the StepLR drop at epoch 100 is visible as a tightening in the loss curve. The model is deliberately *just-strong-enough* — it is the substrate for downstream quantization stress, not a SOTA SR submission. Choosing a higher-capacity backbone would have made the INT8 / mixed-precision findings less informative because there would be more headroom to absorb quantization noise.

### 1.5 Key Scripts

| File | Role |
|---|---|
| [src/training/train.py](src/training/train.py) | Training entry point — argparse config, FP32 loop, validation, checkpointing, optional QAT phase |
| [src/data/dataset.py](src/data/dataset.py) | `SRDataset` — DIV2K HR loader with on-the-fly LR generation |
| [src/data/degradation.py](src/data/degradation.py) | `RealisticDegradation` — composable blur / noise / JPEG / banding / down-sample |
| [src/models/edsr.py](src/models/edsr.py) | EDSR-baseline model definition |
| [src/models/common.py](src/models/common.py) | Shared building blocks (residual block, upsampler) |
| [docs/adr/004_loss_function_choice.md](docs/adr/004_loss_function_choice.md) | Loss-function ADR — why L1 over L2 / Charbonnier / perceptual |

QAT fine-tuning is implemented in the same `train.py` (`--qat` flag, calibration → STE fine-tune → fake-INT8 best checkpoint) and is covered in section 2.
