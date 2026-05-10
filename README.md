# RealSR-Quant

A pre-handoff quantization analysis & optimization pipeline for super-resolution. EDSR-baseline is used as the substrate; the deliverable is a decision package (recipes + reasoning + scope boundary) that lets a downstream NPU/SoC team adapt the work to their silicon, not a single tuned config. All measurements are on a workstation GPU (RTX 3090) and ONNX/TRT — real edge-device verification is explicitly out of scope and marked as such throughout.

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

---

## 2. Optimization Recipes

### 2.1 Framing — decision support, not method demo

The output of this section is **not** "I implemented N quantization methods." It is a **decision package** for a downstream NPU/SoC team:

- **Recipe** — concrete configuration to try first (e.g. *max-abs calibration, top-2 layers in FP32, optional QAT 20-epoch fine-tune*)
- **Reasoning** — the trade-off behind each choice and the measured number that supports it
- **Boundary** — what the decision *doesn't* cover (verified vs hypothesized vs out-of-scope)

Four orthogonal recipes are explored, each backed by a measurement artifact in `results/`:

| § | Recipe axis | Question it answers |
|---|---|---|
| 2.2 | **Format shootout** | What does each precision option (FP32 / FP16 / BF16 / INT8 PTQ / INT8 QAT) cost on PSNR, SSIM, LPIPS, and on-disk size? |
| 2.3 | **Calibration method** | Within INT8 PTQ, which calibration scheme is the safe default? |
| 2.4 | **Per-layer sensitivity** | Which layers actually pay the INT8 quantization tax? |
| 2.5 | **Mixed precision** | If we keep top-N sensitive layers in FP32, how much PSNR comes back, and where is the knee? |
| 2.6 | **QAT fine-tuning** | When PTQ alone leaves too much on the table, does a short QAT phase recover it? |

### 2.2 Format shootout — FP32 / FP16 / BF16 / INT8 PTQ / INT8 QAT

A single table puts every precision option on the same val set, with the same metric implementations, so rows are directly comparable.

**Result** (on DIV2K val, 100 images, EDSR-baseline 200ep)

| Format | PSNR (dB) | ΔPSNR | SSIM | LPIPS | Size (MB) |
|---|---|---|---|---|---|
| FP32 (baseline) | 27.439 | — | 0.7907 | 0.2108 | 5.23 |
| FP16 (autocast) | 27.438 | −0.001 | 0.7907 | 0.2108 | 2.61 |
| BF16 (autocast) | 27.422 | −0.017 | 0.7904 | 0.2093 | 2.61 |
| INT8 PTQ (fake-quant) | 27.359 | −0.080 | 0.7863 | 0.1955 | 1.31 |
| FP32 (QAT weights, fake-quant off) | **27.501** | **+0.063** | 0.7932 | 0.2050 | 5.23 |
| INT8 QAT (fake-quant) | 27.446 | **+0.007** | 0.7893 | 0.1900 | 1.31 |

Two non-obvious rows: the **FP32 (QAT weights)** row isolates the *training-time* effect of QAT (better than the original FP32 baseline because 20 extra fine-tune epochs help), and **INT8 QAT** lands within noise of the original FP32 baseline at ¼ the size. The LPIPS column drops the cleanest under INT8 — interpreted with magnitude check in [learning/int8_perception_finding.md](learning/int8_perception_finding.md).

**How to reproduce**

```bash
# FP32 / FP16 / BF16 / INT8 PTQ rows (also runs sensitivity in 2.4 unless skipped)
python -m src.quantization.analyze \
    --checkpoint results/runs/<fp32_run>/checkpoints/best.pt \
    --output-dir results/quantization/200ep_with_report

# Append the two QAT rows (FP32-mode + INT8-mode of the QAT checkpoint)
python -m src.quantization.eval_qat \
    --qat-checkpoint  results/runs/<qat_run>/checkpoints/best_qat.pt \
    --fp32-checkpoint results/runs/<fp32_run>/checkpoints/best.pt \
    --shootout-csv results/quantization/200ep_with_report/shootout.csv \
    --shootout-md  results/quantization/200ep_with_report/shootout.md
```

**Scripts** — [src/quantization/analyze.py](src/quantization/analyze.py) (shootout entry; LPIPS via `lpips` SqueezeNet backbone), [src/quantization/eval_qat.py](src/quantization/eval_qat.py) (QAT row appender)
**Outputs** — [results/quantization/200ep_with_report/shootout.md](results/quantization/200ep_with_report/shootout.md) · [shootout.csv](results/quantization/200ep_with_report/shootout.csv)

### 2.3 Calibration ablation — max-abs vs percentile

Within INT8 PTQ, the calibration scheme is the first knob a vendor will turn. This compares four schemes on the same model:

| Scheme | PSNR (dB) | ΔPSNR vs FP32 | SSIM | Comment |
|---|---|---|---|---|
| max-abs | 27.364 | −0.075 | 0.7866 | default; preserves outlier weights |
| percentile-99.99 | 27.363 | −0.075 | 0.7887 | matches max-abs PSNR, slightly better SSIM |
| percentile-99.9 | 26.986 | −0.453 | 0.7840 | starts clipping; visibly worse |
| percentile-99.0 | 25.272 | −2.166 | 0.7539 | fully broken |

The PSNR spread between max-abs and percentile-99.99 is < 0.01 dB — **either is a safe default**. Percentile-99.9 is the cliff. → **Vendor input:** default to `max-abs`; if SSIM-leaning, try `percentile-99.99`; do *not* use 99.9 or below as the first attempt.

**How to reproduce**

```bash
python -m src.quantization.calibration_ablation \
    --checkpoint results/runs/<fp32_run>/checkpoints/best.pt \
    --output-dir results/quantization/calibration_ablation
```

**Script** — [src/quantization/calibration_ablation.py](src/quantization/calibration_ablation.py)
**Output** — [results/quantization/calibration_ablation/calibration_ablation.md](results/quantization/calibration_ablation/calibration_ablation.md) · [ablation.csv](results/quantization/calibration_ablation/ablation.csv) · [histograms.png](results/quantization/calibration_ablation/histograms.png)

### 2.4 Per-layer sensitivity

Each of the 36 Conv2d layers is INT8-quantized in isolation while the rest stay FP32. The PSNR drop for that single layer is the layer's *sensitivity score*.

The top of the ranking is concentrated and intuitive: pixel-shuffle / final-projection / first-conv layers hurt the most.

| Rank | Layer | PSNR drop (dB) when this layer is INT8 alone |
|---|---|---|
| 1 | `tail` | 0.029 |
| 2 | `upsampler.0` | 0.020 |
| 3 | `head` | 0.016 |
| 4 | `body.16` | 0.007 |
| 5+ | `body.*.conv2` (residual blocks) | < 0.001 each |

→ The body of the network is **highly INT8-tolerant**; the heads/tail/upsampler carry almost all the sensitivity. This is exactly the input the mixed-precision sweep needs.

**How to reproduce** — sensitivity is computed by the same `analyze.py` invocation as 2.2 (skip with `--skip-sensitivity` if not needed).

**Output** — [sensitivity.md](results/quantization/200ep_with_report/sensitivity.md) · [sensitivity.csv](results/quantization/200ep_with_report/sensitivity.csv)

### 2.5 Mixed-precision sweep — PTQ vs QAT

Walk N from 0 to 8: keep the top-N most-sensitive layers (per 2.4) in FP32, INT8 the rest, measure PSNR. Run the sweep twice — once over the PTQ baseline, once over the QAT-trained weights — and overlay.

![PTQ vs QAT mixed precision sweep](results/mixed_precision/ptq_vs_qat_sweep.png)

| N (FP32 layers) | PTQ PSNR | QAT PSNR | QAT − FP32 baseline |
|---|---|---|---|
| 0 (all-INT8) | 27.358 | 27.446 | +0.007 |
| 2 (tail, upsampler.0) | 27.405 | **27.485** | **+0.046** |
| 4 (top-4) | 27.428 | 27.493 | +0.054 |
| 8 (top-8) | 27.431 | 27.496 | +0.057 |

Two takeaways:
- **PTQ knee at N≈4** — past 4 sensitive layers in FP32, returns flatten.
- **QAT path lifts the entire curve above the original FP32 baseline (27.439)** even at N=0 (all-INT8). For a vendor without FP32 fallback support, that's the meaningful win — mixed precision becomes optional rather than mandatory.

**How to reproduce**

```bash
# PTQ sweep (default)
python -m src.deployment.mixed_precision \
    --checkpoint  results/runs/<fp32_run>/checkpoints/best.pt \
    --sensitivity results/quantization/200ep_with_report/sensitivity.csv \
    --output-dir  results/mixed_precision/edsr_200ep

# QAT sweep (loads QAT weights, skips re-calibration)
python -m src.deployment.mixed_precision --qat \
    --checkpoint  results/runs/<qat_run>/checkpoints/best_qat.pt \
    --sensitivity results/quantization/200ep_with_report/sensitivity.csv \
    --output-dir  results/mixed_precision/edsr_200ep_qat

# Overlay both curves with FP32 baseline reference
python -m src.deployment.compare_mixed_precision \
    --ptq-csv results/mixed_precision/edsr_200ep/mixed_precision_sweep.csv \
    --qat-csv results/mixed_precision/edsr_200ep_qat/mixed_precision_sweep.csv \
    --output  results/mixed_precision/ptq_vs_qat_sweep.png \
    --fp32-baseline 27.439
```

**Scripts** — [src/deployment/mixed_precision.py](src/deployment/mixed_precision.py) · [src/deployment/compare_mixed_precision.py](src/deployment/compare_mixed_precision.py)
**Outputs** — [results/mixed_precision/edsr_200ep/](results/mixed_precision/edsr_200ep/) · [results/mixed_precision/edsr_200ep_qat/](results/mixed_precision/edsr_200ep_qat/) · [ptq_vs_qat_sweep.png](results/mixed_precision/ptq_vs_qat_sweep.png)

### 2.6 QAT fine-tuning

Recipe (implemented in [src/training/train.py:404-551](src/training/train.py#L404-L551), `--qat` flag):

1. Load the FP32 `best.pt`.
2. Wrap every `Conv2d` with `CalibratingConv2d` (fake-quant + activation scale buffer).
3. Run a short calibration pass on training data (default 20 batches) to set per-tensor activation `amax`, then freeze scales.
4. Switch to **QAT mode** (fake-quant on, weight gradients via Straight-Through Estimator) and fine-tune.
5. Default schedule — **20 epochs, lr = 1e-5** (10× smaller than base), CosineAnnealingLR. Validation runs in *quantize mode* (clean fake-INT8 measurement, no STE noise leaking into PSNR).

The conservative LR is intentional: the model is already converged, the fine-tune only needs to absorb quantization noise. Going to 50 epochs / 5e-5 rarely helps in our experiments and risks regression — full reasoning in [learning/when_to_use_qat.md](learning/when_to_use_qat.md).

**How to reproduce**

```bash
# Train FP32 then QAT in one shot
python -m src.training.train --compile --qat

# QAT only (skip FP32 phase) starting from an existing checkpoint
python -m src.training.train --epochs 0 --qat \
    --qat-from results/runs/<fp32_run>/checkpoints/best.pt
```

**Output** — Each QAT run writes a separate `<run>_qat/` directory next to the FP32 run, with `best_qat.pt`, QAT-phase `curves.png`, and `metrics.csv` so the FP32 baseline is preserved untouched.

### 2.7 Decision package — what the vendor receives

| Question | Recipe (this repo's first answer) | Reasoning anchor |
|---|---|---|
| Default INT8 calibration? | `max-abs` | 2.3 — spread vs percentile-99.99 < 0.01 dB |
| First mixed-precision target? | top-2 FP32: `tail` + `upsampler.0` | 2.4/2.5 — 73% of the FP32-vs-INT8 PSNR gap recovered |
| When to invest in QAT? | If PTQ drop > 0.2 dB *or* vendor has no FP32 fallback | 2.5 — QAT all-INT8 already exceeds original FP32 baseline |
| Acceptable Native FP16 / BF16? | Both: FP16 is identical, BF16 within 0.02 dB | 2.2 — keep as ½-size fallbacks |

Each row is intentionally a **path**, not a final config: the vendor can adapt — re-rank layers on their own data, re-run the sweep against their hardware's mixed-precision support, swap calibration to percentile if their distribution warrants it. Three classes are deliberately *out of scope* (architecture choice, INT4 / mixed-bit, customer-distribution edge cases) and surfaced in section 4 rather than hidden — see [learning/senior_deliverable_framing.md](learning/senior_deliverable_framing.md) for the full rationale.

---

## On AI assistance

I used an AI coding assistant (Claude Code) for implementation — typing, boilerplate, and refactoring. The engineering judgments are mine: scope cuts and KPI selection, choice of recipes / detectors and what *not* to include, verification against measured numbers, and the explicit *verified / hypothesized / cannot-verify / out-of-scope* boundary that runs through every finding.
