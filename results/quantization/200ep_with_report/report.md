# Quantization Analysis Report

> **Scope.** This report covers PyTorch-side quantization analysis (fake-quant simulation + per-layer sensitivity sweep + calibration scheme + perceptual cross-check). All measurements are **accuracy / fidelity / sensitivity** -- there is **no latency** here on purpose, because PyTorch fake-quant carries simulation overhead that does not exist in real deploy. Real per-precision latency lives in the ONNX deploy benchmark (`results/onnx_benchmark/edsr_200ep_full/deploy_summary.md`) on a consumer GPU; **TV SoC NPU latency would be measured via the vendor SDK on a dev board** and is not in scope here. The per-layer sensitivity ranking and the mixed-precision recipe below port across hardware; absolute latency does not.

## What was tested

- **Generated**: 2026-05-02T16:25:19
- **Checkpoint**: `results\runs\20260427_143542_ep200_b16_scale2_realistic\checkpoints\best.pt`
  - last modified: 2026-04-28T10:19:45, size: 15.75 MB
- **Model**: EDSR(scale_factor=2, n_resblocks=16, n_feats=64) -- 1,369,859 params
- **Device**: cuda (NVIDIA GeForce RTX 3060 Laptop GPU), PyTorch 2.6.0+cu124
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` -- 100 images, realistic degradation, LR patch 96x96 (deterministic per-index seed)
- **Calibration**: 8 batches (64 LR samples), max-abs (no percentile clipping, no KL-div)
- **Quantization scheme**:
  - activations: symmetric per-tensor INT8 (range -128..127)
  - weights: symmetric per-output-channel INT8 (range -128..127)

## Format shootout (accuracy only)

PSNR + SSIM + LPIPS for each format on the val set. **Latency is intentionally not measured at this stage** -- PyTorch fake-quant and autocast both insert simulation ops that don't exist in real deploy, so reporting them next to accuracy invites readers to draw the wrong conclusion (e.g. "INT8 is slower than FP32" -- true in fake-quant simulation, false on real INT8 backends). Real per-precision latency lives in the ONNX deploy benchmark.

Three-metric stack: PSNR is the SR-literature headline (pixel MSE). SSIM cross-checks local structure. LPIPS uses pretrained CNN features and tracks human perception -- it is the metric that catches banding / posterization artifacts that PSNR systematically under-rates. Watching all three together, especially when they **disagree on direction**, is far more informative than any single number.

| Format | PSNR (dB) | PSNR drop | SSIM | SSIM drop | LPIPS | LPIPS rise | Size (MB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 | 27.439 | +0.000 | 0.7907 | +0.0000 | 0.2108 | +0.0000 | 5.23 | baseline |
| FP16 (autocast) | 27.438 | +0.001 | 0.7907 | -0.0000 | 0.2108 | -0.0000 | 2.61 | weights FP32, ops cast on the fly |
| BF16 (autocast) | 27.422 | +0.017 | 0.7904 | +0.0002 | 0.2093 | -0.0016 | 2.61 | wider exponent than FP16, less overflow risk |
| INT8 PTQ (fake-quant) | 27.359 | +0.079 | 0.7863 | +0.0044 | 0.1955 | **-0.0154** | 1.31 | fake-quant: q-dq inserted in FP32 math (accuracy-only signal) |
| **FP32 (QAT weights)** | **27.501** | **-0.063** | **0.7932** | **+0.0025** | **0.2050** | **-0.0058** | **5.23** | QAT-trained weights, fake-quant OFF (isolates training-time effect) |
| **INT8 QAT (fake-quant)** | **27.446** | **-0.007** | **0.7893** | **+0.0014** | **0.1900** | **-0.0208** | **1.31** | QAT 20-epoch fine-tune from PTQ baseline, lr 1e-5 |

### QAT decomposed: training effect vs inference effect

The "QAT (FP32 weights)" row isolates what 20 epochs of QAT fine-tune did to the underlying weights *independent of inference-time fake-quant*. Two distinct effects show up:

1. **QAT improved the underlying FP32 weights** by +0.063 dB PSNR / +0.0025 SSIM vs the original FP32 baseline. The training-time fake-quant noise acts as a mild regularizer, plus the additional 20 epochs at lr 1e-5 gave the model more time to converge.
2. **INT8 inference on those QAT weights drops modestly** (-0.055 dB from QAT-FP32 to INT8 QAT) -- much smaller than the PTQ-from-baseline drop (-0.079 dB).

Net effect: INT8 QAT (27.446) lands ~at the original FP32 baseline (27.439). Two ways to read this:

- **Naive reading**: "QAT recovered the PTQ drop." Technically true but misses the structure -- QAT actually *improved* the FP32 floor first, then INT8 inference partially gave that back.
- **Senior reading**: QAT delivers two independently-useful outcomes -- regularized FP32 weights *and* INT8 robustness. For deployment, the relevant comparison is "QAT-INT8 vs PTQ-INT8" (27.446 vs 27.359, +0.087 dB), because production deploys INT8 not the FP32 reference.

LPIPS continues the perception-distortion direction we documented in the shootout (-0.0154 PTQ → -0.0208 QAT), even though the QAT FP32 row's LPIPS (-0.0058) is closer to baseline. That is, INT8 inference pushes LPIPS *further from* its FP32 source under QAT, reinforcing that the LPIPS divergence on INT8 is dominated by inference-time fake-quant noise, not by the underlying weight changes.

## Perceptual evaluation (LPIPS): the perception-distortion tradeoff

The INT8 row above shows a finding that contradicts naive intuition: **PSNR and SSIM both worsen on INT8, but LPIPS *improves* (lower = closer to GT in feature space)**. This is not a measurement artifact -- it reproduces robustly across 99 / 100 val images (see `lpips_heatmaps/distribution.png` and `lpips_heatmaps/per_image_lpips.csv`).

The mechanism is the **perception-distortion tradeoff** (Blau & Michaeli, CVPR 2018). EDSR-baseline is L1-trained on a realistic-degradation pipeline; L1 loss is well known to push the network toward an over-smooth conditional mean of the HR distribution, which loses some high-frequency texture relative to natural-image GTs. INT8 quantization adds broadband, feature-level noise. On this val set, that noise pushes the SR output's CNN-feature distribution back **toward** the GT's distribution -- so INT8 SR registers as more perceptually similar to GT than FP32 SR does, even while INT8 has worse pixel-level fidelity.

Implications:

1. **PSNR drop on INT8 is not "perceptual harm".** The 0.08 dB PSNR drop is a real pixel-fidelity loss, but on a metric that correlates with human perception (LPIPS), INT8 is at least as good as FP32 here. For a TV upscaling product where customer-perceived quality is the bottom line, the PSNR-based pessimism about INT8 may be misplaced.

2. **Aggregate LPIPS hides spatial structure.** See `lpips_heatmaps/heatmap_0879.png`: the spatial LPIPS map of INT8-SR vs FP32-SR (3-panel: GT HR | INT8 SR | heatmap overlay) shows the **quantization-induced perceptual changes localize in smooth regions** (the sky behind the cathedral), exactly where banding would be expected. The intricate facade and dome texture barely move. So even though aggregate LPIPS says "INT8 is fine", a sky-heavy frame still risks visible banding, and a smooth-region-aware mitigation (mixed precision on tail / output dithering) remains worthwhile.

3. **Per-image distribution shows variance, not failure cases.** All 100 val images cluster around a small negative rise (mean -0.0154, max -0.0848, min +0.0015). There is no "INT8 catastrophically fails on smooth-heavy images" tail in this metric -- the naive "PSNR-says-OK-but-LPIPS-explodes-on-skies" hypothesis was *not* supported by the data. The failure mode, if any, is sub-perceptual on a CNN-feature scale.

The PSNR / SSIM / LPIPS triplet disagreeing in direction is the headline finding here, not the magnitude of any single drop. See `shootout.md` for the per-format table; see `lpips_heatmaps/` for the per-image distribution and the spatial heatmap on `0879.png`.

## Per-layer sensitivity (INT8, sorted by PSNR drop)

Each row: hold every other layer in FP32, fake-quantize ONLY this Conv2d to INT8, measure PSNR + SSIM drop. Larger drop = more quantization-sensitive. Ranking is by PSNR drop; SSIM is a perceptual cross-check (a layer with big PSNR drop but tiny SSIM drop is dropping pixel fidelity in low-structure regions and may not need extra precision).

| Rank | Layer | PSNR (dB) | PSNR drop | SSIM | SSIM drop | Class |
|---:|---|---:|---:|---:|---:|---|
| 1 | `tail` | 27.409 | +0.029 | 0.7887 | +0.0020 | output |
| 2 | `upsampler.0` | 27.418 | +0.020 | 0.7894 | +0.0012 | upsampler |
| 3 | `head` | 27.423 | +0.016 | 0.7907 | -0.0000 | input |
| 4 | `body.16` | 27.432 | +0.007 | 0.7901 | +0.0005 | post-resblock |
| 5 | `body.0.conv2` | 27.438 | +0.001 | 0.7906 | +0.0001 | resblock-interior |
| 6 | `body.15.conv2` | 27.438 | +0.001 | 0.7906 | +0.0001 | resblock-interior |
| 7 | `body.1.conv2` | 27.438 | +0.001 | 0.7907 | +0.0000 | resblock-interior |
| 8 | `body.13.conv2` | 27.438 | +0.001 | 0.7906 | +0.0001 | resblock-interior |
| 9 | `body.14.conv2` | 27.438 | +0.001 | 0.7906 | +0.0000 | resblock-interior |
| 10 | `body.5.conv2` | 27.438 | +0.001 | 0.7906 | +0.0000 | resblock-interior |
| 11 | `body.12.conv2` | 27.438 | +0.001 | 0.7906 | +0.0000 | resblock-interior |
| 12 | `body.6.conv2` | 27.438 | +0.001 | 0.7906 | +0.0000 | resblock-interior |
| 13 | `body.10.conv2` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 14 | `body.15.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 15 | `body.3.conv1` | 27.438 | +0.000 | 0.7906 | +0.0001 | resblock-interior |
| 16 | `body.11.conv2` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 17 | `body.8.conv2` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 18 | `body.7.conv2` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 19 | `body.9.conv2` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 20 | `body.14.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 21 | `body.13.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 22 | `body.8.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 23 | `body.11.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 24 | `body.6.conv1` | 27.438 | +0.000 | 0.7907 | -0.0000 | resblock-interior |
| 25 | `body.12.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 26 | `body.9.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 27 | `body.10.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 28 | `body.7.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 29 | `body.4.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 30 | `body.5.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 31 | `body.4.conv2` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 32 | `body.0.conv1` | 27.438 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 33 | `body.2.conv2` | 27.439 | +0.000 | 0.7907 | +0.0000 | resblock-interior |
| 34 | `body.2.conv1` | 27.439 | -0.000 | 0.7907 | +0.0000 | resblock-interior |
| 35 | `body.1.conv1` | 27.439 | -0.000 | 0.7907 | -0.0000 | resblock-interior |
| 36 | `body.3.conv2` | 27.439 | -0.000 | 0.7907 | -0.0000 | resblock-interior |

## Mixed-precision recipe (sensitivity-driven)

- Pure-INT8 (all 36 Conv2d) drop vs FP32: **PSNR +0.080 dB** / **SSIM +0.0044**
- Top-3 most sensitive layers (by PSNR drop) contribute **82%** of the PSNR drop and **73%** of the SSIM drop:
  - `tail` (output): PSNR +0.029 dB, SSIM +0.0020
  - `upsampler.0` (upsampler): PSNR +0.020 dB, SSIM +0.0012
  - `head` (input): PSNR +0.016 dB, SSIM +-0.0000

**Recipe**: keep `tail, upsampler.0, head` in higher precision (FP16 / FP32) and INT8 the remaining 33 Conv2d. Estimated combined drop with this mixed scheme: **~PSNR +0.017 dB / ~SSIM +0.0012** (assuming layer drops are approximately additive; the data above supports this -- sum of all single-layer PSNR drops is 0.083 dB vs measured pure-INT8 drop 0.080 dB).

This pattern matches SR PTQ literature (PAMS, 2DQuant): the input conv (`head`), output conv (`tail`), and upsampling convs are quantization-critical; ResBlock interior is robust. The pattern is **expected to port across hardware** (NPU, mobile DSP, vendor SDK) because it reflects information-theoretic sensitivity of the model itself, not a backend-specific quirk. The latency cost / benefit of mixing precisions, however, is hardware-specific and must be re-measured on the actual deploy silicon (NPU dev board for TV SoC).
