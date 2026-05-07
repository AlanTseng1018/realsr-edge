# Calibration Method Ablation

## What was tested

- **Generated**: 2026-05-02T16:30:00
- **Checkpoint**: `results\runs\20260427_143542_ep200_b16_scale2_realistic\checkpoints\best.pt`
  - last modified: 2026-04-28T10:19:45, size: 15.75 MB
- **Model**: EDSR(scale_factor=2, n_resblocks=16, n_feats=64) -- 1,369,859 params
- **Device**: cuda (NVIDIA GeForce RTX 3060 Laptop GPU), PyTorch 2.6.0+cu124
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` -- 100 images, realistic degradation, LR patch 96x96
- **Calibration**: 8 batches (64 LR samples)
- **Histogram bins**: 2048
- **Quantization scheme**: symmetric per-tensor INT8 (activations) + symmetric per-channel INT8 (weights)

## Calibration scheme shootout (accuracy only)

All four schemes share the **same calibration pass** -- the histogram is collected once, and each scheme just chooses a different summary of it (running max for `max-abs`, percentile cutoff for the other three).

PSNR is the primary ranking metric; SSIM is reported alongside as a perceptual cross-check. A calibration scheme that wins PSNR but loses SSIM (or vice versa) is a red flag worth investigating before committing to deploy.

**Latency is intentionally not reported here**. Calibration scheme only changes the per-tensor scale value -- it does not change op count, kernel selection, or anything that affects runtime. Any latency differences between schemes would be measurement noise, not a real deploy signal. Real per-precision latency lives in the ONNX deploy benchmark (`results/onnx_benchmark/.../deploy_summary.md`).

| Scheme | PSNR (dB) | PSNR drop | SSIM | SSIM drop |
|---|---:|---:|---:|---:|
| max-abs | 27.364 | +0.075 | 0.7866 | +0.0041 |
| percentile-99.99 | 27.363 | +0.075 | 0.7887 | +0.0020 |
| percentile-99.9 | 26.986 | +0.453 | 0.7840 | +0.0066 |
| percentile-99.0 | 25.272 | +2.166 | 0.7539 | +0.0368 |

## Per-layer chosen `amax` per scheme

This is the value that drives `scale = amax / 127` for each layer's input quantizer. Smaller values = tighter clipping = the tail of the activation distribution gets saturated.

| Layer | max-abs | percentile-99.99 | percentile-99.9 | percentile-99.0 |
|---|---:|---:|---:|---:|
| `head` | 1.0000 | 1.0000 | 0.9999 | 0.9959 |
| `tail` | 1.8376 | 1.2213 | 1.0106 | 0.7297 |
| `body.16` | 3.1245 | 1.5475 | 1.0721 | 0.5686 |
| `body.0.conv1` | 1.4732 | 1.3646 | 1.1966 | 0.8880 |
| `body.0.conv2` | 1.3181 | 0.7507 | 0.6246 | 0.4735 |
| `body.1.conv1` | 1.2430 | 0.8783 | 0.7476 | 0.5108 |
| `body.1.conv2` | 1.5933 | 0.6634 | 0.4138 | 0.2855 |
| `body.2.conv1` | 1.1912 | 0.7506 | 0.5485 | 0.3562 |
| `body.2.conv2` | 1.3021 | 0.6761 | 0.3972 | 0.1811 |
| `body.3.conv1` | 1.2463 | 0.7355 | 0.5330 | 0.3232 |
| `body.3.conv2` | 1.7200 | 0.7003 | 0.4212 | 0.1893 |
| `body.4.conv1` | 1.2956 | 0.7675 | 0.5466 | 0.3257 |
| `body.4.conv2` | 1.0149 | 0.5333 | 0.3526 | 0.1676 |
| `body.5.conv1` | 1.3610 | 0.7917 | 0.5573 | 0.3203 |
| `body.5.conv2` | 1.2222 | 0.6916 | 0.3687 | 0.1734 |
| `body.6.conv1` | 1.4015 | 0.8073 | 0.5705 | 0.3289 |
| `body.6.conv2` | 1.5073 | 0.7188 | 0.4053 | 0.1719 |
| `body.7.conv1` | 1.5146 | 0.8344 | 0.5869 | 0.3381 |
| `body.7.conv2` | 1.3768 | 0.6200 | 0.3317 | 0.1678 |
| `body.8.conv1` | 1.5545 | 0.8569 | 0.5963 | 0.3425 |
| `body.8.conv2` | 1.5857 | 0.7830 | 0.3700 | 0.1794 |
| `body.9.conv1` | 1.7315 | 0.9124 | 0.6375 | 0.3629 |
| `body.9.conv2` | 1.4081 | 0.7168 | 0.4005 | 0.1840 |
| `body.10.conv1` | 1.7531 | 0.9660 | 0.6781 | 0.3862 |
| `body.10.conv2` | 1.5539 | 0.6884 | 0.3750 | 0.1774 |
| `body.11.conv1` | 1.9127 | 1.0087 | 0.7100 | 0.4000 |
| `body.11.conv2` | 1.5823 | 0.7672 | 0.3888 | 0.1881 |
| `body.12.conv1` | 2.1236 | 1.0711 | 0.7514 | 0.4206 |
| `body.12.conv2` | 1.9389 | 0.9200 | 0.4647 | 0.2168 |
| `body.13.conv1` | 2.3528 | 1.1762 | 0.8075 | 0.4450 |
| `body.13.conv2` | 2.0080 | 1.0676 | 0.5072 | 0.2300 |
| `body.14.conv1` | 2.5995 | 1.3390 | 0.8931 | 0.4829 |
| `body.14.conv2` | 2.0696 | 0.9556 | 0.5333 | 0.2474 |
| `body.15.conv1` | 2.7809 | 1.4551 | 0.9853 | 0.5215 |
| `body.15.conv2` | 2.5023 | 1.1547 | 0.6279 | 0.2680 |
| `upsampler.0` | 2.7222 | 1.6036 | 1.2464 | 0.9100 |

## How to read the histogram figure

`histograms.png` plots the activation-magnitude histogram (collected during calibration) for six representative layers. Vertical dashed lines mark where each scheme places its `amax`:

- **Red (max-abs)**: at the largest `|x|` ever seen. Most conservative (no clipping) but exposed to single outliers.
- **Orange (99.99)**: cut off the very last 0.01% of the tail.
- **Green (99.9)**: typical TensorRT-style aggressive choice.
- **Blue (99.0)**: aggressive clipping; saturates more than just outliers.

Y-axis is log-scale because activation distributions are heavily long-tailed -- a linear axis would render the tail invisible. The amax differences look small numerically, but the resulting INT8 scale (`amax / 127`) is what determines the bin resolution for the bulk of values, which is what shows up in the PSNR table above.

## Takeaway

On this checkpoint and val set, **`max-abs`** wins on PSNR (27.364 dB) and **`percentile-99.99`** wins on SSIM (0.7887). PSNR spread across schemes: 2.091 dB; SSIM spread: 0.0348.

**Mismatch**: PSNR prefers `max-abs` but SSIM prefers `percentile-99.99`. Inspect the histograms before committing -- this usually means one scheme preserves bulk pixel fidelity while the other preserves edge / structure better. Pick based on the deploy use-case (broadcast quality -> SSIM-leaning; benchmark scoring -> PSNR-leaning).

Interpret carefully:

- A small spread (< 0.05 dB) means activations are NOT outlier-heavy for this model -- the calibration choice is mostly cosmetic.
- A large spread (> 0.2 dB) means there ARE outliers that max-abs is exposed to. In that case, percentile clipping is a real win.
- If `percentile-99.0` is best, the tail isn't useful; consider an even tighter cutoff or a learned clipping threshold (PAMS-style).
- If `max-abs` is best, the tail IS informative; percentile clipping is throwing away useful signal.

All four numbers are produced from the same FP32 weights and the same calibration histogram. The infrastructure is in place to add KL-div or MSE-optimal calibration as additional schemes by adding entries to `SCHEMES` in `calibration_ablation.py`.
