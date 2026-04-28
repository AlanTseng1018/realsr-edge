# Quantization Analysis Report

## What was tested

- **Generated**: 2026-04-28T10:36:40
- **Checkpoint**: `results\runs\20260427_143542_ep200_b16_scale2_realistic\checkpoints\best.pt`
  - last modified: 2026-04-28T10:19:45, size: 15.75 MB
- **Model**: EDSR(scale_factor=2, n_resblocks=16, n_feats=64) -- 1,369,859 params
- **Device**: cuda (NVIDIA GeForce RTX 3060 Laptop GPU), PyTorch 2.6.0+cu124
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` -- 100 images, realistic degradation, LR patch 96x96 (deterministic per-index seed)
- **Calibration**: 8 batches (64 LR samples), max-abs (no percentile clipping, no KL-div)
- **Quantization scheme**:
  - activations: symmetric per-tensor INT8 (range -128..127)
  - weights: symmetric per-output-channel INT8 (range -128..127)

## Format shootout

PSNR + forward-pass latency for each format on the val set. **Caveat**: the INT8 latency is fake-quant overhead (q-dq inserted in PyTorch float math), not real INT8 deploy latency. Real deploy-side latency requires ONNX Runtime / TensorRT / vendor NPU SDK.

| Format | PSNR (dB) | Drop vs FP32 | Latency (ms) | Size (MB) | Notes |
|---|---:|---:|---:|---:|---|
| FP32 | 27.439 | +0.000 | 3.89 +/- 0.04 | 5.23 | baseline |
| FP16 (autocast) | 27.438 | +0.001 | 3.54 +/- 0.15 | 2.61 | weights FP32, ops cast on the fly |
| BF16 (autocast) | 27.422 | +0.017 | 3.85 +/- 0.18 | 2.61 | wider exponent than FP16, less overflow risk |
| INT8 PTQ (fake-quant) | 27.359 | +0.079 | 10.71 +/- 1.43 | 1.31 | PSNR is real; latency is fake-quant overhead (NOT deploy latency) |

## Per-layer sensitivity (INT8, sorted by drop)

Each row: hold every other layer in FP32, fake-quantize ONLY this Conv2d to INT8, measure PSNR drop. Larger drop = more quantization-sensitive.

| Rank | Layer | PSNR (dB) | Drop (dB) | Class |
|---:|---|---:|---:|---|
| 1 | `tail` | 27.409 | +0.029 | output |
| 2 | `upsampler.0` | 27.418 | +0.020 | upsampler |
| 3 | `head` | 27.423 | +0.016 | input |
| 4 | `body.16` | 27.432 | +0.007 | post-resblock |
| 5 | `body.0.conv2` | 27.438 | +0.001 | resblock-interior |
| 6 | `body.15.conv2` | 27.438 | +0.001 | resblock-interior |
| 7 | `body.1.conv2` | 27.438 | +0.001 | resblock-interior |
| 8 | `body.14.conv2` | 27.438 | +0.001 | resblock-interior |
| 9 | `body.5.conv2` | 27.438 | +0.001 | resblock-interior |
| 10 | `body.13.conv2` | 27.438 | +0.001 | resblock-interior |
| 11 | `body.12.conv2` | 27.438 | +0.001 | resblock-interior |
| 12 | `body.6.conv2` | 27.438 | +0.001 | resblock-interior |
| 13 | `body.10.conv2` | 27.438 | +0.000 | resblock-interior |
| 14 | `body.15.conv1` | 27.438 | +0.000 | resblock-interior |
| 15 | `body.3.conv1` | 27.438 | +0.000 | resblock-interior |
| 16 | `body.8.conv2` | 27.438 | +0.000 | resblock-interior |
| 17 | `body.11.conv2` | 27.438 | +0.000 | resblock-interior |
| 18 | `body.7.conv2` | 27.438 | +0.000 | resblock-interior |
| 19 | `body.9.conv2` | 27.438 | +0.000 | resblock-interior |
| 20 | `body.0.conv1` | 27.438 | +0.000 | resblock-interior |
| 21 | `body.14.conv1` | 27.438 | +0.000 | resblock-interior |
| 22 | `body.13.conv1` | 27.438 | +0.000 | resblock-interior |
| 23 | `body.8.conv1` | 27.438 | +0.000 | resblock-interior |
| 24 | `body.11.conv1` | 27.438 | +0.000 | resblock-interior |
| 25 | `body.6.conv1` | 27.438 | +0.000 | resblock-interior |
| 26 | `body.12.conv1` | 27.438 | +0.000 | resblock-interior |
| 27 | `body.9.conv1` | 27.438 | +0.000 | resblock-interior |
| 28 | `body.10.conv1` | 27.438 | +0.000 | resblock-interior |
| 29 | `body.7.conv1` | 27.438 | +0.000 | resblock-interior |
| 30 | `body.4.conv1` | 27.438 | +0.000 | resblock-interior |
| 31 | `body.5.conv1` | 27.438 | +0.000 | resblock-interior |
| 32 | `body.4.conv2` | 27.438 | +0.000 | resblock-interior |
| 33 | `body.2.conv2` | 27.439 | -0.000 | resblock-interior |
| 34 | `body.2.conv1` | 27.439 | -0.000 | resblock-interior |
| 35 | `body.1.conv1` | 27.439 | -0.000 | resblock-interior |
| 36 | `body.3.conv2` | 27.439 | -0.000 | resblock-interior |

## Mixed-precision recommendation

- Pure-INT8 (all 36 Conv2d) PSNR drop vs FP32: **+0.079 dB**
- Top-3 most sensitive layers contribute **83%** of the total drop:
  - `tail` (output): +0.029 dB
  - `upsampler.0` (upsampler): +0.020 dB
  - `head` (input): +0.016 dB

**Recipe**: keep `tail, upsampler.0, head` in higher precision (FP16 / FP32) and INT8 the remaining 33 Conv2d. Estimated combined drop with this mixed scheme: **~+0.017 dB** (assuming layer drops are approximately additive; the data above supports this -- sum of all single-layer drops is 0.083 dB vs measured pure-INT8 drop 0.079 dB).

This pattern matches SR PTQ literature (PAMS, 2DQuant): the input conv (`head`), output conv (`tail`), and upsampling convs are quantization-critical; ResBlock interior is robust. Once a real INT8 backend (ONNX RT QInt8, TensorRT) is in place, this recipe should be re-validated on actual deploy latency.
