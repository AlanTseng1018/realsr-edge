# ONNX Deployment Benchmark

Single execution node of the ONNX runtime benchmark. Each row is one (ONNX file, ORT execution provider) pair, evaluated on the same val set with the same input shape for latency.

## What was tested

- **Generated**: 2026-05-10T17:32:51
- **ONNX folder**: `results\onnx_exports\edsr_200ep`
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` (100 images, realistic degradation, LR patch 96x96)
- **Latency input shape**: `[1, 3, 96, 96]` (10 warmup + 50 timed iters)
- **Providers tested**: `tensorrt`, `cuda`, `cpu`
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU
- **ORT version**: 1.25.0

## Shootout table

| ONNX | Provider | PSNR (dB) | PSNR drop | SSIM | SSIM drop | LPIPS | LPIPS drop | Latency (ms) | Speedup vs FP32 same-provider | Size (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `edsr_fp16.onnx` | `tensorrt` | 27.437 | +0.002 | 0.7906 | +0.0001 | 0.2102 | -0.0006 | 1.28 +/- 0.06 | 2.57x faster | 2.63 |
| `edsr_fp16.onnx` | `cuda` | 27.438 | +0.001 | 0.7907 | -0.0000 | 0.2108 | -0.0000 | 4.05 +/- 0.99 | 1.30x faster | 2.63 |
| `edsr_fp16.onnx` | `cpu` | 27.439 | -0.000 | 0.7907 | -0.0000 | 0.2108 | -0.0000 | 50.55 +/- 2.81 | 1.03x slower | 2.63 |
| `edsr_fp32.onnx` | `tensorrt` | 27.439 | +0.000 | 0.7907 | +0.0000 | 0.2108 | +0.0000 | 3.28 +/- 0.77 | 1.00x faster | 5.24 |
| `edsr_fp32.onnx` | `cuda` | 27.439 | +0.000 | 0.7907 | -0.0000 | 0.2108 | -0.0000 | 5.28 +/- 2.23 | 1.00x faster | 5.24 |
| `edsr_fp32.onnx` | `cpu` | 27.439 | +0.000 | 0.7907 | +0.0000 | 0.2108 | -0.0000 | 49.17 +/- 2.10 | 1.00x faster | 5.24 |
| `edsr_int8_static.onnx` | `tensorrt` | 27.268 | +0.171 | 0.7796 | +0.0111 | 0.1841 | -0.0267 | 4.33 +/- 3.47 | 1.32x slower | 1.43 |
| `edsr_int8_static.onnx` | `cuda` | 27.267 | +0.171 | 0.7796 | +0.0110 | 0.1839 | -0.0270 | 6.57 +/- 1.70 | 1.24x slower | 1.43 |
| `edsr_int8_static.onnx` | `cpu` | 27.268 | +0.171 | 0.7796 | +0.0111 | 0.1842 | -0.0266 | 56.25 +/- 4.43 | 1.14x slower | 1.43 |

## How to read

- **PSNR** is the deploy-side accuracy: ONNX session output evaluated on the val set against HR ground truth. **Provider-invariant within rounding** -- if it differs much between CUDA and CPU for the same ONNX, that's a debug signal.
- **SSIM** is the perceptual cross-check on the same tensor pair. PSNR can drop 0.2 dB without visible artefacts; SSIM going down by ~0.001 is usually safe, ~0.005+ is the point at which side-by-side viewing starts to reveal the loss. Disagreement between PSNR and SSIM rankings is a signal that the precision is hurting structure (edges) more than bulk fidelity, or vice versa.
- **LPIPS** is the perceptual distance from a SqueezeNet feature embedding (lower = more perceptually similar to GT; same backbone as §2.2's quantization analysis so the numbers are directly comparable). Often disagrees with PSNR — INT8 frequently *reduces* LPIPS even when PSNR drops, because the quantization noise pattern is closer to natural image statistics than the FP32 baseline's smoother output.
- **Drop vs FP32** uses the FP32 PSNR / SSIM / LPIPS as baseline. Should match (within ~0.1 dB / ~0.001 SSIM / ~0.005 LPIPS) the fake-quant prediction in `results/quantization/200ep_with_report/report.md`.
- **Latency** is forward-pass only. **Provider-specific**: INT8 ONNX often runs slower on CUDA EP than FP32 due to QDQ insertion + memcpy nodes; on CPU EP, INT8 typically wins because of VNNI / native INT8 instructions. **TensorRT EP is the right path for true GPU INT8 deploy** (not benchmarked here).
- **Speedup** is per provider: it answers "if I'm deploying on this hardware, what does each precision give me?" -- not "is X precision globally fastest".

