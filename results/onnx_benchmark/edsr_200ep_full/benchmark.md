# ONNX Deployment Benchmark

Single execution node of the ONNX runtime benchmark. Each row is one (ONNX file, ORT execution provider) pair, evaluated on the same val set with the same input shape for latency.

## What was tested

- **Generated**: 2026-04-30T01:01:59
- **ONNX folder**: `results\onnx_exports\edsr_200ep`
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` (100 images, realistic degradation, LR patch 96x96)
- **Latency input shape**: `[1, 3, 96, 96]` (10 warmup + 50 timed iters)
- **Providers tested**: `tensorrt`, `cuda`, `cpu`
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU
- **ORT version**: 1.25.0

## Shootout table

| ONNX | Provider | PSNR (dB) | Drop vs FP32 | Latency (ms) | Speedup vs FP32 same-provider | Size (MB) |
|---|---|---:|---:|---:|---:|---:|
| `edsr_fp16.onnx` | `tensorrt` | 27.437 | +0.002 | 1.28 +/- 0.06 | 2.57x faster | 2.63 |
| `edsr_fp16.onnx` | `cuda` | 27.438 | +0.001 | 4.05 +/- 0.99 | 1.30x faster | 2.63 |
| `edsr_fp16.onnx` | `cpu` | 27.439 | -0.000 | 50.55 +/- 2.81 | 1.03x slower | 2.63 |
| `edsr_fp32.onnx` | `tensorrt` | 27.439 | +0.000 | 3.28 +/- 0.77 | 1.00x faster | 5.24 |
| `edsr_fp32.onnx` | `cuda` | 27.439 | +0.000 | 5.28 +/- 2.23 | 1.00x faster | 5.24 |
| `edsr_fp32.onnx` | `cpu` | 27.439 | +0.000 | 49.17 +/- 2.10 | 1.00x faster | 5.24 |
| `edsr_int8_static.onnx` | `tensorrt` | 27.268 | +0.171 | 4.33 +/- 3.47 | 1.32x slower | 1.43 |
| `edsr_int8_static.onnx` | `cuda` | 27.267 | +0.171 | 6.57 +/- 1.70 | 1.24x slower | 1.43 |
| `edsr_int8_static.onnx` | `cpu` | 27.268 | +0.171 | 56.25 +/- 4.43 | 1.14x slower | 1.43 |

## How to read

- **PSNR** is the deploy-side accuracy: ONNX session output evaluated on the val set against HR ground truth. **Provider-invariant within rounding** -- if it differs much between CUDA and CPU for the same ONNX, that's a debug signal.
- **Drop vs FP32** uses the FP32 PSNR as baseline. Should match (within ~0.1 dB) the fake-quant prediction in `results/quantization/200ep_with_report/report.md`.
- **Latency** is forward-pass only. **Provider-specific**: INT8 ONNX often runs slower on CUDA EP than FP32 due to QDQ insertion + memcpy nodes; on CPU EP, INT8 typically wins because of VNNI / native INT8 instructions. **TensorRT EP is the right path for true GPU INT8 deploy** (not benchmarked here).
- **Speedup** is per provider: it answers "if I'm deploying on this hardware, what does each precision give me?" -- not "is X precision globally fastest".

