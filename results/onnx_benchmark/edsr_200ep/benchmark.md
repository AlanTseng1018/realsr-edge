# ONNX Deployment Benchmark

Single execution node of the ONNX runtime benchmark. Each row is one (ONNX file, ORT execution provider) pair, evaluated on the same val set with the same input shape for latency.

## What was tested

- **Generated**: 2026-04-28T17:11:11
- **ONNX folder**: `results\onnx_exports\edsr_200ep`
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` (100 images, realistic degradation, LR patch 96x96)
- **Latency input shape**: `[1, 3, 96, 96]` (10 warmup + 50 timed iters)
- **Providers tested**: `cuda`, `cpu`
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU
- **ORT version**: 1.25.0

## Shootout table

| ONNX | Provider | PSNR (dB) | Drop vs FP32 | Latency (ms) | Speedup vs FP32 same-provider | Size (MB) |
|---|---|---:|---:|---:|---:|---:|
| `edsr_fp16.onnx` | `cuda` | 27.438 | +0.001 | 3.36 +/- 0.13 | 1.64x faster | 2.63 |
| `edsr_fp16.onnx` | `cpu` | 27.439 | -0.000 | 45.94 +/- 2.93 | 1.01x slower | 2.63 |
| `edsr_fp32.onnx` | `cuda` | 27.439 | +0.000 | 5.52 +/- 2.41 | 1.00x faster | 5.24 |
| `edsr_fp32.onnx` | `cpu` | 27.439 | +0.000 | 45.28 +/- 3.42 | 1.00x faster | 5.24 |
| `edsr_int8_static.onnx` | `cuda` | 27.322 | +0.117 | 6.29 +/- 1.53 | 1.14x slower | 1.41 |
| `edsr_int8_static.onnx` | `cpu` | 27.321 | +0.118 | 53.01 +/- 2.70 | 1.17x slower | 1.41 |

## How to read

- **PSNR** is the deploy-side accuracy: ONNX session output evaluated on the val set against HR ground truth. **Provider-invariant within rounding** -- if it differs much between CUDA and CPU for the same ONNX, that's a debug signal.
- **Drop vs FP32** uses the FP32 PSNR as baseline. Should match (within ~0.1 dB) the fake-quant prediction in `results/quantization/200ep_with_report/report.md`.
- **Latency** is forward-pass only. **Provider-specific**: INT8 ONNX often runs slower on CUDA EP than FP32 due to QDQ insertion + memcpy nodes; on CPU EP, INT8 typically wins because of VNNI / native INT8 instructions. **TensorRT EP is the right path for true GPU INT8 deploy** (not benchmarked here).
- **Speedup** is per provider: it answers "if I'm deploying on this hardware, what does each precision give me?" -- not "is X precision globally fastest".

