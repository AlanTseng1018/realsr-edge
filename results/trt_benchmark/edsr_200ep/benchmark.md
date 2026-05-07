# Native TensorRT Engine Benchmark

Engines built with the **TensorRT Python API** (not ORT TRT EP).
TRT fully fuses the graph — INT8 uses native INT8 tensor cores,
FP16 uses FP16 tensor cores. This is the correct way to measure
TRT INT8 latency on consumer NVIDIA hardware (QDQ ONNX fed to ORT
TRT EP does not fuse and shows no INT8 gain — see deploy_summary.md
Section 8 for the failure-mode trace).

> **Scope.** This is a **consumer GPU reference benchmark on a 96x96 LR per-tile
> input**, used to (a) validate the native-TRT path works end-to-end with
> calibrator-based INT8, (b) confirm the ORT-TRT-EP INT8 paradox is a backend
> issue not a fundamental INT8 limitation. **It is not a TV product latency
> measurement.** Real TV SoC NPU latency would be measured via the vendor
> SDK on a dev board; the per-precision *fidelity* numbers (PSNR drops) port
> across hardware, the absolute latency does not.

## What was tested

- **Generated**: 2026-05-04T16:15:16
- **ONNX folder**: `results\onnx_exports\edsr_200ep`
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` (100 images)
- **Latency input shape**: `[1, 3, 96, 96]` (20 warmup + 100 timed iters) -- 96x96 LR matches the EDSR-baseline training patch size; this is **per-tile** latency, not full-frame
- **TensorRT version**: 10.16.1.11
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU (consumer Ampere, sm86) -- **NOT** a TV SoC NPU

## Results

| Precision | ONNX source | PSNR (dB) | Drop vs FP32 | Latency (ms) | Speedup vs FP32 | Engine size (MB) |
|---|---|---:|---:|---:|---:|---:|
| `FP32` | `edsr_fp32.onnx` | 27.439 | +0.000 | 3.46 +/- 0.22 | 1.00x faster | 9.53 |
| `FP16` | `edsr_fp32.onnx` | 27.438 | +0.000 | 1.50 +/- 0.19 | 2.31x faster | 2.92 |
| `INT8` | `edsr_fp32.onnx` | 27.357 | +0.082 | 1.93 +/- 0.07 | 1.80x faster | 1.63 |

## INT8 notes

INT8 engine is built from `edsr_fp32.onnx` + `IInt8EntropyCalibrator2`.
TRT calibrates activation ranges on 64 val-set LR patches, then builds
native INT8 kernels with full graph fusion.  QDQ ONNX is **not** used
because TRT 10 rejects INT32 bias dequantize nodes produced by
`onnxruntime.quantization.quantize_static`.

## Why FP16 wins INT8 here, but INT8 is still the right TV-NPU target

On this consumer Ampere GPU (sm86) FP16 (1.50 ms) beats INT8 (1.93 ms)
because FP16 tensor cores already saturate on a 1.4M-param SR model at
96x96 LR -- there is no compute headroom for INT8 to fill. On TV SoC
NPU silicon the picture inverts: the NPU has dedicated INT8 MAC arrays
and often *no* FP16 path at all, the workload is memory-bound (where
INT8's 4x weight compression matters), and the dataflow is designed
around INT8 conv. The "FP16 beats INT8" finding here is hardware-specific
to consumer Ampere on small SR models -- it is **not** evidence against
INT8 deployment on NPU.

