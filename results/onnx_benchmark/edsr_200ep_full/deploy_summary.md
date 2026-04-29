# Deployment Performance Summary

Aggregated view of the ONNX runtime benchmark, organized so a deploy-team reader can answer the three questions:

1. **What latency / accuracy do I get at each precision?**
2. **How does my chosen runtime affect the answer?**
3. **What should I deploy on my target hardware?**

## 1. Test configuration

- **Generated**: 2026-04-30T01:04:10
- **Source benchmark**: `results\onnx_exports\edsr_200ep`
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` (100 images, realistic degradation)
- **Latency input shape**: `(1, 3, 96, 96)` (10 warmup + 50 timed iters)
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU
- **ORT version**: 1.25.0
- **Available EPs**: `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`

## 2. Headline latency matrix (ms, lower is better)

| Precision \ Provider | `tensorrt` | `cuda` | `cpu` |
|---|---:|---:|---:|
| **FP32** | 3.28 +/- 0.77 | 5.28 +/- 2.23 | 49.17 +/- 2.10 |
| **FP16** | 1.28 +/- 0.06 | 4.05 +/- 0.99 | 50.55 +/- 2.81 |
| **INT8** | 4.33 +/- 3.47 | 6.57 +/- 1.70 | 56.25 +/- 4.43 |

## 3. Speedup vs FP32 (same provider)

Per cell: `latency(FP32 same-EP) / latency(this cell)`. **Bold** = faster than FP32 same EP.

| Precision \ Provider | `tensorrt` | `cuda` | `cpu` |
|---|---:|---:|---:|
| **FP32** | baseline | baseline | baseline |
| **FP16** | **2.57x faster** | **1.30x faster** | 1.03x slower |
| **INT8** | 1.32x slower | 1.24x slower | 1.14x slower |

## 4. Accuracy per precision (PSNR on val set)

PSNR is provider-invariant within float-rounding noise; we report the mean across providers per precision.

| Precision | mean PSNR (dB) | range across providers | drop vs FP32 |
|---|---:|---:|---:|
| **FP32** | 27.439 | 0.000 | +0.000 |
| **FP16** | 27.438 | 0.002 | +0.001 |
| **INT8** | 27.268 | 0.001 | +0.171 |

## 5. Per-provider deep dive

### `tensorrt`

| Precision | PSNR (dB) | Drop vs FP32 | Latency (ms) | Speedup vs FP32 | Build (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| FP32 | 27.439 | -0.000 | 3.28 +/- 0.77 | baseline | 4267 |  |
| FP16 | 27.437 | +0.002 | 1.28 +/- 0.06 | **2.57x faster** | 6127 |  |
| INT8 | 27.268 | +0.171 | 4.33 +/- 3.47 | 1.32x slower | 6035 |  |

### `cuda`

| Precision | PSNR (dB) | Drop vs FP32 | Latency (ms) | Speedup vs FP32 | Build (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| FP32 | 27.439 | +0.000 | 5.28 +/- 2.23 | baseline | 115 |  |
| FP16 | 27.438 | +0.001 | 4.05 +/- 0.99 | **1.30x faster** | 957 |  |
| INT8 | 27.267 | +0.171 | 6.57 +/- 1.70 | 1.24x slower | 143 |  |

### `cpu`

| Precision | PSNR (dB) | Drop vs FP32 | Latency (ms) | Speedup vs FP32 | Build (ms) | Notes |
|---|---:|---:|---:|---:|---:|---|
| FP32 | 27.439 | +0.000 | 49.17 +/- 2.10 | baseline | 47 |  |
| FP16 | 27.439 | -0.000 | 50.55 +/- 2.81 | 1.03x slower | 30 |  |
| INT8 | 27.268 | +0.171 | 56.25 +/- 4.43 | 1.14x slower | 94 |  |

## 6. Per-precision deep dive

### FP32

| Provider | PSNR (dB) | Latency (ms) | Size (MB) | Active EP | Notes |
|---|---:|---:|---:|---|---|
| `tensorrt` | 27.439 | 3.28 +/- 0.77 | 5.24 | `TensorrtExecutionProvider` |  |
| `cuda` | 27.439 | 5.28 +/- 2.23 | 5.24 | `CUDAExecutionProvider` |  |
| `cpu` | 27.439 | 49.17 +/- 2.10 | 5.24 | `CPUExecutionProvider` |  |

### FP16

| Provider | PSNR (dB) | Latency (ms) | Size (MB) | Active EP | Notes |
|---|---:|---:|---:|---|---|
| `tensorrt` | 27.437 | 1.28 +/- 0.06 | 2.63 | `TensorrtExecutionProvider` |  |
| `cuda` | 27.438 | 4.05 +/- 0.99 | 2.63 | `CUDAExecutionProvider` |  |
| `cpu` | 27.439 | 50.55 +/- 2.81 | 2.63 | `CPUExecutionProvider` |  |

### INT8

| Provider | PSNR (dB) | Latency (ms) | Size (MB) | Active EP | Notes |
|---|---:|---:|---:|---|---|
| `tensorrt` | 27.268 | 4.33 +/- 3.47 | 1.43 | `TensorrtExecutionProvider` |  |
| `cuda` | 27.267 | 6.57 +/- 1.70 | 1.43 | `CUDAExecutionProvider` |  |
| `cpu` | 27.268 | 56.25 +/- 4.43 | 1.43 | `CPUExecutionProvider` |  |

## 7. Deploy recommendation matrix

**Lowest latency on this hardware**: `FP16` on `tensorrt` -> 1.28 ms

| Target | Best precision | Provider | Reason |
|---|---|---|---|
| **NVIDIA Jetson / Orin / Drive** | FP16 (or INT8 on larger models) | TensorRT | Tensor Core FP16 saturates on small SR models; INT8 wins only for larger / batched workloads |
| **NVIDIA desktop edge** | FP16 | TensorRT | Same as Jetson reasoning |
| **x86 CPU server / edge** | FP32 (or INT8 on larger models) | ORT CPU | All precisions roughly equivalent for small models on CPU; VNNI INT8 wins on big models |
| **Mobile / TV SoC NPU** | INT8 | Vendor SDK | NPU silicon is INT8-native, memory-bound; this benchmark is reference only -- vendor SDK gives final numbers |

## 8. Notes and caveats

### Why INT8 isn't always faster on GPU

For this 1.37M-param SR model on a consumer Tensor Core GPU, FP16 outperforms INT8 because:

- Tensor Core FP16 saturates at small batch / small model sizes (no compute headroom for INT8 to fill).
- INT8 adds Q/DQ ops + scale arithmetic; the overhead dominates for small graphs.
- INT8's main lever -- 4× weight compression / memory bandwidth -- is only decisive on memory-bound hardware (NPUs, mobile DSPs). On Tensor Core, compute is rarely the bottleneck for small SR models.

**INT8 expected to win** on: larger models (5M+ params), higher batch sizes, NPU silicon, or 4K input resolution (where memory bandwidth matters).

### TensorRT INT8 calibration must be symmetric

ORT's `quantize_static` defaults to **asymmetric** (non-zero zero point). TensorRT EP rejects that with "Non-zero zero point is not supported". The export pipeline forces `ActivationSymmetric=True` + `WeightSymmetric=True` + ``quant_pre_process`` to make the INT8 ONNX TRT-compatible. The trade-off: ~0.05 dB more PSNR drop than asymmetric.

### ORT CUDA EP + INT8 anti-pattern

ORT's CUDA EP doesn't have native INT8 conv kernels for QDQ format. It runs Q/DQ ops on CPU, conv on GPU FP32, and inserts Memcpy nodes between. Result is slower than FP32 CUDA. The fix is using TensorRT EP (this benchmark shows it works) or vendor NPU SDKs.

## 9. Cross-references

- Raw benchmark: `results\onnx_benchmark\edsr_200ep_full/benchmark.md` and `benchmark.csv`
- Accuracy analysis (PyTorch fake-quant): `results/quantization/200ep_with_report/report.md`
- Calibration scheme ablation: `results/quantization/calibration_ablation/calibration_ablation.md`
- ONNX export: `results/onnx_exports/edsr_200ep/README.md`
- Deploy methodology framework: `learning/deployment_methodology.md`
- Lessons learned: `learning/deployment_lessons_learned.md`
