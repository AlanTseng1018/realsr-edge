# TensorRT Profiling Report

- **Generated**: 2026-05-05T10:36:32
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU
- **Bench shape**: [1, 3, 96, 96]

## Summary

| Precision | Latency (ms) | Achieved GFLOPS | Arith. Intensity | Ridge Point | Region |
|---|---:|---:|---:|---:|---|
| `FP32` | 3.17 | 8007.4 | 310.87 | 104.5 | **compute-bound** |
| `FP16` | 1.38 | 18381.9 | 617.55 | 207.6 | **compute-bound** |
| `INT8` | 0.91 | 27946.3 | 1218.67 | 409.7 | **compute-bound** |

> **How to read**: if Achieved GFLOPS << Ridge Point, the model is
> memory-bound (bottleneck = DRAM bandwidth). If Achieved GFLOPS ~= Peak,
> the model is compute-bound. Both are ceiling-limited; everything else
> (e.g. kernel launch overhead) shows as Achieved GFLOPS << both ceilings.

## Kernel breakdown

### FP32

- Compute kernels : 0.0%
- Memory transfer  : 4.1%
- Other / overhead : 95.9%

Top CUDA kernels (by total device time):

| Kernel | Calls | Avg (us) | Share |
|---|---:|---:|---:|
| `trt_infer_FP32` | 20 | 3314.7 | 92.9% |
| `aten::copy_` | 60 | 48.9 | 4.1% |
| `aten::detach` | 20 | 45.3 | 1.3% |
| `detach` | 20 | 19.9 | 0.6% |
| `aten::lift_fresh` | 20 | 15.4 | 0.4% |
| `aten::to` | 20 | 10.3 | 0.3% |
| `aten::resolve_conj` | 20 | 9.7 | 0.3% |
| `aten::resolve_neg` | 20 | 7.6 | 0.2% |

### FP16

- Compute kernels : 0.0%
- Memory transfer  : 6.0%
- Other / overhead : 94.0%

Top CUDA kernels (by total device time):

| Kernel | Calls | Avg (us) | Share |
|---|---:|---:|---:|
| `trt_infer_FP16` | 20 | 1499.0 | 91.3% |
| `aten::copy_` | 60 | 32.9 | 6.0% |
| `aten::detach` | 20 | 21.3 | 1.3% |
| `detach` | 20 | 7.9 | 0.5% |
| `aten::lift_fresh` | 20 | 5.3 | 0.3% |
| `aten::resolve_conj` | 20 | 3.8 | 0.2% |
| `aten::to` | 20 | 3.2 | 0.2% |
| `aten::resolve_neg` | 20 | 2.7 | 0.2% |

### INT8

- Compute kernels : 0.0%
- Memory transfer  : 8.4%
- Other / overhead : 91.6%

Top CUDA kernels (by total device time):

| Kernel | Calls | Avg (us) | Share |
|---|---:|---:|---:|
| `trt_infer_INT8` | 20 | 1158.6 | 85.8% |
| `aten::copy_` | 60 | 37.8 | 8.4% |
| `aten::detach` | 20 | 34.0 | 2.5% |
| `detach` | 20 | 15.2 | 1.1% |
| `aten::lift_fresh` | 20 | 14.5 | 1.1% |
| `aten::to` | 20 | 6.3 | 0.5% |
| `aten::resolve_neg` | 20 | 4.8 | 0.4% |
| `aten::resolve_conj` | 20 | 3.3 | 0.2% |

## Interpretation

**FP32**: Achieved 8007.4 GFLOPS, ridge point 104.5 GFLOPS — **compute-bound**.
**FP16**: Achieved 18381.9 GFLOPS, ridge point 207.6 GFLOPS — **compute-bound**.
**INT8**: Achieved 27946.3 GFLOPS, ridge point 409.7 GFLOPS — **compute-bound**.

*See `roofline.png` for the visual summary.*
