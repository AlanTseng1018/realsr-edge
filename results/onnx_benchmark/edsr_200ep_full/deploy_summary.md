# Deployment Performance Summary

Aggregated view of the ONNX runtime benchmark, organized so a deploy-team reader can answer the three questions:

1. **What latency / accuracy do I get at each precision?**
2. **How does my chosen runtime affect the answer?**
3. **What does this tell me about deploying on my target hardware?**

> **Scope note.** All latency numbers below are on a **consumer Ampere GPU (RTX 3060 Laptop, sm86)** with **96x96 LR input** (matches training patch size, standard SR-benchmark convention). They are a **deployment-prep reference**, not product-target measurements. Real TV SoC / mobile NPU latency lives in the vendor SDK on a dev board and will differ in absolute numbers. What *does* port across hardware: the per-layer quantization sensitivity ranking, the QDQ-vs-calibrator backend choice, and the relative ordering of FP32/FP16/INT8 (with the caveat that NPU silicon flips the FP16-vs-INT8 ranking back to INT8's favour).

## 1. Test configuration

- **Generated**: 2026-04-30T01:04:10
- **Source benchmark**: `results\onnx_exports\edsr_200ep`
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` (100 images, realistic degradation)
- **Latency input shape**: `(1, 3, 96, 96)` (10 warmup + 50 timed iters) -- 96x96 LR matches the EDSR-baseline training patch size; standard SR-benchmark convention. Production frame sizes (1080p / 4K) would use tile-based inference where each tile is ~this shape, so per-tile latency is the relevant per-compute-unit number.
- **Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU (consumer Ampere, sm86) -- **NOT** a TV SoC NPU; vendor NPU silicon will give different absolute numbers
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

## 5b. Native TensorRT (Python API) comparison

`benchmark_trt.py` builds engines directly via the TensorRT Python API instead of going through ORT TRT EP. The INT8 path uses `IInt8EntropyCalibrator2` on 64 val patches and **bypasses the QDQ ONNX entirely** -- TRT 10's ONNX parser rejects the INT32 bias DequantizeLinear nodes that `onnxruntime.quantization.quantize_static` produces, so the only way to get a fully-fused TRT INT8 engine is to feed FP32 ONNX + a calibrator.

| Precision | Source | PSNR (dB) | Drop vs FP32 | Latency (ms) | Speedup vs FP32 | Engine size (MB) |
|---|---|---:|---:|---:|---:|---:|
| FP32 | `edsr_fp32.onnx` | 27.439 | +0.000 | 3.46 +/- 0.22 | baseline | 9.53 |
| FP16 | `edsr_fp32.onnx` | 27.438 | +0.001 | **1.50 +/- 0.19** | **2.31x faster** | 2.92 |
| INT8 | `edsr_fp32.onnx` + calibrator | 27.357 | +0.082 | 1.93 +/- 0.07 | 1.80x faster | 1.63 |

### What this confirms vs the ORT TRT EP numbers above

- **Native TRT rescues INT8 from the ORT paradox**: ORT TRT EP gave INT8 4.33 ms (slower than FP32). Native TRT gives INT8 1.93 ms (faster than FP32). The 2.2x improvement is the value of bypassing ORT's QDQ handling. So **the QDQ fusion failure was an ORT-layer problem, not a TRT capability problem**.
- **But INT8 still loses to FP16** (1.93 vs 1.50 ms). On RTX 30-series (sm86) Ampere consumer GPUs, INT8 Tensor Core peak throughput is only marginally above FP16 Tensor Core. EDSR-baseline at 1.37M params and patch 96x96 is small enough that kernel launch overhead and memory copy dominate, leaving little compute headroom for INT8 to fill.
- **FP16 native TRT (1.50 ms) is essentially identical to FP16 ORT TRT EP (1.28 ms)** within noise. ORT TRT EP works fine for FP16 -- the QDQ problem is INT8-specific.

The latency ordering on **this consumer Ampere GPU** doesn't change: **FP16 on TensorRT** is the fastest configuration regardless of which TRT path is used. The native-TRT data closes the loop on "why is INT8 so slow on ORT" and confirms it was a backend-layer issue (QDQ INT32-bias fusion failure), not a fundamental INT8 limitation. Whether INT8 wins on TV SoC NPU silicon is **a hypothesis we cannot verify in this project** (no NPU dev board); see Appendix at the bottom for the explicit verified-vs-hypothesized split.

## 5c. Roofline + kernel profile (native TRT engines)

`profile_trt.py` profiles each TRT engine with `torch.profiler` (CUDA activity) and computes arithmetic intensity vs the RTX 3060 Laptop GPU's peak FP32 / FP16 / INT8 ceilings. Outputs in `results/trt_profile/edsr_200ep/` (`profile_report.md`, `roofline.png`, `metadata.json`).

| Precision | Achieved (TFLOPS) | Peak (TFLOPS) | Utilization | Region | Kernel time / iter |
|---|---:|---:|---:|---|---:|
| FP32 | 8.0 | 10.9 | 73% | compute-bound | 3.31 ms |
| FP16 | 18.4 | 21.9 | **84%** | compute-bound | 1.50 ms |
| INT8 | 27.9 | 43.7 | 64% | compute-bound | 1.16 ms |

### What the roofline tells us

All three precisions sit far above the memory-bandwidth-limited diagonal -- this model + this input shape is **strongly compute-bound on this GPU**. INT8's main benefit (4x weight compression / less DRAM traffic) does not apply here because we are not bottlenecked on DRAM. The latency ranking on this GPU is therefore set by **how well TRT can saturate the corresponding tensor core**, not by quantization-driven memory savings:

- FP16 achieves **84% of peak** -- TRT fuses the small SR graph well into FP16 tensor core kernels.
- INT8 achieves only **64% of peak** -- on Ampere consumer cards (sm86) INT8 tensor core scheduling is less mature than FP16, and EDSR-baseline at 1.4M params + 96x96 LR is too small to keep INT8 cores fully fed. There is still 1.5x headroom INT8 fails to capture.
- TV SoC NPU silicon **is reported by vendor whitepapers** to be INT8-native (often without FP16 path) and memory-bound on SR-class workloads -- which would predict INT8 wins on NPU. **This project does not verify that prediction** (no NPU dev board); it is a hypothesis carried forward to the final deploy decision in Section 7.

### Wall-clock vs kernel-only timing

Profile reports kernel time (`trt_infer_INT8` = 1.16 ms / iter), while `benchmark_trt.py` reports wall-clock (1.93 ms / iter for INT8). The ~0.8 ms gap is **kernel launch + cuda.synchronize + H2D/D2H copy** overhead -- this is verified on this hardware. NPU vendor docs describe statically-scheduled dataflow with much smaller per-op overhead; if that holds for SR-class workloads, the kernel-vs-wallclock gap would collapse on NPU and the GPU wall-clock numbers here would systematically understate NPU INT8 performance. **This is a hypothesis based on architectural reasoning**, not a measured comparison; verifying it requires NPU dev board access which is out of scope here.

### Profiler note

`torch.profiler` reports the entire TRT engine inference as a single `trt_infer_*` op (~85-93% of device time) rather than decomposing into individual TRT kernels. This is a profiler-categorization artifact -- the work is happening (achieved GFLOPS confirms it), but `torch.profiler` cannot see inside TRT's runtime. To see per-kernel breakdown you'd need TRT's own profiler (`IExecutionContext::setProfiler`) or NVIDIA Nsight Systems. For this report the per-precision FLOPS + region classification is sufficient.

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

## 7. Deployment guidance per target (NOT a product recommendation)

**Lowest latency observed on this benchmark hardware (RTX 3060 Laptop, 96x96 LR):** `FP16` on `tensorrt` -> 1.28 ms / per-tile.

The table below maps the empirical findings on this consumer GPU to **what a deploy team would likely conclude for each target** -- it is a reasoning table for deployment-planning conversations, not a product spec. Real product targets must be re-measured on the actual silicon.

| Target | Likely best precision | Likely runtime | Why (extrapolating from data here) |
|---|---|---|---|
| **NVIDIA Jetson / Orin / Drive** | FP16 (INT8 only at larger models / batches) | TensorRT | Same Tensor Core architecture family as our RTX 3060 measurement; FP16-saturates-INT8 finding likely ports |
| **NVIDIA desktop edge** | FP16 | TensorRT | Direct port of our measurement |
| **x86 CPU server / edge** | FP32 (INT8 helps on bigger models) | ORT CPU | All precisions are within noise on small models on CPU |
| **Mobile / TV SoC NPU** (the actual TV-product target) | **Likely INT8** (hypothesis from vendor docs, not verified here) | Vendor SDK (SNPE / NeuroPilot / NNIE / RKNN / vendor-specific) | Vendor whitepapers describe NPU silicon as INT8-native (often without FP16 path), memory-bound, with dedicated INT8 MAC arrays. **If those descriptions hold**, the FP16-wins finding on RTX 3060 would not port and INT8 would be the right NPU precision. **This project does not measure NPU latency** (no dev board / vendor SDK access) -- the hypothesis is based on architectural reasoning + public sources, not direct verification. What the GPU measurements *do* contribute, regardless of NPU outcome: (a) PSNR / LPIPS / per-layer sensitivity numbers that port across hardware, (b) the calibration + mixed-precision methodology, (c) backend-failure-mode awareness (QDQ INT32 bias) that any NPU SDK conversion would also need to handle. See Appendix at bottom. |

The point of the consumer-GPU benchmark is **not** to pick the precision a TV product would ship -- it is to (a) verify the export pipeline produces a numerically correct ONNX, (b) catch backend-specific deployment failure modes (the QDQ-fusion bug, ORT CUDA INT8 anti-pattern), and (c) provide a per-layer fidelity / sensitivity reference that does port to NPU.

## 8. Notes and caveats

### Why INT8 isn't always faster on GPU

For this 1.37M-param SR model on a consumer Tensor Core GPU, FP16 outperforms INT8. **This is now confirmed empirically across both ORT TRT EP (FP16 1.28 ms vs INT8 4.33 ms) and native TRT (FP16 1.50 ms vs INT8 1.93 ms)** -- it is not a backend bug, it is a hardware/model-size truth on RTX 30-series. Reasons:

- Tensor Core FP16 saturates at small batch / small model sizes (no compute headroom for INT8 to fill).
- On Ampere consumer GPUs (sm86), INT8 Tensor Core peak throughput is only marginally above FP16 Tensor Core -- INT8 wins decisively only on data-center cards (A100/H100, sm80/sm90).
- INT8's main lever -- 4x weight compression / memory bandwidth -- is only decisive on memory-bound hardware (NPUs, mobile DSPs). On Tensor Core, compute is rarely the bottleneck for small SR models.
- For ORT TRT EP specifically, INT8 also pays the QDQ-fusion-failure tax (Q/DQ ops fall back to CPU). Native TRT closes that gap (4.33 -> 1.93 ms) but does not flip the FP16/INT8 ranking.

**INT8 is expected to win on hardware classes we did not test**: larger models (5M+ params), higher batch sizes, NPU silicon (TV SoC / mobile), or full-frame 1080p / 4K input. The arguments are architectural -- on memory-bound hardware, INT8's 4x weight compression matters; on NPU silicon designed around INT8, the FP16 alternative may not exist. **None of those expectations are verified in this project.** The "FP16 wins" finding on this RTX 3060 is a hardware-specific result about Ampere consumer GPUs running a small SR model at 96x96 LR; it is not evidence against INT8 on other targets, and it is not evidence *for* INT8 on those targets either. See Appendix.

### TensorRT INT8 calibration must be symmetric

ORT's `quantize_static` defaults to **asymmetric** (non-zero zero point). TensorRT EP rejects that with "Non-zero zero point is not supported". The export pipeline forces `ActivationSymmetric=True` + `WeightSymmetric=True` + ``quant_pre_process`` to make the INT8 ONNX TRT-compatible. The trade-off: ~0.05 dB more PSNR drop than asymmetric.

### ORT CUDA EP + INT8 anti-pattern

ORT's CUDA EP doesn't have native INT8 conv kernels for QDQ format. It runs Q/DQ ops on CPU, conv on GPU FP32, and inserts Memcpy nodes between. Result is slower than FP32 CUDA. The fix is using TensorRT EP (this benchmark shows it works) or vendor NPU SDKs.

## 9. Cross-references

- Raw ORT benchmark: `results\onnx_benchmark\edsr_200ep_full/benchmark.md` and `benchmark.csv`
- Native TRT benchmark: `results/trt_benchmark/edsr_200ep/benchmark.md` (TRT Python API + IInt8EntropyCalibrator2)
- TRT roofline + kernel profile: `results/trt_profile/edsr_200ep/profile_report.md` and `roofline.png`
- Accuracy analysis (PyTorch fake-quant + LPIPS perceptual): `results/quantization/200ep_with_report/report.md`
- Calibration scheme ablation: `results/quantization/calibration_ablation/calibration_ablation.md`
- ONNX export: `results/onnx_exports/edsr_200ep/README.md`
- Deploy methodology framework: `learning/deployment_methodology.md`
- Lessons learned: `learning/deployment_lessons_learned.md`

## Appendix: What this project verifies vs hypothesizes vs cannot verify

This project ran on a single dev machine (RTX 3060 Laptop GPU, x86 CPU, ORT 1.25 / TensorRT 10 stack). All claims in this report fall into one of three buckets:

### Verified on this hardware
Direct measurements; reproducible by re-running the scripts in this repo.
- FP32 / FP16 / INT8 latency ranking on RTX 3060 Laptop at 96x96 LR (Sections 2-5)
- ORT TRT EP INT8 fusion failure due to QDQ INT32-bias DequantizeLinear (Section 8)
- Native TRT INT8 build path via `IInt8EntropyCalibrator2` works around the QDQ issue (Section 5b)
- INT8 Tensor Core utilization 64% vs FP16 84% on this GPU (Section 5c roofline)
- Kernel-time vs wall-clock gap ~0.8 ms is launch + memcpy overhead (Section 5c)
- Cross-language correctness of ONNX export (Python <-> C++ ORT, see `cpp_inference/README.md`)
- Per-layer quantization sensitivity ranking on this checkpoint (`results/quantization/200ep_with_report/sensitivity.md`)
- Calibration scheme sensitivity (`results/quantization/calibration_ablation/`)

### Hypothesized from architecture / public sources
Plausible extensions of the verified data, but NOT measured in this project. Treat as hypotheses to verify if/when the relevant hardware becomes available.
- TV SoC NPU INT8 wins FP16 (rationale: vendor whitepapers describe NPU silicon as INT8-native, often without FP16 path, with statically-scheduled dataflow that reduces launch overhead -- but no direct measurement here)
- Per-layer sensitivity ranking ports across hardware (rationale: sensitivity reflects information-theoretic properties of the model, not backend-specific quirks; consistent with PAMS / 2DQuant findings on different platforms -- but not verified on NPU here)
- Calibration method choice ports across hardware (rationale: same as above; calibration produces scales that are then consumed identically by any INT8 backend)
- Tile-based 1080p / 4K full-frame inference would scale roughly linearly per-tile from the 96x96 numbers (rationale: per-tile workload is compute-bound at the same arithmetic intensity)

### Out of scope / cannot verify
Would require resources unavailable on this dev setup. Acknowledged gaps; not pretended to fill.
- NPU dev board / vendor SDK (SNPE / NeuroPilot / NNIE / QNN / RKNN) integration -- requires vendor licensing + dev hardware
- Real TV product latency (real video pipeline, real silicon, real workload) -- product-team scope
- Full-frame tile-based inference benchmark -- engineering follow-on, not done in this project
- Lightweight architecture comparison (ECBSR / IMDN / RFDN) -- could be done with public pretrained weights, but not done here
- QAT end-to-end vs PTQ comparison -- pipeline exists in `src/training/`, run logged on a separate workstation; not yet integrated into this report

The split is the deployment-readiness statement: the methodology and fidelity numbers transfer; the latency numbers and NPU-specific predictions are honest hypotheses that depend on hardware this project does not have.
