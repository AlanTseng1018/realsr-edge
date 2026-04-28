# Quantization Terminology — Cheat Sheet

A reference for the vocabulary used in quantization / precision analysis,
organized so that it doubles as both:

1. **Implementation index** — which terms are already realized in this
   project's code, and where to look (links below).
2. **Portfolio / interview vocabulary** — the right English term + a
   one-line definition for each concept, so verbal explanations land
   without paraphrasing.

Terms marked **✓** appear by name (or by mechanism) in this project's code,
report, or the [learning/sr_design_thinking.md](sr_design_thinking.md)
notes. Treat those as the priority set; the rest are "you'll see this in
papers / vendor docs."

---

## 1. Precision formats (data types)

| Term | 中文 | Definition |
|---|---|---|
| **FP32** ✓ | 單精度浮點 | 32-bit IEEE 754 float. SR training default. |
| **FP16** ✓ | 半精度浮點 | 16-bit float (5-bit exponent, 10-bit mantissa). Narrow exponent → easy to overflow on activations with large magnitude. |
| **BF16** ✓ | brain float 16 | 16-bit float (**8-bit exponent**, 7-bit mantissa). Same dynamic range as FP32, half the precision. Google TPU origin. |
| **FP8 (E4M3, E5M2)** | 8-bit float | New on NVIDIA Hopper / Blackwell. Mostly LLM-relevant for now. |
| **INT8** ✓ | 8-bit 整數 | The mainstream quantization target for edge AI. Range -128..127 (symmetric) or 0..255 (asymmetric). |
| **INT4** | 4-bit 整數 | LLM mainstream (GPTQ / AWQ). Usually too aggressive for SR — PSNR loss large. |
| **Mixed Precision** ✓ | 混合精度 | Different layers in different precisions (e.g. head/tail in FP16, body in INT8). What our `report.md` recommends. |

---

## 2. Quantization methods (when, against what data)

| Term | 中文 | Definition |
|---|---|---|
| **PTQ** (Post-Training Quantization) ✓ | 訓練後量化 | Quantize a trained model without retraining. The path our `analyze.py` exercises. |
| **PTQ Static** ✓ | 靜態 PTQ | Activation scales are computed once from a calibration set, then **frozen at deploy time**. Standard for edge inference. |
| **PTQ Dynamic** | 動態 PTQ | Activation scales recomputed at every forward pass. Lighter to set up, slower at runtime. |
| **QAT** (Quantization-Aware Training) | 量化感知訓練 | Inject fake-quant ops into the training graph; the model **learns** to be robust to quantization. Use when PTQ accuracy is insufficient. |
| **Fake Quantization** ✓ | 偽量化 | `quantize -> dequantize` round-trip done in float math. Simulates the precision loss of INT8 deploy without needing an INT8 backend. **Our `CalibratingConv2d` is a fake-quant wrapper.** |
| **Q-DQ Nodes** | Q-DQ 節點 | Explicit `QuantizeLinear` / `DequantizeLinear` ops in the ONNX graph. The way ONNX represents fake-quant statically for downstream tools. |

---

## 3. Quantization schemes (the math layout)

| Term | 中文 | Definition |
|---|---|---|
| **Symmetric** ✓ | 對稱量化 | Quantization grid centered at 0. No zero-point. Standard for **weight** quantization. |
| **Asymmetric** | 非對稱量化 | Quantization grid offset by a zero-point. Better for one-sided activation distributions (post-ReLU). |
| **Per-Tensor** ✓ | 整個 tensor 共用一個 scale | One scalar `scale` for the whole tensor. Simple, hardware-friendly. Standard for **activations**. |
| **Per-Channel** ✓ | 每個 output channel 各自的 scale | One `scale` per output channel of a Conv2d weight. Higher fidelity, slightly more compute. **Standard for weights**. |
| **Per-Group / Group-wise** | 每 N 個 channel 一組 scale | Compromise between per-tensor and per-channel. Common in LLM weight quantization. |
| **Uniform Quantization** ✓ | 均勻量化 | Quantization grid is equally spaced. Almost always what people mean. |
| **Non-uniform / Log Quantization** | 非均勻 / 對數量化 | Grid is unevenly spaced (e.g. log scale). Useful for long-tailed distributions. |

---

## 4. Calibration (how scales are picked)

| Term | 中文 | Definition |
|---|---|---|
| **Calibration Set** ✓ | 校準集 | A small batch of representative inputs. Forward through the FP32 model with quant observers attached, collect per-tensor statistics. **We use 8 batches × 8 = 64 LR samples**. |
| **Max-Abs / Min-Max** ✓ | 最大絕對值 / 極值法 | Take the largest `\|x\|` seen during calibration as the saturation level. **Simplest, worst-case** — what we use. |
| **Percentile Clipping** | 百分位截斷 | Use the 99.9th-percentile instead of the absolute max, so a single outlier doesn't blow up the scale. |
| **KL-divergence Calibration** | KL 散度校準 | TensorRT's default. Choose the scale that minimizes KL-divergence between FP32 and INT8 activation histograms. |
| **MSE Calibration** | 最小化均方誤差 | Pick the scale that minimizes `mean((x - q-dq(x))^2)`. |
| **Entropy Calibration** | 熵校準 | Synonym for KL-divergence calibration (the information-theoretic angle). |

---

## 5. Analysis vocabulary (what the numbers mean)

| Term | 中文 | Definition |
|---|---|---|
| **Sensitivity Analysis** ✓ | 敏感度分析 | Measure each layer's individual contribution to the total quantization-induced PSNR drop. **What `analyze.py:run_sensitivity` does**. |
| **Quantization-Critical Layer** ✓ | 量化關鍵層 | A layer whose individual quantization contributes disproportionately to total accuracy loss. In our model: `tail`, `upsampler.0`, `head`. |
| **Quantization Error** | 量化誤差 | `‖x - q-dq(x)‖`. Per-tensor, per-image, per-layer — useful for visual inspection of what each layer's quantization is doing. |
| **SQNR** (Signal-to-Quantization-Noise Ratio) | 信雜訊比 | `10 * log10(var(x) / var(x - q-dq(x)))`. A finer-grained accuracy proxy than PSNR drop. |
| **STE** (Straight-Through Estimator) | 直通估計器 | A trick to make `round()` "differentiable" — backward pass treats `round(x) = x` (identity), forward pass quantizes. Required for QAT. |
| **Long-Tail Distribution** ✓ | 長尾分佈 | Activation distribution where most values are small, a few are large. Common in SR (no BN to constrain magnitudes). Hostile to max-abs calibration. |
| **Outlier Channel** | 異常通道 | A weight or activation channel whose range is much larger than its neighbors. Drives the case for per-channel (vs. per-tensor) scales. |

---

## 6. SR-specific quantization vocabulary

These come up in the SR quantization literature (PAMS, 2DQuant, DAQ,
Tu et al. CVPR 2023):

| Term | 中文 | Definition |
|---|---|---|
| **Activation Range Variability** | 激活值範圍變動性 | Dynamic range of activations differs per input image. One of three key SR-quantization difficulties (Tu et al.). |
| **Wide Dynamic Activation Range** | 動態範圍寬 | Without BN, SR activation magnitudes span more orders of magnitude than classification. Second of the three difficulties. |
| **Learnable Activation Clamp** (PAMS, ECCV 2020) | 可學習激活截斷 | Make the activation clipping threshold a trained parameter — equivalent to PyTorch's `FakeQuantize`. A QAT variant. |
| **Two-Stage PTQ** (2DQuant, NeurIPS 2024) | 兩階段 PTQ | Coarse PTQ first, then distillation-based refinement. State-of-art SR PTQ recipe. |
| **Quantization-Aware Distillation** | 量化感知蒸餾 | Use the FP32 model as a teacher; train the INT8 (student) to match its outputs. |
| **Per-Channel Weight Quantization for SR** (DAQ, WACV 2022) | 逐通道權重量化 | A SR-specific paper arguing per-channel is essential for SR weight quantization (we do this by default). |

---

## 7. Deployment / hardware vocabulary

| Term | 中文 | Definition |
|---|---|---|
| **Execution Provider (EP)** ✓ | 執行提供器 | ONNX Runtime's plug-in backend (`CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`, ...). The same `.onnx` runs on any EP. |
| **Operator Fusion** | 算子融合 | Combine multiple ops into one kernel (Conv+ReLU+Add). The main thing `torch.compile` and most NN compilers do. |
| **Graph Optimization** | 圖優化 | Constant folding, dead code elimination, layout transforms — all the rewrites done before kernel codegen. |
| **Memory-Bound vs Compute-Bound** | 記憶體頻寬限 vs 算力限 | Whether your model is bottlenecked by DRAM bandwidth or by FLOPs. Most edge SR is memory-bound, which is why INT8 (4× smaller weights) helps so much. |
| **Roofline Model** | 屋頂線模型 | Plot of "achievable performance vs arithmetic intensity" — the canonical way to argue memory- vs compute-bound. |
| **Tensor Core / VNNI** | 張量核 / 向量神經網路指令 | Hardware-level INT8 accelerators (NVIDIA tensor cores, Intel VNNI, ARM dotprod). The actual reason real INT8 is faster than FP32. |
| **Op Support Matrix** | 算子支援表 | Vendor-published table of which ops are supported on the NPU vs fall back to CPU. **The first thing to check before targeting a vendor**. |

---

## 8. How these map to this project's code

So the vocabulary doesn't sit in the abstract:

| Term | Where in code |
|---|---|
| Symmetric per-tensor (activation) | [`fake_quant.py:per_tensor_scale`](../src/quantization/fake_quant.py) |
| Symmetric per-channel (weight) | [`fake_quant.py:per_channel_scale`](../src/quantization/fake_quant.py) |
| Fake-quant wrapper | [`fake_quant.py:CalibratingConv2d`](../src/quantization/fake_quant.py) |
| Calibration (max-abs) | The `mode='calibrate'` branch of `CalibratingConv2d.forward` |
| PTQ static path | `mode='quantize'` branch + `analyze.py:calibrate_int8` |
| Per-layer sensitivity sweep | [`analyze.py:run_sensitivity`](../src/quantization/analyze.py) |
| Mixed-precision recommendation | [`analyze.py:write_report_md`](../src/quantization/analyze.py) (the `top_n` / `other_drop_sum` block) |
| Quantization-critical layers | The `head` / `tail` / `upsampler.0` rows of [report.md](../results/quantization/200ep_with_report/report.md) |

---

## 9. Talking points (portfolio / interview ready)

These are full sentences that combine several terms cleanly. They're all
true of the current code:

- "I ran a **post-training quantization analysis** on EDSR-baseline. The
  scheme is **symmetric per-tensor INT8** for activations, **symmetric
  per-channel INT8** for weights — the standard recipe most edge runtimes
  consume."
- "Calibration uses **max-abs**, deliberately the simplest choice — the
  model isn't sensitive enough yet to require percentile clipping or
  KL-divergence calibration. I left those as drop-in upgrades in the
  primitives."
- "I ran a **per-layer sensitivity sweep** to identify
  **quantization-critical layers**. Three layers (input conv, output
  conv, upsampler conv) account for 83% of the total INT8 PSNR drop —
  this is the **classic SR PTQ pattern** described in PAMS and 2DQuant."
- "From the sensitivity ranking I derived a **mixed-precision recipe**:
  keep the three critical layers in FP16, INT8 the rest. The estimated
  PSNR drop falls from 0.077 dB (pure INT8) to ~0.017 dB."
- "All accuracy analysis runs on **fake-quantization in PyTorch float
  math** — no INT8 backend required. The **deployment latency** numbers
  are deferred to a separate ONNX Runtime QInt8 / TensorRT pass once
  `benchmark_onnx.py` is in place."
- "Once on a vendor NPU, the picture changes again — those are
  **memory-bound** workloads, not compute-bound, so the INT8 speedup
  comes mostly from the **4× smaller weight footprint** rather than
  faster math. The accuracy analysis transfers; the latency analysis
  does not."

---

## 10. Terms not yet relevant — but you'll see them

If you read papers in adjacent fields:

| Term | Where it shows up |
|---|---|
| **GPTQ / AWQ / SmoothQuant** | LLM weight-only quantization; SR doesn't really use them. Asked about anyway in interviews. |
| **W8A8 / W4A8 / W4A4** | Bit-width naming convention. `W` = weight bits, `A` = activation bits. |
| **Activation Outlier / Outlier Suppression** | Big topic in LLM quantization. Some carry-over to SR but minor. |
| **Calibration dataset diversity** | Phenomenon: too small / too uniform a calibration set leads to a too-narrow scale, missing the true activation range at deploy. |
| **Quantization-aware deployment / device-aware quantization** | The deploy-side counterpart of QAT — design the quantization scheme around what the target NPU supports. |

---

## How to use this doc

* **Reading code**: when you encounter a function name like
  `per_channel_scale` or a comment about "max-abs calibration", search
  this doc for the term — should be in section 3 or 4.
* **Writing the report**: section 9's sentences are paraphrasable to
  re-explain the same data with the right terminology.
* **Reading papers**: PAMS / 2DQuant / DAQ assume the vocabulary in
  sections 1, 3, 4, 6. Sections 5 and 7 are background.
* **Talking to a vendor / deploy engineer**: section 7 is the shared
  vocabulary you'll switch into.
