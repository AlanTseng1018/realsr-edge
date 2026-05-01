# Edge AI Deployment — Lessons Learned

A field-notes companion to [`deployment_methodology.md`](deployment_methodology.md).
The methodology doc tells you the *correct* flow; this one tells you the
*surprises* you'll hit when actually running it. Each entry is a specific
finding that's not obvious from documentation, with the wrong assumption it
contradicts and the signal that exposes it.

---

## Lesson 1: ONNX Runtime CUDA EP is the wrong tool for INT8 on GPU

### TL;DR

Running an INT8 (QDQ-format) ONNX through ORT's `CUDAExecutionProvider`
typically produces latency **equal to or slower than FP32**, not faster.
This is a property of how that EP is implemented, not a property of INT8
or of the hardware.

### The wrong assumption

"INT8 weights are 4x smaller and INT8 hardware is faster than FP32, so
INT8 ONNX on CUDA EP will be 2-4x faster than FP32 ONNX on CUDA EP."

### The reality

ORT's `CUDAExecutionProvider` does not have native INT8 conv kernels for
the QDQ pattern. When it loads a QDQ-format INT8 ONNX, the
`QuantizeLinear` and `DequantizeLinear` ops are scheduled on **CPU**,
while the convolution itself runs as **FP32** on GPU. The graph ends up
with one `Memcpy` node before each Q/DQ op (CUDA → CPU) and one after
(CPU → CUDA). For a model with N convolutions, you get roughly **4N
Memcpy ops** in the final graph.

This is signaled at session-build time:

```
[W:onnxruntime: ...] N Memcpy nodes are added to the graph for
CUDAExecutionProvider. It might have negative impact on performance ...
```

The end result is a graph that does *more* work than the FP32 version
(extra Q/DQ + extra memcpy in both directions per layer) and uses
*neither* the CUDA Tensor Cores' FP16 path nor the INT8 path. Latency
goes up, not down.

### How to detect it

* Read the warning above at session-build time.
* Compare the FP32 and INT8 latencies on the same EP. If INT8 is
  slower, you've hit this trap.
* Inspect the optimized graph: if the count of Memcpy nodes scales with
  the number of convs, the EP is bouncing through CPU.

### The right approach

For real GPU INT8 inference, use a backend that compiles QDQ patterns
into INT8 Tensor Core kernels:

* **TensorRT** (NVIDIA) — handles QDQ correctly, expected ~2-4x speedup
  vs FP32 on Ampere/Ada/Blackwell.
* **ORT TensorRT EP** — wraps TensorRT under the same ORT API. Same
  speedup, but inherits TensorRT's caveats (engine build time, engine
  non-portability — see Lesson 5).
* **Vendor NPU SDKs** (SNPE, NeuroPilot, etc.) — designed around INT8
  from the silicon up.

For CPU INT8, ORT's `CPUExecutionProvider` does dispatch QDQ to native
INT8 instructions (VNNI on x86, dotprod on ARM) — that path is
legitimate.

---

## Lesson 2: `onnxruntime-gpu` lists TensorRT as available without actually having it

### TL;DR

`ort.get_available_providers()` returning `'TensorrtExecutionProvider'`
does **not** mean TensorRT is installed and usable. It means ORT was
built with TensorRT support compiled in. The actual TensorRT runtime
DLLs (NVIDIA's) are a separate installation. Without them, ORT
silently falls back to CUDA EP at session-creation time.

### The wrong assumption

"`onnxruntime-gpu` is installed and TensorrtExecutionProvider is in the
provider list, so my INT8 latency benchmark is using TensorRT."

### The reality

`onnxruntime-gpu` ships a thin shim DLL
(`onnxruntime_providers_tensorrt.dll`) that depends on NVIDIA's
TensorRT runtime DLL (`nvinfer_*.dll`, version-suffixed). If the NVIDIA
DLL is missing from `PATH`, the shim fails to load, but the EP name is
still listed by ORT. When you create an InferenceSession requesting
`TensorrtExecutionProvider`, ORT prints an `EP Error` and falls back
to the next provider in the chain (typically `CUDAExecutionProvider`).

The symptom is misleading: the session creates, queries report
"`active_provider = CUDAExecutionProvider`", and the benchmark
proceeds — but the user thinks they tested TensorRT.

### How to detect it

At session creation, look for:

```
*************** EP Error ***************
EP Error ... Error loading "onnxruntime_providers_tensorrt.dll"
which depends on "nvinfer_NN.dll" which is missing.
Falling back to ['CUDAExecutionProvider', 'CPUExecutionProvider'] ...
```

Or programmatically: after creating the session, check
`session.get_providers()[0]`. If you asked for `tensorrt` and got back
something else, the fallback fired.

### The right approach

* **Don't trust `get_available_providers()` alone**. Always inspect the
  active provider after session creation.
* **Install TensorRT explicitly**: `pip install tensorrt-cu12` (matching
  the CUDA version) or download from NVIDIA Developer Center. Both add
  the required DLLs to a discoverable location.
* **Verify with a deliberate test**: use a model with QDQ ops and a
  dynamic axis and try to build a TensorRT engine. If it succeeds and
  `active_provider == 'TensorrtExecutionProvider'`, the install is real.

---

## Lesson 3: Fake-quant accuracy and real-backend INT8 accuracy differ by ~0.05 dB

### TL;DR

A PyTorch fake-quant analysis predicts X dB PSNR drop for INT8. The
actual measured drop after running the QDQ ONNX through ORT
(`quantize_static`) typically differs by ±0.03 to ±0.10 dB from that
prediction. This is normal, not a bug.

### The wrong assumption

"Both fake-quant and ORT static quantize use INT8 with calibration on
the same val set, so they should give the same PSNR."

### The reality

The two pipelines are independent in three places:

1. **Calibration algorithm**: ORT defaults to MinMax; a typical
   fake-quant implementation uses max-abs running. Even when nominally
   "the same" (both effectively cover the seen range), the per-tensor
   resolution of activation observation and the rounding policy differ.
2. **Op fusion order**: ORT may fuse Conv+Add+ReLU before quantization,
   placing Q/DQ at different points than a layer-by-layer fake-quant
   wrapper.
3. **Numeric bit-exactness of round/clamp**: PyTorch and ORT's INT8
   kernels handle ties (e.g., `0.5` rounding) and saturation differently.

The cumulative effect on a small model is typically a fraction of a dB.

### How to detect it

Compare the two reports against the same FP32 baseline. If the gap is
within ~0.1 dB, treat it as expected backend variance. If it's larger
(say > 0.3 dB), investigate:

* Did calibration use the same data?
* Are Q/DQ ops placed at the same points?
* Did ORT's preprocessing (`shape_inference.quant_pre_process`) change
  the graph?

### The right approach

* **Use fake-quant as a fast estimator**, not as the final number.
* **The deploy-side number is the backend's**, always. Stage 2 of the
  methodology gives you the *direction*; Stage 5 gives you the
  *magnitude*.
* **Document the gap explicitly** in the deploy report so reviewers
  don't think one of them is "wrong".

---

## Lesson 4: FP16 on Tensor Cores is often a better deploy choice than INT8

### TL;DR

For models whose deploy target has FP16 Tensor Cores (NVIDIA Ampere+,
many recent ARM mobile NPUs), **FP16 inference is often the
practical sweet spot**: ~1.5-2x faster than FP32, with PSNR drop near
zero (sub-0.01 dB on most vision models). INT8 typically wins
*latency-wise* by another 1.5-2x, but adds calibration complexity,
backend constraints, and small accuracy loss.

### The wrong assumption

"Edge AI deploy means INT8 — it's the standard and obviously fastest."

### The reality

INT8 *is* the right answer for ultra-constrained edge (mobile NPU, TV
SoC, microcontroller-class hardware) where memory bandwidth is the
hard wall and where 4× weight compression is decisive. But on a
*general-purpose GPU edge target* (Jetson, NVIDIA edge servers,
discrete GPU desktops), FP16 often wins on the engineering ROI:

* No calibration step required.
* No risk of an unsupported INT8 op forcing a CPU fallback.
* No mixed-precision split between critical and robust layers.
* Tensor Cores accelerate FP16 to within ~30-50% of INT8 throughput.
* Accuracy preservation is ~free.

### How to detect it

Run the same model on the same hardware in three precisions (FP32,
FP16, INT8 via the appropriate backend). Compare the (latency × accuracy)
trade-off. If FP16 reaches the latency target, picking it over INT8
saves significant deploy engineering.

### The right approach

* **Always measure FP16 alongside INT8**. Don't skip it.
* **Decide based on which constraint binds first**: if memory size is
  binding, INT8 wins (4× compression). If compute time is binding *and*
  Tensor Cores are present, FP16 may be enough.
* **For NPU targets without FP16 Tensor Cores** (most TV SoCs), INT8 is
  still the right answer. The lesson here is "don't skip FP16 on GPUs",
  not "always use FP16".

---

## Lesson 5: TensorRT engine files (.plan / .engine) are hardware-specific

### TL;DR

A TensorRT engine built on one GPU model will not run on a different
GPU model — even within the same generation, even with the same
TensorRT version. The build is specialized to the exact compute
capability, SM count, and memory hierarchy of the target.

### The wrong assumption

"I built the engine once; I can ship the `.plan` file alongside the
ONNX and skip re-building on user machines."

### The reality

TensorRT serializes the optimized kernel choices, tile sizes, and tensor
layouts that were autotuned at build time. Those choices are tied to:

* Compute capability (e.g., SM 8.6 vs SM 8.9 vs SM 9.0).
* Number of SMs (different SKUs of the same architecture differ here).
* Memory configuration (HBM vs GDDR6, bandwidth, L2 size).
* TensorRT major version (engines built with TRT 8 don't load in TRT
  10; sometimes minor versions break too).

Loading on a mismatched device errors out at session creation, not at
inference time. There's no fallback.

### How to detect it

Try to load a `.plan` built for one device on another. The error message
will include the device the engine was built for vs the device trying
to load it.

### The right approach

* **Treat TensorRT engines as build-time artifacts**, not as
  shippable assets. Ship the ONNX; build the engine on the target.
* **Cache the engine on the target** (not in your repo) using
  `trt_engine_cache_enable` so the second run on a given device is fast.
* **Add `.plan` and `trt_engine_cache/` to `.gitignore`**.
* **For multi-target deploys**, include an engine-build step in the
  install / first-run path of the deployed application.

---

## Lesson 6: ONNX 是交換格式，不是部署格式

### TL;DR

ONNX 讓你用同一個檔案餵進不同的推論框架，但每個部署目標最終執行的都是自己的格式。
把 ONNX 當「終點」是誤解；它是「中間站」。

### The wrong assumption

「我有 ONNX 了，就可以直接部署到任何 edge 裝置上。」

### The reality

各平台的實際執行格式：

| 目標平台 | 執行格式 | ONNX 的角色 |
|---|---|---|
| NVIDIA GPU | TensorRT `.engine` | 入口（要重新編譯） |
| Qualcomm NPU | QNN `.bin` | 入口（要重新轉換） |
| Intel NPU | OpenVINO IR | 入口（要重新轉換） |
| Apple ANE | CoreML `.mlpackage` | 入口（要重新轉換） |
| ARM CPU | TFLite / NNAPI | 入口（要重新轉換） |
| ORT 本身 | ONNX | 可直接執行（但無 HW 優化） |

ORT 直接跑 ONNX 是最「方便」的路，但也是最「不優化」的路。
每個 edge target 都需要一個平台專屬的編譯步驟，才能吃到硬體加速（Tensor Core、INT8 kernel、memory layout optimization）。

正確的部署心智模型：
```
PyTorch 訓練
    ↓
ONNX 匯出（中間站，和框架解耦）
    ↓
目標平台編譯
    ├─ trtexec / TRT Python API → .engine（NVIDIA）
    ├─ mo.py → OpenVINO IR（Intel）
    ├─ coremltools → .mlpackage（Apple）
    └─ qnn-onnx-converter → .bin（Qualcomm）
    ↓
在目標硬體上執行最終格式
```

### How to detect it

如果你跑 `ort.InferenceSession(onnx)` 測 INT8 延遲，結果比 FP32 慢，
你測到的是 ORT 解釋 ONNX 的效能，不是目標硬體的效能。

### The right approach

* **以目標平台決定最終格式**，ONNX 只是通往那裡的通用入口。
* **PSNR 驗證用 ORT**（快、不需額外編譯、結果可靠）。
* **延遲測量用目標平台的 native runtime**（TRT engine、CoreML、SNPE 等）。
* 兩件事分開做，不要混用同一個工具同時驗精度和測速度。

---

## Lesson 7: onnxruntime `quantize_static` 的 QDQ ONNX 與 TensorRT 10 不相容

### TL;DR

`onnxruntime.quantization.quantize_static` 產生的 INT8 QDQ ONNX，
在 TensorRT 10 的 Python API 解析時會因 bias 的 `DequantizeLinear` 節點使用
`int32` zero_point 而報錯，無法建 engine。

### The wrong assumption

「我已經有 INT8 QDQ ONNX 了，直接用 TRT Python API 解析就能建 INT8 engine。」

### The reality

ORT `quantize_static` 對 bias 做量化時，會輸出：
```
DequantizeLinear(bias_quantized, scale, zero_point=int32)
```
TRT 10 的 `IDequantizeLayer` 只接受 `int8`、`fp8`、`fp4`、`int4` 作為 precision，
不接受 `int32`。解析時會拋出：
```
IDequantizeLayer::setPrecision: Error Code 3: API Usage Error
(condition: isQuantized(dataType) || ... A DequantizeLayer can only run in
DataType::kINT8, DataType::kFP8, DataType::kFP4, or DataType::kINT4 ...)
```

根本原因：bias 通常用 `int32` 量化（因為 bias 的 scale = input_scale × weight_scale，
精度要求比 activation 高），這在 ORT 的 CPU INT8 kernel 裡是合法的，
但 TRT 的 DequantizeLayer 不支援這個路徑。

### How to detect it

TRT parser 在 `parse()` 返回 False，錯誤訊息包含：
```
DequantizeLinear ... A DequantizeLayer can only run in DataType::kINT8 ...
```

### The right approach

**不要用 QDQ ONNX 建 TRT INT8 engine**，改用 TRT 的原生 INT8 calibration：

```python
# 用 FP32 ONNX + IInt8EntropyCalibrator2
calibrator = ValSetCalibrator(calib_batches, cache_path)
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = calibrator
engine = builder.build_serialized_network(network, config)
```

TRT 會自己從 FP32 graph 計算 activation range，建出完整 fuse 的 INT8 engine，
不依賴 QDQ 節點。Calibration 結果會寫入 cache 檔案，第二次建 engine 時直接讀取。

---

## Lesson 8: `time.perf_counter()` per-call 會把 Python overhead 算進 GPU 延遲

### TL;DR

用 `time.perf_counter()` 逐次測量 TRT inference 延遲，Python 函式呼叫 overhead
（約 1–2 ms）會掩蓋真實的 GPU 執行時間。對 FP32（本身就慢）影響小，
但對 FP16 / INT8（GPU 只跑 < 1 ms）會讓測量值膨脹 2–3x，
**導致看起來「INT8 比 FP32 慢」的錯誤結論**。

### The wrong assumption

「`time.perf_counter()` 包在 `stream.synchronize()` 後面，所以量到的是真實延遲。」

### The reality

即使 `stream.synchronize()` 確保 GPU 已完成，測量還是包含：
- Python 函式呼叫 overhead
- `numpy.asarray` / `torch.copy_` 等 CPU 端資料搬移
- `time.perf_counter()` 自身的讀取成本

對 FP32（GPU 執行 ~1.5 ms），這些 overhead（~0.3–0.5 ms）佔比小。
對 FP16（GPU 執行 ~0.8 ms），overhead 佔 > 50%，測出來反而比 FP32 慢。

實測對比（RTX 3090，bench shape 1×3×96×96，EDSR 1.37M 參數）：

|           | `time.perf_counter()` | CUDA Events（正確） |
|-----------|----------------------:|--------------------:|
| FP32      | 1.56 ms               | 1.54 ms             |
| FP16      | **1.79 ms (看似較慢)** | **1.23 ms (較快)**  |
| INT8      | **2.60 ms (看似最慢)** | **1.90 ms**         |

初版 benchmark 用 `time.perf_counter()` 得到「INT8 比 FP32 慢 1.66x」，
經 CUDA Events 重測後，INT8 延遲從 2.60 ms 降到 1.90 ms，FP16 從 1.79 ms
降到 1.23 ms。**FP16 實際上比 FP32 快 20%**。

### How to detect it

前後量測值差異 > 20% 即為警訊。用 `torch.profiler` 比較 device time
（GPU 執行時間）vs wall-clock time，差距大代表測量方式有問題。

### The right approach

用 CUDA Events 在 **同一個 stream** 包住批次呼叫，再除以迭代數：

```python
t_start = torch.cuda.Event(enable_timing=True)
t_end   = torch.cuda.Event(enable_timing=True)
t_start.record()
for _ in range(n_iter):
    engine.infer(sample)   # 內部含 stream.synchronize()
t_end.record()
torch.cuda.synchronize()
latency_ms = t_start.elapsed_time(t_end) / n_iter
```

* `t_start` / `t_end` 在 default（null）stream 上，null stream 會等所有其他
  stream 完成才執行，因此能正確包住 custom stream 上的 GPU work。
* 批次量再除以 n，可攤薄單次 Event 建立的 overhead。
* **不要用 `time.perf_counter()` 逐次測量 GPU 延遲**，它量到的是 wall-clock，
  不是 GPU 執行時間。

---

## Lesson 9: `onnxruntime` 與 `onnxruntime-gpu` 同時安裝會互相覆蓋

### TL;DR

在同一個環境裡同時裝了 `onnxruntime`（CPU-only）和 `onnxruntime-gpu`，
`get_available_providers()` 只會回傳 `['CPUExecutionProvider']`，
CUDA EP 和 TensorRT EP 消失。

### The wrong assumption

「`onnxruntime-gpu` 已經裝了，CUDA EP 應該要出現。」

### The reality

`onnxruntime` 和 `onnxruntime-gpu` 共用同一個 Python module 名稱（`onnxruntime`）。
pip 安裝 `onnxruntime` 時，它會覆蓋 `onnxruntime-gpu` 的 `onnxruntime` package，
讓 CPU-only 版本的 `onnxruntime_providers_cuda.dll` 等 DLL 消失。
`import onnxruntime` 成功，但實際上 import 到的是 CPU-only 版本。

這個問題在 `pip install onnx onnxruntime-gpu` 後又跑了
`pip install onnxruntime`（例如因為別的工具有這個 dependency）時會悄悄發生。

### How to detect it

```python
import onnxruntime as ort
print(ort.get_available_providers())
# 應該要有 CUDAExecutionProvider；如果只有 CPUExecutionProvider，就中招了
```

### The right approach

```bash
# 一次清掉兩個，重裝 gpu 版
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu
```

安裝後用 `get_available_providers()` 驗證，應看到：
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

* **在 `requirements.txt` 裡只列 `onnxruntime-gpu`，不要同時列兩個**。
* 如果某個工具強制要求 `onnxruntime`（CPU-only），把它裝在另一個 venv。

---

## Lesson 10: torch 2.6+ 預設 dynamo exporter，不支援 `dynamic_axes`

### TL;DR

PyTorch 2.6 開始，`torch.onnx.export()` 預設使用新的 dynamo-based exporter。
這個 exporter 忽略 `dynamic_axes` 參數，輸出一個幾乎空的 ONNX（~0.03 MB），
verification 會 FAIL。

### The wrong assumption

「`torch.onnx.export()` 用法沒變，之前能跑的 code 現在應該也能跑。」

### The reality

PyTorch 2.6 改變了預設行為（將在 2.9 正式成為唯一路徑）：

* **舊 exporter**（TorchScript-based）：支援 `dynamic_axes`，輸出完整的 ONNX graph with weights。
* **新 exporter**（dynamo-based）：使用 `torch.export.export()`，
  `dynamic_axes` 被忽略（會印 UserWarning），改用 `dynamic_shapes`；
  如果 dynamic shapes 沒設定好，輸出的 ONNX 會是靜態 graph 或不完整的 graph。

症狀很明顯：匯出的 ONNX 只有 0.03 MB，遠小於預期（FP32 EDSR 應該是 ~5 MB）。

### How to detect it

檢查匯出後的 ONNX 檔案大小。如果異常地小（< 1 MB 對一個 1M+ 參數的模型），
就是 dynamo exporter 沒有正確 export weights。

### The right approach

在完全遷移到新 exporter 之前，加 `dynamo=False` 強制使用舊版：

```python
torch.onnx.export(
    model, dummy_input, path,
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    dynamo=False,  # 明確使用 TorchScript-based exporter
)
```

長期計畫：改用 `dynamic_shapes` API 遷移到新 exporter，
但需要配合 `torch.export.export()` 的語法，是個較大的改動。

---

## Cross-references

* The full pipeline structure these lessons sit inside:
  [`deployment_methodology.md`](deployment_methodology.md).
* The terminology used here ("QDQ", "execution provider",
  "memory-bound", "Tensor Core") is defined in
  [`quantization_terminology.md`](quantization_terminology.md).
* For why max-abs calibration beats percentile clipping on SR
  activations specifically — a different kind of "wrong default" —
  see [`reading_calibration_histograms.md`](reading_calibration_histograms.md).

## How to use this doc

* **Before deploying**: skim it, especially Lessons 1, 2, 5. These are
  the silent-failure modes that break benchmarks and derail handoffs.
* **When something doesn't match expectation**: search here first.
  Most of these lessons describe symptoms before they describe causes,
  so the symptom you're seeing should jump out.
* **When extending the project**: each lesson lists the *signal* that
  exposes the trap. Bake that signal as an assertion or warning into
  your benchmarking / deploy code so the next person doesn't hit it.
