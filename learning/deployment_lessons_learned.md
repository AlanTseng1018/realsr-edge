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
