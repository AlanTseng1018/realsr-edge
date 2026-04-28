# Edge AI Deployment: Training → Quantization → Deploy Methodology

A reference for the strategic flow of taking a deep-learning model from a
training script all the way onto an edge inference target (NPU on a TV
SoC, mobile DSP, embedded GPU, etc.). Focuses on **precision decisions and
analysis** at each stage — what runs in FP32, what runs in fake-quant,
what runs in true INT8, and how the artifacts hand off between stages.

This is a methodology document, not a tutorial. The aim is the **structure
of the work**: what to do in what order, what artifact each stage owns,
and what gates the transition between stages. Specific tools (PyTorch,
ONNX Runtime, TensorRT, SNPE, NeuroPilot, ...) are interchangeable inside
this skeleton.

---

## The 5-stage pipeline

```
Stage 1            Stage 2                Stage 3            Stage 4              Stage 5
──────             ──────                 ──────             ──────               ──────
TRAIN              ANALYZE                DECIDE             QUANTIZE             VALIDATE
                                                             (per backend)        (on hardware)

FP32 weights       FP32 weights +         human + report     real precision       real precision +
in PyTorch /       fake-quant             from stage 2       conversion per       real hardware
TensorFlow /       simulation                                target backend
JAX

   │                  │                       │                   │                    │
   ▼                  ▼                       ▼                   ▼                    ▼
 best.pt /         report.md             "use INT8 +          edsr_int8.onnx        deployment_
 best_ckpt         (PSNR drops,           FP16 head/tail"      edsr.plan             report.md
                   per-layer                                   edsr.dlc              (real latency,
                   sensitivity,                                edsr.dla              real PSNR,
                   mixed-precision                             ...                   real memory)
                   recipe)
```

Each `→` is a hand-off; each `↑` (not drawn) is an iteration loop.

---

## Stage 1 — Training (FP32 baseline)

| Aspect | Content |
|---|---|
| **Precision** | FP32 throughout. |
| **Why FP32** | Training gradients have wide dynamic range. FP16 frequently overflows / underflows during loss + gradient computation; BF16 training is feasible but the deploy artifact still has to be cast. The point of stage 1 is to produce an **accuracy ground truth** — quantization is a deployment concern, not a training concern. |
| **What you analyze** | Train / val curves, task metrics (PSNR / SSIM / LPIPS for SR; mAP / accuracy for detection / classification; etc.). Deliberately **no quantization analysis at this stage**. |
| **Artifacts** | A FP32 checkpoint (`best.pt` / `model.h5` / saved-model dir). Possibly per-epoch checkpoints for resume. |
| **Exit gate** | The model meets the task accuracy target on the FP32 baseline. If it doesn't pass here, **no later stage will rescue it** — get the FP32 model right first. |

**Exception — QAT (Quantization-Aware Training)**: if PTQ analysis later
shows accuracy is unrecoverable at the target precision, the team comes
back to stage 1 and inserts fake-quant ops into the training graph. QAT
is a **stage 1 modification driven by a stage 3 decision** — never the
default first move.

---

## Stage 2 — Pre-deployment Analysis (FP32 + fake-quant)

| Aspect | Content |
|---|---|
| **Precision** | FP32 weights on disk and in RAM. **Fake-quantization** simulates the precision behavior of FP16 / BF16 / INT8 inside PyTorch float math, without touching weight dtype. |
| **Why this stage exists** | To answer "**how much accuracy will we lose at deploy precision X**" without paying the engineering cost of a real backend conversion. PTQ tooling per-backend is expensive (calibration set design, op-support investigation, hours of compile time on TensorRT, etc.). Fake-quant gives an answer in an afternoon, on any hardware. |
| **Three questions to answer** | 1) Pure-INT8 PSNR drop? 2) Which layers are quantization-critical? 3) What's the estimated drop with mixed precision? |
| **Key analyses** | a) **Format shootout**: FP32 vs FP16-autocast vs BF16-autocast vs INT8-fake-quant on the val set. b) **Per-layer sensitivity sweep**: hold all layers in FP32, fake-quantize ONLY one layer to INT8, measure drop. Repeat for every quantizable layer. Sort by drop. c) **Mixed-precision recipe**: keep top-N most-sensitive layers in higher precision, INT8 the rest. Estimate combined drop assuming layer effects are roughly additive (verified by the data). |
| **Artifacts** | A markdown report with: model identity (checkpoint path, mtime, size, params), test setup (val set, calibration set), shootout table, per-layer sensitivity table, mixed-precision recommendation. Plus CSV files for programmatic consumption. |
| **Exit gate** | The team has a defensible answer to "**should we deploy at INT8, mixed, FP16, or BF16?**" — and a recipe to back it up. |

**The accuracy numbers from this stage are hardware-agnostic.** A 0.X dB
drop measured here will replicate (within ~0.05–0.1 dB) on ONNX RT QInt8,
TensorRT INT8, SNPE INT8, or any other INT8 backend, because the root
cause is the INT8 grid resolution — not the runtime.

**The latency numbers from this stage are NOT real deploy latency.**
Fake-quant adds quant-dequant overhead on top of FP32 ops; the resulting
latency will often be slower than FP32. Treat fake-quant latency as a
debugging signal, not as deploy timing.

---

## Stage 3 — Decision (human + report)

This is the most undervalued stage. The decisions made here drive every
downstream artifact, and they are **not in code** — they are in a written
rationale, ideally as an Architecture Decision Record (ADR).

### Inputs
- Stage 2's report (sensitivity ranking, mixed-precision estimate)
- Target hardware's op support matrix (vendor doc)
- Accuracy budget (how much PSNR / accuracy can the product afford to lose?)
- Memory budget (does the model + activations fit in NPU SRAM?)
- Latency budget (real-time? per-frame? batch?)

### Decision tree

```
Total INT8 drop > accuracy budget?
  ├─ NO → Plan: deploy at INT8.
  │
  └─ YES → Mixed precision (top-N sensitive layers in FP16/FP32) feasible?
            ├─ NO  (still over budget)            → Plan: QAT (loop back to stage 1)
            ├─ YES (within budget)                → Plan: deploy at mixed precision
            └─ Mixed brings it close, not under   → Hybrid: mixed + QAT for the
                                                            critical layers

Side checks (independent of accuracy):
  - Is every op supported on the target backend? Any silent CPU fallback?
  - Does mixed-precision INT8 ↔ FP16 transition cost more than uniform precision?
  - Does the model fit in target memory?
  - Does the architecture have backend-hostile ops (PixelShuffle / DepthToSpace,
    GroupNorm, custom layers, dynamic shapes) that need a graph rewrite?
```

### Output

A short ADR-style document: "we chose precision X for these reasons,
expected accuracy is Y, target backend is Z." This document stays the
single source of truth across stages 4-5; if validation results disagree
with it, the decision is revisited explicitly.

---

## Stage 4 — Backend-Specific Quantization

This is the first stage that **produces a non-FP32 artifact**. Each
backend is its own pipeline.

```
                       FP32 ONNX  (the freeze point)
                            │
            ┌───────────────┼────────────────┬─────────────────┐
            ▼               ▼                ▼                 ▼
        ORT QInt8       TensorRT          SNPE              NeuroPilot
        (Microsoft)     (NVIDIA)          (Qualcomm)        (MediaTek)
            │               │                │                 │
            ▼               ▼                ▼                 ▼
     edsr_int8.onnx     edsr.plan        edsr.dlc          edsr.dla
            │               │                │                 │
            ▼               ▼                ▼                 ▼
    onnxruntime         TensorRT          SNPE-Runtime     APU-Runtime
    Python / C++        Runtime           (mobile, TV)     (mobile, TV)
```

### What each path does

| Backend | Quantize step | Calibration model | Output |
|---|---|---|---|
| ONNX Runtime | `quantize_static()` / `quantize_dynamic()` | A `CalibrationDataReader` that yields LR samples | Quantized `.onnx` with `QLinearConv` ops |
| TensorRT | `trtexec --int8 --calib=cache.bin` | Engine builder + calibrator interface | `.plan` engine (hardware-specific!) |
| SNPE | `snpe-onnx-to-dlc` then `snpe-dlc-quantize` | calibration set as raw arrays | `.dlc` container |
| NeuroPilot | `mtk_converter --quantize` | calibration TFRecord / npy | `.dla` blob |
| Edge TPU | `edgetpu_compiler` on a TFLite-quantized model | calibration during TFLite quantize | `.tflite` Edge TPU variant |

### Key cross-cutting points

- **Each backend has its own calibration algorithm**. ONNX RT defaults to
  MinMax; TensorRT defaults to KL-divergence (entropy); SNPE has its own.
  Even on the same FP32 model, the resulting INT8 PSNR will vary slightly
  (~0.05-0.1 dB).
- **Each backend's quantized artifact is hardware-specific**. A TensorRT
  `.plan` built for SM 8.9 will not run on SM 9.0. Treat backend
  artifacts as build-time outputs, **never** as portable models.
- **The FP32 ONNX is the freeze point**. Every backend starts from it.
  This decouples the upstream training/research work from the downstream
  deploy work — research can change the model freely, and the same
  hand-off interface (FP32 ONNX) drives all backend pipelines.

---

## Stage 5 — Hardware Validation

This stage answers "**did stage 2 predict reality correctly?**" — and
produces the deploy-side numbers that inform shipping decisions.

| Validation | What it checks | Why it matters |
|---|---|---|
| **Accuracy parity** | Backend INT8 PSNR vs stage-2 fake-quant prediction (target: within ± 0.1 dB) | If the gap is larger, the backend's calibration / quant scheme differs from the analysis assumption — debug before shipping |
| **Real latency** | Forward-pass latency on target hardware (with proper warmup, cuda.synchronize / device sync, multi-iter mean ± std) | Stage 2's latency was fake-quant overhead; this is the first ground-truth deploy timing |
| **Op support** | Confirm every op runs on the accelerator, no silent fallback to CPU | Silent fallback can make INT8 *slower* than FP32. Use vendor profiler / verbose logging |
| **Memory peak** | Maximum memory during inference (weights + activations + intermediate buffers) | Edge memory is much tighter than desktop. Activation peak may matter more than weight size |
| **Power / thermals** | Sustained frame rate over time (not just first 30 seconds) | Edge devices throttle. Cold latency ≠ steady-state latency |

### Common debugging when validation diverges

| Symptom | Likely cause |
|---|---|
| INT8 backend PSNR much worse than fake-quant predicted | Different calibration algorithm (MinMax vs KL-div vs your max-abs); or per-tensor where you assumed per-channel; or unsupported asymmetric scheme |
| INT8 backend latency *slower* than FP32 | Op fallback to CPU; layout transforms inserted between INT8 / FP32 segments; INT8 hardware not actually engaged on the target |
| First inference very slow but subsequent fast | Cold-start: graph optimization, kernel compilation, cuDNN algorithm selection, .plan deserialization. Not a real deploy issue if you can warm up |
| Latency variance huge (std > mean / 2) | Thermal throttling, OS interrupts, too few iterations, no synchronization barrier. Increase iter count, sync explicitly, control thermals |

### Artifact

A deployment report mirroring stage 2's structure, but with **real
numbers**: accuracy on validation set, latency mean ± std, memory peak,
and an op-by-op confirmation of where each op ran.

---

## Cross-cutting principles

### 1. Training stays FP32 unless QAT is required

QAT is reactive, not preventive. Always train FP32 first; let stage 2
tell you whether QAT is needed. Most edge models don't need it.

### 2. Stage 2 fake-quant is mandatory before stage 4

Skipping stage 2 means starting backend quantization with no idea whether
the model is quantizable at all. Stage 4 is expensive (per-backend
toolchain learning, calibration tuning, op-support debugging); stage 2
is cheap (PyTorch-only, hours not days). Always pay for stage 2.

### 3. ONNX (or another backend-neutral format) is the freeze point

Don't push PyTorch artifacts to deploy engineers. The handoff is an
ONNX file with verified numerical equivalence. This decouples upstream
research from downstream deploy and is what enables multi-vendor
deployment from a single model.

### 4. Every stage's claim must be validated by the next stage

- Stage 1's claim (target accuracy reachable) → validated by stage 2's
  FP32 baseline number.
- Stage 2's claim (INT8 drop is X dB) → validated by stage 5's measured
  drop.
- Stage 3's claim (precision plan Y is appropriate) → validated by
  stage 5's full report.

When validation disagrees with the claim, re-open the previous stage's
ADR and update it. Don't paper over the gap.

### 5. Iteration is the norm

The pipeline rarely runs end-to-end on the first try. Common loops:

- Stage 5 → Stage 3: latency too slow → swap mixed-precision recipe.
- Stage 5 → Stage 1: accuracy too low → run QAT.
- Stage 4 → Stage 1: unsupported op → architectural change.

Plan for at least one full re-execution of stages 4-5 after the first
deploy report; budget time accordingly.

### 6. Each stage owns one artifact, and only one

This makes ownership traceable:

| Stage | Artifact |
|---|---|
| 1 | FP32 checkpoint |
| 2 | Analysis report (markdown + CSV) |
| 3 | Precision-decision ADR |
| 4 | Backend-specific quantized model file(s) |
| 5 | Deployment report (real-hardware numbers) |

If two stages co-own an artifact, you have a bug in the workflow.

---

## What changes per project type

This pipeline applies broadly to vision / speech / signal-processing
models. Specific projects flex it slightly:

| Project type | Notable adjustments |
|---|---|
| **SR (super-resolution)** | Activation distributions are long-tailed (no BN); per-channel weight quantization is essential; first conv / last conv / upsampler are typically quantization-critical |
| **Detection / classification** | BN is present, activations are better-bounded; INT8 PTQ usually works without QAT; calibration set composition (class balance) matters more |
| **Segmentation** | Output is dense; even small INT8 errors propagate spatially; sensitivity sweep usually identifies the decoder upsamplers as critical |
| **LLMs** | Different game entirely — weight-only INT4 is the norm (GPTQ, AWQ, SmoothQuant); activations are the hard part (outliers); pipeline above is qualitatively similar but the methods are different |
| **Real-time video** | Stage 5's "sustained throughput" matters more than peak latency; thermal / power budget gates the answer |

---

## Glossary cross-reference

The vocabulary used throughout this document is defined in
[`quantization_terminology.md`](quantization_terminology.md):

- "Fake-quant" → section 2 of the terminology doc
- "Per-tensor / per-channel" → section 3
- "Calibration / max-abs / KL-div" → section 4
- "Sensitivity analysis / quantization-critical" → section 5
- "Execution provider / op support matrix" → section 7

When this methodology is paired with the terminology doc, code reading and
deployment discussions both go faster — the principles tell you *what* to
do at each stage, the terminology tells you what to *call it*.
