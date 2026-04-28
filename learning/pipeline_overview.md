# RealSR-Edge — Pipeline Overview

A project-specific snapshot of how the codebase is wired end-to-end:
training data → trained model → quantization analysis → deployment
artifacts. Pairs with the generic methodology in
[`deployment_methodology.md`](deployment_methodology.md) — that doc
explains the *why* of each stage; this one shows *where* in this repo
each stage actually lives, with the concrete artifact each step
produces.

The document highlights **seven analysis nodes** — the points in the
pipeline where a number / chart / decision is produced and consumed
downstream. These are the touchpoints to inspect when debugging,
extending, or explaining the project.

---

## Pipeline at a glance

```
[Stage 1: DATA]
  data/DIV2K/  +  src/data/degradation.py  ->  src/data/dataset.py
  ┌──────────────────────────────────────────────────────────────┐
  │ DIV2K_train_HR (800)  +  Realistic Degradation               │
  │ DIV2K_valid_HR (100)     (blur/noise/JPEG/banding/downsample/│
  │                           chroma -- chroma kept but not in   │
  │                           training pipeline)                 │
  └──────────────────────────────────────────────────────────────┘
                   |  (HR/LR tensor pairs)
                   v
  ◇ Analysis Node 1: degradations look realistic?
                     -> notebooks/01_degradation_demo.ipynb

[Stage 2: TRAIN]
  src/training/train.py  +  src/models/edsr.py
  ┌──────────────────────────────────────────────────────────────┐
  │ Adam lr=1e-4 + L1 + StepLR, 200 epochs, batch 16, patch 96   │
  └──────────────────────────────────────────────────────────────┘
                   |
                   v
  ◇ Analysis Node 2: training converged?
                     -> curves.png + metrics.csv
                   |
  results/runs/<timestamp>/checkpoints/best.pt
                   (PSNR 27.44 dB / SSIM 0.79 on this run)

[Stage 3: QUANTIZATION ANALYSIS]
  src/quantization/{fake_quant.py, analyze.py, calibration_ablation.py}
  ┌──────────────────────────────────────────────────────────────┐
  │ Fake-quant simulation in PyTorch (FP32 backend):             │
  │   - Format shootout (FP32 / FP16 / BF16 / INT8)              │
  │   - Per-layer sensitivity sweep (36 Conv2d layers)           │
  │   - Calibration ablation (max-abs / pct 99.99 / 99.9 / 99.0) │
  └──────────────────────────────────────────────────────────────┘
                   |
                   v
  ◇ Analysis Node 3: each precision's PSNR drop?
                     -> shootout.md
  ◇ Analysis Node 4: which layers are quantization-critical?
                     -> sensitivity.csv + bar chart
  ◇ Analysis Node 5: which calibration scheme wins?
                     -> calibration_ablation.md + histograms.png
                   |
  Decision: deploy at INT8 + max-abs, with optional mixed precision
            (head + tail + upsampler.0 in FP16) if accuracy demands

[Stage 4: DEPLOY]
  src/deployment/{export_pipeline.py, benchmark_onnx.py}
  ┌──────────────────────────────────────────────────────────────┐
  │ Export pipeline: best.pt -> 3 ONNX (FP32, FP16, INT8 static) │
  │ Benchmark:       each ONNX x {CUDA, CPU, [TRT]}              │
  └──────────────────────────────────────────────────────────────┘
                   |
                   v
  ◇ Analysis Node 6: does ONNX preserve PyTorch numerics?
                     -> verification.md
  ◇ Analysis Node 7: does deploy match the analysis prediction?
                     -> benchmark.md
                   |
  Output: a deploy-side report card with PSNR + latency
          across precisions and backends
```

---

## Stage 1 — DATA: training data + realistic degradation

| Aspect | Value |
|---|---|
| **Purpose** | Build supervised HR/LR pairs that approximate real TV-broadcast content (not just bicubic-downsampled academic LR). |
| **Code** | [`src/data/degradation.py`](../src/data/degradation.py) — five degradation methods (blur, noise, JPEG, banding, downsample) + retained `chroma_subsampling` (not in training pipeline). [`src/data/dataset.py`](../src/data/dataset.py) — `SRDataset`: random crop + horizontal flip + 90° rotations, deterministic-seed val. |
| **Inputs** | `data/DIV2K/DIV2K_train_HR/` (800 PNG), `data/DIV2K/DIV2K_valid_HR/` (100 PNG) |
| **Outputs** | Per `__getitem__`: `(lr_tensor, hr_tensor)` shape `(3, 96, 96)` / `(3, 192, 192)`, float32 in `[0, 1]` |
| **Design notes** | Downsample is forced (not 50% probabilistic) so DataLoader can batch consistently; other augmentations stay 50%; val uses deterministic seed = idx so PSNR is comparable across epochs. |

### ◇ Analysis Node 1 — degradation visualization

[`notebooks/01_degradation_demo.ipynb`](../notebooks/01_degradation_demo.ipynb)
— 7 cells: per-degradation panel + PSNR, intensity sweeps, random
pipeline variability across seeds, HR/LR pair preview, banding
deep-dive on smooth regions.

**Decision criterion**: by eye, do the LR images look like degraded TV
content? If they look "too clean" (only mild blur), the pipeline
parameters need to be more aggressive. If they look "too destructive"
(everything saturated), they need to be tamer.

---

## Stage 2 — TRAIN: model training

| Aspect | Value |
|---|---|
| **Purpose** | Train an FP32 EDSR-baseline checkpoint that hits SR-literature PSNR / SSIM. |
| **Code** | [`src/training/train.py`](../src/training/train.py) — CLI script with `--compile` (torch.compile), `--degradation realistic\|bicubic` (Track A vs B switch). [`src/models/edsr.py`](../src/models/edsr.py) — EDSR-baseline (16 ResBlock × 64 feats, PixelShuffle 2x). |
| **Hyperparameters** | Adam (lr=1e-4), L1 loss, StepLR (step=100, γ=0.5), batch 16, patch 96 LR / 192 HR, 200 epochs |
| **Outputs** | `results/runs/<timestamp>/`: `checkpoints/best.pt`, `metrics.csv` (per-epoch PSNR + SSIM), `curves.png`, `val_samples/*.png` (5 images) |
| **Result on this run** | val PSNR 27.44 dB, SSIM 0.79, L1 loss 0.10 → 0.029 |

### ◇ Analysis Node 2 — training-curve convergence

`curves.png` + `metrics.csv`. Look for:
- L1 loss decay over epochs (should be exponential, plateauing in last quarter)
- val PSNR rising (should reach ~25 dB by epoch 10, ~27 dB by epoch 100)
- No overfitting signature (val should not be falling while train still improves)

**Decision criterion**: loss curve still falling at epoch 200 → consider
more epochs. Val flat for 50+ epochs → stop early. Loss spiking →
something wrong (LR too high, bad batch, etc.).

---

## Stage 3 — ANALYZE: quantization analysis (fake-quant, no backend)

This stage is pure PyTorch — answers the **accuracy** question without
touching deploy backends. Three analysis nodes here.

| Aspect | Value |
|---|---|
| **Purpose** | From the same FP32 checkpoint, simulate FP16 / BF16 / INT8 precision loss; identify quantization-critical layers; pick a calibration scheme. |
| **Code** | [`src/quantization/fake_quant.py`](../src/quantization/fake_quant.py) — `CalibratingConv2d` with 3 modes (fp32/calibrate/quantize) + symmetric per-tensor activation + symmetric per-channel weight + max-abs / histogram primitives. [`analyze.py`](../src/quantization/analyze.py) — shootout + per-layer sensitivity. [`calibration_ablation.py`](../src/quantization/calibration_ablation.py) — 4-scheme comparison + histogram visualization. |
| **Inputs** | `best.pt` + DIV2K val set (used for calibration AND evaluation) |
| **Outputs** | Two folders: `results/quantization/200ep_with_report/` (shootout + sensitivity + report) and `results/quantization/calibration_ablation/` (4-scheme + histograms.png) |

### ◇ Analysis Node 3 — format shootout

`shootout.md`:

| Format | PSNR | Drop |
|---|---:|---:|
| FP32 | 27.439 | — |
| FP16 (autocast) | 27.438 | +0.001 (essentially free) |
| BF16 (autocast) | 27.422 | +0.017 |
| INT8 (fake-quant) | 27.359 | +0.079 (within typical SR PTQ range) |

### ◇ Analysis Node 4 — per-layer sensitivity

`sensitivity.csv` ranking (quantize one layer at a time, hold others FP32):

```
1. tail (output)           +0.029 dB
2. upsampler.0             +0.020 dB
3. head (input)            +0.016 dB
4. body.16 (post-resblock) +0.007 dB
5. body.0.conv2            +0.001 dB
... 31 ResBlock interior layers ≈ 0 dB
```

Top-3 critical layers contribute 83% of total INT8 drop. Drives the
mixed-precision recipe: keep those three in FP16, INT8 the rest.

### ◇ Analysis Node 5 — calibration ablation

`calibration_ablation.md` + `histograms.png`:

| Scheme | PSNR | Drop |
|---|---:|---:|
| max-abs | 27.361 | +0.077 ← winner |
| pct-99.99 | 27.358 | +0.080 |
| pct-99.9 | 26.975 | +0.464 |
| pct-99.0 | 25.071 | +2.367 |

The exponential degradation with more aggressive percentile clipping is
the signature of "tail = signal" (no BN → activation tail is
informative, not outliers). Visual proof in `histograms.png`. The full
reading workflow is in
[`reading_calibration_histograms.md`](reading_calibration_histograms.md).

---

## Stage 4 — DEPLOY: ONNX export + benchmark

The handoff from PyTorch to backend-neutral artifacts, then real
deploy-side measurement.

### Stage 4a — Multi-precision ONNX export

| Aspect | Value |
|---|---|
| **Purpose** | From a single `best.pt`, produce three ONNX files at different precisions, all verified. |
| **Code** | [`src/deployment/export_pipeline.py`](../src/deployment/export_pipeline.py) |
| **Outputs** | `results/onnx_exports/edsr_200ep/`: `edsr_fp32.onnx` (5.24 MB), `edsr_fp16.onnx` (2.63 MB), `edsr_int8_static.onnx` (1.41 MB), `README.md`, `metadata.json`, `verification.md` |
| **How each ONNX is produced** | FP32: `torch.onnx.export` (opset 17, dynamic axes). FP16: `onnxconverter-common.float16.convert_float_to_float16` on the FP32 ONNX. INT8: ORT `quantize_static` (QDQ format, symmetric per-tensor activation + per-channel weight), with calibration on 64 LR images from val set. |

### ◇ Analysis Node 6 — ONNX numeric verification

`verification.md` — each ONNX runs on multiple input shapes; max abs
diff vs PyTorch is recorded:

| Format | atol | observed max abs diff |
|---|---:|---:|
| FP32 | 1e-4 | 0.00e+00 (bit-level identical via shared cuDNN kernels) |
| FP16 | 5e-2 | ~9e-4 (within FP16 precision floor) |
| INT8 | 1e-1 | ~2.7e-2 (~half a quantization step, normal) |

If a row fails the tolerance, the export is broken — every downstream
deploy benchmark is suspect until that's fixed.

### Stage 4b — Deployment benchmark

| Aspect | Value |
|---|---|
| **Purpose** | Same 3 ONNX files, multiple ORT execution providers, real PSNR + latency numbers. |
| **Code** | [`src/deployment/benchmark_onnx.py`](../src/deployment/benchmark_onnx.py) |
| **Outputs** | `results/onnx_benchmark/<run_name>/`: `benchmark.md`, `benchmark.csv`, `metadata.json` |

### ◇ Analysis Node 7 — deploy real numbers vs analysis prediction

`benchmark.md` — typical 6-row shape (3 ONNX × 2 providers, with TRT a
third option once installed):

| Precision | Provider | PSNR | Drop | Latency (ms) | vs FP32 same-EP |
|---|---|---:|---:|---:|---|
| FP32 | CUDA | 27.439 | — | 5.52 | baseline |
| **FP16** | **CUDA** | 27.438 | +0.001 | **3.36** | **1.64x faster** ← realistic sweet spot |
| INT8 | CUDA | 27.322 | +0.117 | 6.29 | 1.14x slower (anti-pattern, see deployment_lessons_learned) |
| FP32 | CPU | 27.439 | — | 45.28 | baseline |
| FP16 | CPU | 27.439 | -0.000 | 45.94 | 1.01x slower (CPU has no FP16 acceleration) |
| INT8 | CPU | 27.321 | +0.118 | 53.01 | 1.17x slower (small model, VNNI overhead doesn't amortize) |

**Stage 3 prediction vs Stage 4 measurement parity check**:

* Fake-quant predicted INT8 drop: 0.077 dB
* ORT real INT8 drop: 0.117 dB
* Gap: 0.04 dB — within "expected backend variance"
  (see [`deployment_lessons_learned.md`](deployment_lessons_learned.md) Lesson 3)

### Stage 4c — C++ inference reference

| Aspect | Value |
|---|---|
| **Purpose** | Reference C++ deployment using ONNX Runtime API, demonstrating the cross-language deploy path. |
| **Code** | `cpp_inference/CMakeLists.txt`, `cpp_inference/src/main.cpp` (~200 lines), `cpp_inference/README.md` |
| **Status** | Skeleton complete; not yet built (requires user to download ONNX RT release zip + stb headers; documented in the README). |
| **Test plan when built** | sr_cli.exe input.png output.png ; bit-compare output.png with PyTorch SR result on same input. Expect max pixel diff ≤ 1 LSB. |

---

## All 7 analysis nodes at a glance

| # | Where | Question answered | Decision driven |
|---|---|---|---|
| **1** | `01_degradation_demo.ipynb` | Are degradations realistic? | Tune degradation parameter ranges if not |
| **2** | `curves.png` + `metrics.csv` | Did training converge? | Stop early / extend training / debug instability |
| **3** | `shootout.md` | What's each precision's accuracy cost? | Whether FP16/INT8 is even feasible |
| **4** | `sensitivity.csv` | Which layers are quantization-critical? | Mixed-precision recipe (keep top-N high precision) |
| **5** | `calibration_ablation.md` + `histograms.png` | Which calibration scheme is right? | Stick with max-abs (for SR) or switch to percentile (for outlier-prone models) |
| **6** | `verification.md` | Did ONNX export preserve numerics? | Block deploy until passes; debug export if not |
| **7** | `benchmark.md` | Real backend PSNR + latency vs prediction? | Final precision/provider deploy choice |

---

## File map (where each output lives)

```
RealSR-Edge/
├── data/DIV2K/                                       (gitignored)
├── notebooks/
│   ├── 01_degradation_demo.ipynb                     (Node 1)
│   ├── 02_training_demo.ipynb                        (training tutorial)
│   └── 04_quantization_analysis.ipynb                (Node 3-5 interactive)
├── src/
│   ├── data/{degradation.py, dataset.py}             (Stage 1)
│   ├── models/{edsr.py, common.py}                   (Stage 2 model)
│   ├── training/train.py                             (Stage 2)
│   ├── quantization/
│   │   ├── fake_quant.py                             (primitives)
│   │   ├── analyze.py                                (Node 3, 4)
│   │   └── calibration_ablation.py                   (Node 5)
│   └── deployment/
│       ├── export_onnx.py                            (Stage 4a single)
│       ├── export_pipeline.py                        (Stage 4a multi)
│       └── benchmark_onnx.py                         (Stage 4b)
├── cpp_inference/
│   ├── CMakeLists.txt + src/main.cpp                 (Stage 4c)
├── results/
│   ├── runs/<timestamp>/checkpoints/best.pt          (Node 2)
│   ├── runs/<timestamp>/{metrics.csv, curves.png}    (Node 2)
│   ├── quantization/200ep_with_report/               (Node 3, 4)
│   ├── quantization/calibration_ablation/            (Node 5)
│   ├── onnx_exports/edsr_200ep/                      (Stage 4a + Node 6)
│   └── onnx_benchmark/edsr_200ep/                    (Stage 4b + Node 7)
└── learning/
    ├── pipeline_overview.md                          (this file)
    ├── sr_design_thinking.md
    ├── deployment_methodology.md
    ├── deployment_lessons_learned.md
    ├── quantization_terminology.md
    └── reading_calibration_histograms.md
```

---

## Cross-references

* The generic 5-stage methodology this pipeline instantiates:
  [`deployment_methodology.md`](deployment_methodology.md).
* The vocabulary used here ("QDQ", "execution provider", "calibration",
  "sensitivity"): [`quantization_terminology.md`](quantization_terminology.md).
* Surprises hit while running this pipeline (ORT CUDA EP + INT8,
  fake-quant vs real backend gap, etc.):
  [`deployment_lessons_learned.md`](deployment_lessons_learned.md).
* Visual workflow for analysis Node 5:
  [`reading_calibration_histograms.md`](reading_calibration_histograms.md).
* EDSR design rationale (Stage 2):
  [`sr_design_thinking.md`](sr_design_thinking.md).

---

## How to update this document

This is a project-state snapshot. Update it when:

* **A stage's code changes structurally** (new script, renamed file,
  new analysis node).
* **An analysis node's deliverable format changes** (e.g. CSV column
  added, new chart type).
* **A new precision / format / backend joins the deploy benchmark**
  (e.g. TensorRT EP becomes available, an INT4 path is added).
* **Numbers in the example tables change materially** (a bigger or
  smaller training run replaces the 200-epoch one).

Don't update it for:

* Per-run hyperparameter tweaks (those go into ADRs / commit messages).
* Local environment changes (`requirements.txt` is the source of truth).
* Lessons-learned content (those go into `deployment_lessons_learned.md`).

The audit question for this doc is: **if a new collaborator reads only
this file, can they navigate the project end-to-end?** When the answer
becomes "no", refresh.
