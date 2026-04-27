# RealSR-Edge: Project Specification

**Document Version**: 1.0  
**Last Updated**: 2026-04-26  
**Project Duration**: 3 weeks (2026-04-26 to 2026-05-17)  
**Status**: In Progress  

---

## 0. Executive Summary

### 0.1 Project Identity

**Name**: RealSR-Edge  
**Tagline**: Realistic Degradation-Aware Super-Resolution with Compiler-Optimized Edge Deployment  
**Repository**: `realsr-edge`  

### 0.2 One-Sentence Definition

> A complete super-resolution pipeline targeting edge AI deployment, featuring realistic degradation-aware training, multi-format quantization analysis, and NN compiler acceleration—designed as a methodological prototype for TV SoC AI IP development.

### 0.3 Strategic Objectives

This project demonstrates three core competencies aligned with edge AI engineering roles:

| Objective | Validation Method | Section |
|---|---|---|
| **End-to-end SR system** | Trained model + benchmark + visual demo | §1 |
| **NN compiler awareness** | torch.compile acceleration + analysis | §2 |
| **Edge AI quantization expertise** | Multi-format comparison + sensitivity analysis | §3 |

### 0.4 Design Philosophy

This project adopts an **edge-first methodology**: every architectural and training decision is evaluated against deployment constraints (model size, latency, precision, operator compatibility). The goal is not algorithmic novelty but **systematic validation of a transferable methodology** applicable to vendor-specific NPUs.

---

## 1. SR Project Specification

### 1.1 Task Definition

**Task Type**: Single Image Super-Resolution (SISR)

**Input/Output**:
- Input: RGB image, shape `(H, W, 3)`, normalized to `[0, 1]`
- Output: RGB image, shape `(2H, 2W, 3)` (2× upscaling)

**Scale Factor**: 2× (primary), 4× (optional, time-permitting)

**Rationale**: 2× upscaling is the most common scenario in TV SoC pipelines (FHD→4K, HD→FHD). Training is fast, evaluation is straightforward, and the methodology generalizes to 4×.

### 1.2 Model Architecture

#### Primary Model: EDSR-baseline

**Specifications**:
- 16 residual blocks
- 64 channels per layer
- No batch normalization (intentional)
- Approximately 1.5M parameters
- Pure convolutional + residual structure

**Selection Rationale**:

| Criterion | EDSR-baseline | Alternative (e.g., SwinIR) |
|---|---|---|
| Operator standardization | ✅ Pure ONNX standard | ❌ LayerNorm, Softmax |
| Quantization friendliness | ✅ Well-studied | ❌ Transformer quantization is complex |
| Train/inference consistency | ✅ No BN | ⚠️ BN folding required |
| Edge compiler compatibility | ✅ Universal | ❌ Limited support |
| Training cost | ✅ Hours on RTX 3090 | ❌ Days |

#### Secondary Model: IMDN (Optional)

If time permits, IMDN serves as a mobile-friendly architecture comparison. Demonstrates awareness of architectural trade-offs in mobile SR design.

### 1.3 Realistic Degradation Pipeline (Project Differentiator)

**Design Intent**: Bridge the gap between academic SR (assuming bicubic degradation) and real TV content (multi-source degradation).

#### Degradation Components

| Component | Real-World Phenomenon | Implementation | Random Range |
|---|---|---|---|
| Blur | Lens defocus, motion blur | Gaussian blur (variable kernel) | σ ∈ [0.1, 2.0] |
| Downsample | Encoder/scaler differences | Random: bicubic / bilinear / area / nearest | Fixed scale=2 |
| Noise | Sensor noise | Gaussian noise (variable σ) | σ ∈ [0, 25/255] |
| Compression | H.264/JPEG artifacts | JPEG re-encoding (variable quality) | Q ∈ [60, 95] |

**Pipeline Order**: Randomized per training sample.

#### Comparison Tracks

| Track | Training Degradation | Purpose |
|---|---|---|
| Track A (Baseline) | Pure bicubic | Academic standard reference |
| Track B (Realistic) | Full random degradation | Proposed methodology |

**Critical Design Constraint**: Both tracks use **identical model architecture, hyperparameters, and training duration**. The only variable is the degradation strategy.

### 1.4 Dataset Specification

#### Training Set

| Dataset | Size | Purpose |
|---|---|---|
| DIV2K (HR images only) | 800 images | Primary training |
| Flickr2K (optional) | 2650 images | Diversity augmentation |

**Note**: We download HR images only and synthesize LR using our degradation pipeline. This is the project's core methodology.

#### Validation Set

DIV2K validation split (100 images).

#### Test Sets (Three Categories)

| Category | Source | Count | Purpose |
|---|---|---|---|
| **Academic Benchmark** | Set5, Set14, BSD100 | ~120 | Standard literature comparison |
| **Real TV Content** | YouTube/Pexels 4K | ~100 | **Project differentiator** |
| **Extreme Degradation** | Synthesized from HR | ~50 | Stress test for Track B advantage |

**Real TV Content Distribution**:
- Nature scenes: 20 images
- Face close-ups: 20 images
- Sports footage: 20 images
- Animation: 20 images
- Text/UI: 20 images

### 1.5 Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Loss function | L1 (primary) | Optionally + perceptual loss |
| Optimizer | Adam | lr=1e-4 |
| Scheduler | Step LR | Decay 0.5 every 100 epochs |
| Batch size | 16 | Constrained by patch + GPU memory |
| Patch size | 96×96 (LR), 192×192 (HR) | Standard SR practice |
| Total epochs | 200 | Each track |
| Hardware | RTX 3090 (24GB) | |
| Estimated training time | 6-12 hours per track | |

### 1.6 Evaluation Protocol

#### Objective Metrics (Required)

| Metric | Library | Notes |
|---|---|---|
| PSNR | `skimage.metrics.peak_signal_noise_ratio` | Standard |
| SSIM | `skimage.metrics.structural_similarity` | Standard |
| LPIPS | `lpips` package, AlexNet backbone | Perceptual quality |

#### Subjective Evaluation (Optional, Time-Permitting)

- 8-12 evaluators
- 20 image comparison pairs
- 1-5 scale rating, blind testing
- Statistical analysis (mean, std, p-value)

#### Visualization Outputs

Required deliverables:
- Comparison table (markdown + CSV)
- Side-by-side visual comparisons (LR / Bicubic / Track A / Track B / HR)
- Multi-dimensional radar charts
- Failure case analysis (where Track B underperforms)
- 30-second demo video

---

## 2. torch.compile Acceleration Specification

### 2.1 Objective

Demonstrate practical use and theoretical understanding of NN compilers, specifically `torch.compile`, as evidence of compiler-aware ML engineering capability.

### 2.2 Experiment Matrix

| Mode | Compilation Cost | Expected Speedup | Use Case |
|---|---|---|---|
| Eager (baseline) | None | 1.0× | Reference |
| `torch.compile()` default | Medium | 1.3-1.7× | General use |
| `mode='reduce-overhead'` | Medium | 1.5-1.8× | Small model, frequent inference |
| `mode='max-autotune'` | High (minutes) | 1.8-2.5× | Production deployment |

### 2.3 Measurement Protocol

| Parameter | Value |
|---|---|
| Hardware | RTX 3090 (CUDA) |
| Input shape | (1, 3, 256, 256) |
| Output shape | (1, 3, 512, 512) |
| Warmup iterations | 10 |
| Measurement iterations | 100 (averaged) |
| Synchronization | `torch.cuda.synchronize()` mandatory |

**Metrics Captured**:
- Latency (ms): mean ± std
- Throughput (fps): 1000/latency
- Compilation time (one-time cost)
- GPU memory peak usage

### 2.4 Deep-Dive Analysis Components

To demonstrate principle understanding (not just usage):

#### A. Graph Visualization
- Use `torch._dynamo.explain()` to capture computation graph
- Document graph break points
- Screenshots for presentation deck

#### B. Generated Code Inspection
- Set `TORCH_LOGS="output_code"` environment variable
- Capture sample Triton kernel code
- Identify fusion opportunities exploited

#### C. Fusion Behavior Analysis
- Compare kernel count: pre-compile vs post-compile
- Quantify memory bandwidth savings from fusion
- Document specific patterns (Conv→ReLU, Conv→BN→ReLU)

### 2.5 Documentation Output

**File**: `docs/torch_compile_analysis.md`

**Required Sections**:
1. Motivation: Why NN models benefit from compilation
2. torch.compile architecture: TorchDynamo / AOTAutograd / Inductor
3. Experimental results (latency comparison table)
4. Principle correspondence analysis (where fusion saves cost)
5. Comparison with ONNX, TVM, TensorRT
6. Implications for vendor-specific NPU compilers (NDPU bridge)
7. Limitations and unverified scope (honest boundary marking)

### 2.6 Strategic Framing

This component positions the candidate as **compiler-aware ML engineer (L2)** rather than **compiler internal developer (L1)**. The deliverables demonstrate ability to:
- Use NN compilers effectively
- Understand internal mechanisms at architectural level
- Transfer principles to vendor-specific compilers
- Communicate effectively with compiler engineering teams

---

## 3. Edge AI Quantization Specification

### 3.1 Objective

Demonstrate comprehensive familiarity with edge AI quantization formats through systematic comparison and sensitivity analysis on SR tasks.

### 3.2 Format Coverage Matrix

| Format | Tool | Implementation | Expected PSNR Loss |
|---|---|---|---|
| FP32 | PyTorch native | Baseline | 0 dB (reference) |
| FP16 | PyTorch autocast | Implemented | < 0.1 dB |
| BF16 | PyTorch autocast | Implemented | < 0.1 dB |
| INT8 PTQ static | ONNX Runtime | Implemented | 0.3-0.8 dB |
| INT8 PTQ dynamic | ONNX Runtime | Implemented | 0.5-1.2 dB |
| INT8 QAT | PyTorch QAT | Implemented | 0.1-0.4 dB |
| Mixed Precision (FP16+INT8) | Custom config | Implemented | 0.2-0.5 dB |
| INT4 | Literature analysis | Documented only | 1-3 dB (per literature) |

### 3.3 Evaluation Dimensions

For each format, measure:

| Dimension | Description |
|---|---|
| PSNR loss | Relative to FP32 baseline |
| Latency | ONNX Runtime CPU + CUDA |
| Model size | File size (MB) |
| Memory peak | Maximum runtime memory |

### 3.4 Advanced Analysis Components

#### A. Per-Layer Sensitivity Analysis

**Method**: Sequential single-layer quantization with full-precision elsewhere; measure individual layer contribution to PSNR loss.

**Output**: 
- Per-layer PSNR drop heatmap
- Identification of "quantization-critical" layers
- Recommended high-precision retention list (typically first/last conv)

**Strategic Value**: Demonstrates the diagnostic-first mindset essential for edge deployment debugging.

#### B. Calibration Dataset Impact Analysis

Test PTQ accuracy with calibration set sizes: 10 / 50 / 100 / 500 images.

**Hypothesis**: Diminishing returns above ~100 representative samples.

#### C. Format Selection Decision Tree

Given inputs:
- Hardware operator support
- Precision tolerance
- Model size budget

Output: Recommended quantization strategy.

Visualized as flowchart for documentation and presentation.

### 3.5 Literature Grounding

Quantization decisions must be grounded in SR-specific literature:

| Reference | Contribution to This Project |
|---|---|
| PAMS (ECCV 2020) | Learned activation clamp (PyTorch FakeQuantize equivalent) |
| Tu et al. (CVPR 2023) | Three quantization-unfriendly properties of SR activations |
| 2DQuant (NeurIPS 2024) | Two-stage PTQ + distillation methodology |
| DAQ (WACV 2022) | Per-channel weight quantization for SR |

**SR Quantization Difficulty (Documented Justification)**:
1. Most SR models lack BN, leading to large dynamic activation ranges
2. SR activations exhibit long-tailed distributions
3. Activation ranges vary highly per input sample

### 3.6 Documentation Output

**File**: `docs/quantization_formats.md`

**Required Sections**:
1. Quantization fundamentals
2. Format specifications (each format ~200 words)
3. Experimental result matrix
4. Per-layer sensitivity analysis
5. Edge platform support matrix (TensorRT, SNPE, NeuroPilot, ONNX Runtime, TFLite)
6. Format selection decision tree
7. NDPU adaptation recommendations (bridge section)

### 3.7 Failure Mode Analysis

Document handling of "INT8 accuracy too low to deploy" scenario as a five-layer escalation strategy:

1. **Diagnose**: Per-layer sensitivity analysis
2. **Improve calibration**: Algorithm + dataset + per-channel
3. **Mixed precision**: Retain critical layers in FP16
4. **QAT**: Re-train with quantization simulation
5. **Architecture redesign**: Replace quantization-hostile operators

This escalation strategy is itself a deliverable demonstrating senior-level problem decomposition.

---

## 4. Integration & Deployment Specification

### 4.1 End-to-End Pipeline

```