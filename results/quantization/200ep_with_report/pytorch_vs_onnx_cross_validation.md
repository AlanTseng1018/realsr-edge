# Stage-2 (PyTorch Fake-Quant) vs Stage-3 (ONNX Deploy) — Cross-Validation Report

**Generated**: 2026-05-17
**Source data**: DIV2K val (100 HR images, realistic degradation, deterministic per-index seed)
**Source checkpoint**: `results/checkpoints/edsr_baseline/final.pt`
**ONNX exports**: `results/onnx_exports/edsr_200ep/{edsr_fp32, edsr_int8_static}.onnx`
**FP32 ONNX verification**: PASS @ atol=1e-04 (see `results/onnx_exports/edsr_200ep/verification.md`)
**INT8 ONNX**: ORT `quantize_static`, QDQ format, QInt8 symmetric per-tensor activations, QInt8 symmetric per-channel weights, MinMax calibration

---

## Purpose

Validate that the spatial perceptual / structural detectors developed in Stage 2 (PyTorch fake-quant) produce the same qualitative conclusions when re-run on Stage 3 (ONNX deploy artefacts). If the two backends diverged in direction (not just magnitude), Stage-2 optimization decisions would not necessarily transfer to deploy, and the pipeline would have a methodology gap.

## Method

The same two scripts run with `--source pytorch` (Stage 2) and `--source onnx` (Stage 3):

- `src/quantization/lpips_heatmap.py` — per-image LPIPS distribution + spatial LPIPS heatmap on a target image
- `src/quantization/structural_heatmap.py` — per-image gradient-orientation structural delta + 9-panel structural heatmap + val-set scan

Backend swap is the only change; LPIPS network (SqueezeNet), HR cropping, deterministic LR generation, target image (0879.png), and metric definitions are identical across both runs.

## 1. Per-image LPIPS distribution (n = 100 val images)

| Metric | PyTorch (Stage 2) | ONNX (Stage 3) | Δ (ONNX − PT) | Interpretation |
|---|---:|---:|---:|---|
| FP32 LPIPS — mean   | 0.2108 | 0.2108 | 0.0000 | FP32 export is bit-faithful (matches verification.md @ atol=1e-04) |
| FP32 LPIPS — median | 0.2129 | 0.2129 | 0.0000 | Same |
| INT8 LPIPS — mean   | 0.1954 | 0.1841 | −0.0113 | ONNX INT8 scores ~6% lower (better) LPIPS than fake-quant predicted |
| INT8 LPIPS — median | 0.1874 | 0.1762 | −0.0112 | Same direction |
| `lpips_rise` (signed INT8−FP32) — mean | −0.0154 | −0.0268 | −0.0114 | **ONNX shows ~74% more INT8-induced perceptual gain than fake-quant predicted** |
| `int8_vs_fp32_lpips` — mean | 0.0083 | 0.0235 | +0.0152 | ONNX INT8 diverges from its FP32 ~2.8× more in feature space than fake-quant predicted |

**Take**: directions agree (INT8 perceptually beneficial in both backends). Magnitudes differ — fake-quant systematically under-predicts the deploy-side perceptual delta.

## 2. Per-image agreement on "which images change most"

### Top-10 by most-negative `lpips_rise` (largest INT8-induced perceptual gain)

- PyTorch: `0893, 0857, 0845, 0812, 0838, 0855, 0839, 0821, 0866, 0889`
- ONNX:    `0893, 0838, 0857, 0855, 0845, 0839, 0812, 0864, 0836, 0889`
- **Overlap: 8/10** = `0812, 0838, 0839, 0845, 0855, 0857, 0889, 0893`

### Top-10 by largest GT-vs-INT8 structural angular delta

- PyTorch: `0862, 0834, 0852, 0895, 0865, 0894, 0863, 0896, 0848, 0826`
- ONNX:    `0862, 0834, 0852, 0895, 0865, 0894, 0896, 0807, 0890, 0826`
- **Overlap: 8/10** = `0826, 0834, 0852, 0862, 0865, 0894, 0895, 0896`

**Take**: per-image patterns are strongly preserved across backends. Fake-quant predictions of *which* images get the largest LPIPS gain or most structural distortion are reliable, even though absolute numbers differ. Selective-mixed-precision strategies that target the worst per-image cases transfer from Stage 2 to Stage 3.

## 3. 0879 case study (target image used for heatmap PNGs in README §3.2)

| Detector | PyTorch | ONNX | Δ (ONNX − PT) |
|---|---:|---:|---:|
| LPIPS spatial mean (INT8 vs FP32) | 0.0097 | 0.0235 | +0.0138 (≈2.4×) |
| Structural Δ — GT vs FP32 (mean over edges) | 10.68° | 9.81° | −0.87° |
| Structural Δ — GT vs INT8 (mean over edges) | 10.69° | 9.86° | −0.83° |
| Structural Δ — FP32 vs INT8 (mean over edges) | 0.58° | 0.96° | +0.38° |
| Auto-zoom worst-region mean Δ (256-px) | 12.24° | 11.56° | −0.68° |

**Visual evidence**:

- LPIPS heatmaps: [`lpips_heatmaps/heatmap_0879.png`](lpips_heatmaps/heatmap_0879.png) (PyTorch) · [`lpips_heatmaps/heatmap_0879_onnx.png`](lpips_heatmaps/heatmap_0879_onnx.png) (ONNX)
- Structural 9-panel: [`structural_heatmaps/structural_heatmap_0879.png`](structural_heatmaps/structural_heatmap_0879.png) (PyTorch) · [`structural_heatmaps/structural_heatmap_0879_onnx.png`](structural_heatmaps/structural_heatmap_0879_onnx.png) (ONNX)

## 4. Conclusions

1. **Detector backend port is clean.** Both LPIPS and structural detectors produce qualitatively identical results across PyTorch fake-quant and ONNX deploy. Stage-2 optimization decisions can be carried into Stage-3 deploy with confidence.

2. **Fake-quant under-predicts deploy-side spatial delta** by ~2.4× on spatial LPIPS and ~1.6× on FP32-vs-INT8 structural. Most likely cause: ORT `quantize_static` uses MinMax calibration on the activation distribution, while PyTorch fake-quant uses max-abs over the calibration set. The two methods produce mildly different scales; the divergence is amplified by the cascading effect of activation quantization through 36 conv layers + 17 skip-Add ops. Direction is preserved; magnitude is not.

3. **Per-image ranking is preserved** — 8/10 overlap on top-10 worst-rise for both LPIPS and structural detectors. This validates that fake-quant predictions of which images need special attention (e.g., for selective mixed precision per image) transfer to deploy. Absolute thresholds may need re-tuning per backend; priority order does not.

4. **The "INT8 perceptually wins" finding survives deployment** — and is in fact more pronounced on ONNX. Vendor-handoff messaging should default to the ONNX numbers (0.184 LPIPS vs fake-quant 0.196), with the fake-quant figure noted as the conservative Stage-2 prediction.

## Reproduce

```bash
# Stage 2 (PyTorch fake-quant) — default --source pytorch
python -m src.quantization.lpips_heatmap \
    --checkpoint results/checkpoints/edsr_baseline/final.pt \
    --output-dir results/quantization/200ep_with_report/lpips_heatmaps \
    --target-image 0879.png

python -m src.quantization.structural_heatmap \
    --checkpoint results/checkpoints/edsr_baseline/final.pt \
    --output-dir results/quantization/200ep_with_report/structural_heatmaps \
    --target-image 0879.png

# Stage 3 (ONNX deploy) — same scripts, --source onnx
python -m src.quantization.lpips_heatmap --source onnx \
    --fp32-onnx results/onnx_exports/edsr_200ep/edsr_fp32.onnx \
    --int8-onnx results/onnx_exports/edsr_200ep/edsr_int8_static.onnx \
    --output-dir results/quantization/200ep_with_report/lpips_heatmaps \
    --target-image 0879.png

python -m src.quantization.structural_heatmap --source onnx \
    --fp32-onnx results/onnx_exports/edsr_200ep/edsr_fp32.onnx \
    --int8-onnx results/onnx_exports/edsr_200ep/edsr_int8_static.onnx \
    --output-dir results/quantization/200ep_with_report/structural_heatmaps \
    --target-image 0879.png
```

Outputs land in the same directories with `_onnx` suffix so Stage-2 and Stage-3 artefacts coexist.
