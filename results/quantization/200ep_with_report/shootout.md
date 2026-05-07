# Quantization Shootout (accuracy only)

Accuracy comparison across FP32 / FP16 / BF16 / INT8 (fake-quant) on the val set. **Latency is intentionally NOT reported here** -- this is a pre-ONNX analysis using PyTorch fake-quant + autocast, both of which add simulation overhead that does not exist in real deploy. The project convention is: pre-ONNX analyses report accuracy only; real per-precision latency lives in the ONNX backend benchmark (`results/onnx_benchmark/.../deploy_summary.md`).

**Three-metric stack**: PSNR (pixel MSE), SSIM (local statistics), LPIPS (CNN feature distance, perceptual). PSNR is the SR-literature headline but under-rates banding / smooth-region artifacts because their per-pixel error is small. LPIPS rises when the perceptual gap widens even if PSNR barely moves -- watch the relative drop ratios across the three metrics, not absolute values.

| Format | PSNR (dB) | PSNR drop | SSIM | SSIM drop | LPIPS | LPIPS rise | Size (MB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 | 27.439 | +0.000 | 0.7907 | +0.0000 | 0.2108 | +0.0000 | 5.23 | baseline |
| FP16 (autocast) | 27.438 | +0.001 | 0.7907 | -0.0000 | 0.2108 | -0.0000 | 2.61 | weights FP32, ops cast on the fly |
| BF16 (autocast) | 27.422 | +0.017 | 0.7904 | +0.0002 | 0.2093 | -0.0016 | 2.61 | wider exponent than FP16, less overflow risk |
| INT8 PTQ (fake-quant) | 27.359 | +0.079 | 0.7863 | +0.0044 | 0.1955 | -0.0154 | 1.31 | fake-quant: q-dq inserted in FP32 math (accuracy-only signal) |
| FP32 (QAT weights) | 27.501 | -0.063 | 0.7932 | -0.0026 | 0.2050 | -0.0058 | 5.23 | QAT-trained weights, fake-quant OFF (isolates training-time effect) |
| INT8 QAT (fake-quant) | 27.446 | -0.007 | 0.7893 | +0.0014 | 0.1900 | -0.0208 | 1.31 | QAT 20-epoch fine-tune from PTQ baseline, lr 1e-5 |
