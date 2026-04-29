# ONNX Export Pipeline Output

Single execution node of the multi-precision ONNX export pipeline. All three ONNX artifacts in this folder come from the same source checkpoint and the same calibration set; they are directly comparable.

## Source

- **Checkpoint**: `results\runs\20260427_143542_ep200_b16_scale2_realistic\checkpoints\best.pt`
  - mtime: 2026-04-28T10:19:45, size: 15.75 MB
- **Model**: EDSR(scale_factor=2, n_resblocks=16, n_feats=64) -- 1,369,859 params
- **Generated**: 2026-04-30T01:00:12
- **Device used for verification**: cuda (NVIDIA GeForce RTX 3060 Laptop GPU)
- **PyTorch**: 2.6.0+cu124, **ONNX**: 1.21.0, **ORT**: 1.25.0

## Calibration set

- **Source**: `data\DIV2K\DIV2K_valid_HR` (100 images, realistic degradation, deterministic seed)
- **Samples used**: 64 LR images, batch 8 -> 8 batches
- **LR patch size**: 96x96

## Artifacts

| File | Size (MB) | Verified vs PyTorch |
|---|---:|:---:|
| `edsr_fp32.onnx` | 5.24 | PASS (atol=1e-04) |
| `edsr_fp16.onnx` | 2.63 | PASS (atol=5e-02) |
| `edsr_int8_static.onnx` | 1.43 | PASS (atol=1e-01) |

Per-shape numeric diffs are in `verification.md`. The raw shape and size info is in `metadata.json` for programmatic consumption.

## Quantization scheme (INT8)

- **Tool**: ONNX Runtime `quantize_static`
- **Format**: QDQ
- **Activations**: QInt8 (symmetric per-tensor)
- **Weights**: QInt8 (symmetric per-channel), per-channel = True
- **Calibration method**: ORT default (MinMax)

## Next steps

These three ONNX files become the input for:

1. `benchmark_onnx.py` (planned) -- runs each on the val set, outputs per-format PSNR + latency + memory across providers.
2. `cpp_inference/sr_cli` -- C++ deploy reference; can load any of the three.
3. Vendor toolchains (TensorRT / SNPE / NeuroPilot) -- consume `edsr_fp32.onnx` and produce backend-native engines.
