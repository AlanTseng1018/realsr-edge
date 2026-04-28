# Quantization Shootout

| Format | PSNR (dB) | drop vs FP32 | Latency (ms) | Size (MB) | Notes |
|---|---:|---:|---:|---:|---|
| FP32 | 27.439 | +0.000 | 3.89 +/- 0.04 | 5.23 | baseline |
| FP16 (autocast) | 27.438 | +0.001 | 3.54 +/- 0.15 | 2.61 | weights FP32, ops cast on the fly |
| BF16 (autocast) | 27.422 | +0.017 | 3.85 +/- 0.18 | 2.61 | wider exponent than FP16, less overflow risk |
| INT8 PTQ (fake-quant) | 27.359 | +0.079 | 10.71 +/- 1.43 | 1.31 | PSNR is real; latency is fake-quant overhead (NOT deploy latency) |
