# Quantization Shootout

| Format | PSNR (dB) | drop vs FP32 | Latency (ms) | Size (MB) | Notes |
|---|---:|---:|---:|---:|---|
| FP32 | 27.439 | +0.000 | 4.41 +/- 0.52 | 5.23 | baseline |
| FP16 (autocast) | 27.438 | +0.001 | 5.95 +/- 0.27 | 2.61 | weights FP32, ops cast on the fly |
| BF16 (autocast) | 27.422 | +0.017 | 6.03 +/- 0.29 | 2.61 | wider exponent than FP16, less overflow risk |
| INT8 PTQ (fake-quant) | 27.362 | +0.077 | 15.28 +/- 0.32 | 1.31 | PSNR is real; latency is fake-quant overhead (NOT deploy latency) |
