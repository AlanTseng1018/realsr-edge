# Per-layer Quantization Sensitivity

Each row: hold every other layer in FP32, fake-quantize ONLY this Conv2d to INT8, measure PSNR drop. Larger drop = layer is more quantization-sensitive.

| Rank | Layer | PSNR (dB) | Drop (dB) | Class |
|---:|---|---:|---:|---|
| 1 | `tail` | 27.409 | +0.029 | output |
| 2 | `upsampler.0` | 27.418 | +0.020 | upsampler |
| 3 | `head` | 27.423 | +0.016 | input |
| 4 | `body.16` | 27.432 | +0.007 | post-resblock |
| 5 | `body.0.conv2` | 27.438 | +0.001 | resblock-interior |
| 6 | `body.15.conv2` | 27.438 | +0.001 | resblock-interior |
| 7 | `body.1.conv2` | 27.438 | +0.001 | resblock-interior |
| 8 | `body.14.conv2` | 27.438 | +0.001 | resblock-interior |
| 9 | `body.5.conv2` | 27.438 | +0.001 | resblock-interior |
| 10 | `body.13.conv2` | 27.438 | +0.001 | resblock-interior |
| 11 | `body.12.conv2` | 27.438 | +0.001 | resblock-interior |
| 12 | `body.6.conv2` | 27.438 | +0.001 | resblock-interior |
| 13 | `body.10.conv2` | 27.438 | +0.000 | resblock-interior |
| 14 | `body.15.conv1` | 27.438 | +0.000 | resblock-interior |
| 15 | `body.3.conv1` | 27.438 | +0.000 | resblock-interior |
| 16 | `body.8.conv2` | 27.438 | +0.000 | resblock-interior |
| 17 | `body.11.conv2` | 27.438 | +0.000 | resblock-interior |
| 18 | `body.7.conv2` | 27.438 | +0.000 | resblock-interior |
| 19 | `body.9.conv2` | 27.438 | +0.000 | resblock-interior |
| 20 | `body.0.conv1` | 27.438 | +0.000 | resblock-interior |
| 21 | `body.14.conv1` | 27.438 | +0.000 | resblock-interior |
| 22 | `body.13.conv1` | 27.438 | +0.000 | resblock-interior |
| 23 | `body.8.conv1` | 27.438 | +0.000 | resblock-interior |
| 24 | `body.11.conv1` | 27.438 | +0.000 | resblock-interior |
| 25 | `body.6.conv1` | 27.438 | +0.000 | resblock-interior |
| 26 | `body.12.conv1` | 27.438 | +0.000 | resblock-interior |
| 27 | `body.9.conv1` | 27.438 | +0.000 | resblock-interior |
| 28 | `body.10.conv1` | 27.438 | +0.000 | resblock-interior |
| 29 | `body.7.conv1` | 27.438 | +0.000 | resblock-interior |
| 30 | `body.4.conv1` | 27.438 | +0.000 | resblock-interior |
| 31 | `body.5.conv1` | 27.438 | +0.000 | resblock-interior |
| 32 | `body.4.conv2` | 27.438 | +0.000 | resblock-interior |
| 33 | `body.2.conv2` | 27.439 | -0.000 | resblock-interior |
| 34 | `body.2.conv1` | 27.439 | -0.000 | resblock-interior |
| 35 | `body.1.conv1` | 27.439 | -0.000 | resblock-interior |
| 36 | `body.3.conv2` | 27.439 | -0.000 | resblock-interior |
