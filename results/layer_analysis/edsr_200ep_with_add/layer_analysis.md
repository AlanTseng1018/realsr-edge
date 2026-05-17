# Per-Layer Quantization Precision Analysis

- **Source**: `results\runs\20260427_143542_ep200_b16_scale2_realistic\checkpoints\best.pt`
- **Generated**: 2026-05-14T18:27:52
- **E2E fake-INT8 PSNR** (FP32 vs all-layers-quantized): **45.65 dB**

> Table sorted by **isolated PSNR** ascending — most sensitive layers appear first.
> Isolated PSNR: PSNR between FP32 output and output when *only this layer* is INT8 fake-quantized.

| # | Layer | Shape | W amax | W scale (mean) | Act amax | Act scale | Isolated PSNR (dB) |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | `tail` | 3×64×3×3 | 4.5065e-02 | 3.4698e-04 | 1.6690e+00 | 1.3141e-02 | **53.1** |
| 2 | `upsampler.0` | 256×64×3×3 | 5.8228e-02 | 3.8672e-04 | 2.2971e+00 | 1.8087e-02 | **55.1** |
| 3 | `long_skip_add` |  | nan | nan | nan | nan | **55.1** |
| 4 | `head` | 64×3×3×3 | 2.0110e-01 | 1.4662e-03 | 1.0000e+00 | 7.8740e-03 | **56.7** |
| 5 | `body.16` | 64×64×3×3 | 7.3276e-02 | 3.9106e-04 | 2.7833e+00 | 2.1916e-02 | **58.5** |
| 6 | `body.15.skip_add` |  | nan | nan | nan | nan | **59.3** |
| 7 | `body.14.skip_add` |  | nan | nan | nan | nan | **59.7** |
| 8 | `body.0.skip_add` |  | nan | nan | nan | nan | 60.1 |
| 9 | `body.13.skip_add` |  | nan | nan | nan | nan | 60.4 |
| 10 | `body.12.skip_add` |  | nan | nan | nan | nan | 61.1 |
| 11 | `body.11.skip_add` |  | nan | nan | nan | nan | 61.8 |
| 12 | `body.1.skip_add` |  | nan | nan | nan | nan | 62.0 |
| 13 | `body.10.skip_add` |  | nan | nan | nan | nan | 62.4 |
| 14 | `body.9.skip_add` |  | nan | nan | nan | nan | 62.5 |
| 15 | `body.7.skip_add` |  | nan | nan | nan | nan | 62.8 |
| 16 | `body.6.skip_add` |  | nan | nan | nan | nan | 62.8 |
| 17 | `body.5.skip_add` |  | nan | nan | nan | nan | 62.9 |
| 18 | `body.3.skip_add` |  | nan | nan | nan | nan | 63.0 |
| 19 | `body.2.skip_add` |  | nan | nan | nan | nan | 63.1 |
| 20 | `body.8.skip_add` |  | nan | nan | nan | nan | 63.2 |
| 21 | `body.4.skip_add` |  | nan | nan | nan | nan | 63.3 |
| 22 | `body.15.conv2` | 64×64×3×3 | 1.0774e-01 | 5.7662e-04 | 2.1779e+00 | 1.7149e-02 | 68.6 |
| 23 | `body.12.conv2` | 64×64×3×3 | 1.0983e-01 | 5.9701e-04 | 1.9391e+00 | 1.5268e-02 | 69.1 |
| 24 | `body.13.conv2` | 64×64×3×3 | 1.0360e-01 | 5.8105e-04 | 1.8952e+00 | 1.4922e-02 | 69.4 |
| 25 | `body.1.conv2` | 64×64×3×3 | 2.6036e-01 | 8.3521e-04 | 1.3671e+00 | 1.0765e-02 | 69.6 |
| 26 | `body.14.conv2` | 64×64×3×3 | 8.4266e-02 | 4.8997e-04 | 1.8779e+00 | 1.4786e-02 | 70.0 |
| 27 | `body.3.conv2` | 64×64×3×3 | 1.0586e-01 | 6.0089e-04 | 1.4048e+00 | 1.1062e-02 | 70.4 |
| 28 | `body.2.conv2` | 64×64×3×3 | 1.0695e-01 | 6.1316e-04 | 1.3021e+00 | 1.0253e-02 | 70.8 |
| 29 | `body.0.conv2` | 64×64×3×3 | 1.0787e-01 | 5.8597e-04 | 1.0999e+00 | 8.6608e-03 | 70.8 |
| 30 | `body.8.conv2` | 64×64×3×3 | 1.0989e-01 | 5.8081e-04 | 1.3358e+00 | 1.0518e-02 | 71.3 |
| 31 | `body.11.conv2` | 64×64×3×3 | 9.7052e-02 | 5.5824e-04 | 1.3760e+00 | 1.0835e-02 | 71.6 |
| 32 | `body.5.conv2` | 64×64×3×3 | 8.8891e-02 | 5.2300e-04 | 1.2221e+00 | 9.6231e-03 | 71.7 |
| 33 | `body.15.conv1` | 64×64×3×3 | 1.3434e-01 | 4.7522e-04 | 2.4347e+00 | 1.9171e-02 | 71.9 |
| 34 | `body.6.conv2` | 64×64×3×3 | 9.4206e-02 | 5.3280e-04 | 1.1687e+00 | 9.2024e-03 | 72.1 |
| 35 | `body.10.conv2` | 64×64×3×3 | 1.1787e-01 | 5.3778e-04 | 1.3619e+00 | 1.0723e-02 | 72.4 |
| 36 | `body.7.conv2` | 64×64×3×3 | 9.8625e-02 | 5.8390e-04 | 1.1122e+00 | 8.7575e-03 | 72.5 |
| 37 | `body.13.conv1` | 64×64×3×3 | 1.0527e-01 | 4.7507e-04 | 2.0681e+00 | 1.6284e-02 | 72.8 |
| 38 | `body.14.conv1` | 64×64×3×3 | 8.0622e-02 | 4.3464e-04 | 2.2963e+00 | 1.8081e-02 | 72.8 |
| 39 | `body.0.conv1` | 64×64×3×3 | 1.2087e-01 | 4.7656e-04 | 1.4314e+00 | 1.1271e-02 | 73.2 |
| 40 | `body.9.conv2` | 64×64×3×3 | 1.0163e-01 | 5.3751e-04 | 1.1662e+00 | 9.1826e-03 | 73.3 |
| 41 | `body.12.conv1` | 64×64×3×3 | 9.6064e-02 | 4.7233e-04 | 1.8614e+00 | 1.4657e-02 | 73.7 |
| 42 | `body.11.conv1` | 64×64×3×3 | 1.0097e-01 | 4.6157e-04 | 1.6809e+00 | 1.3235e-02 | 74.0 |
| 43 | `body.4.conv2` | 64×64×3×3 | 9.1803e-02 | 5.3614e-04 | 9.0238e-01 | 7.1054e-03 | 74.7 |
| 44 | `body.7.conv1` | 64×64×3×3 | 9.9503e-02 | 4.6908e-04 | 1.3698e+00 | 1.0785e-02 | 74.7 |
| 45 | `body.1.conv1` | 64×64×3×3 | 1.1109e-01 | 4.8579e-04 | 1.2062e+00 | 9.4975e-03 | 75.1 |
| 46 | `body.2.conv1` | 64×64×3×3 | 1.4398e-01 | 4.7077e-04 | 1.1382e+00 | 8.9623e-03 | 75.2 |
| 47 | `body.9.conv1` | 64×64×3×3 | 9.6094e-02 | 4.6485e-04 | 1.4968e+00 | 1.1786e-02 | 75.3 |
| 48 | `body.3.conv1` | 64×64×3×3 | 1.1914e-01 | 5.0032e-04 | 1.1199e+00 | 8.8181e-03 | 75.3 |
| 49 | `body.5.conv1` | 64×64×3×3 | 8.8167e-02 | 4.7569e-04 | 1.2648e+00 | 9.9588e-03 | 75.3 |
| 50 | `body.10.conv1` | 64×64×3×3 | 1.1012e-01 | 4.5426e-04 | 1.5747e+00 | 1.2400e-02 | 75.4 |
| 51 | `body.6.conv1` | 64×64×3×3 | 9.1737e-02 | 4.5925e-04 | 1.2943e+00 | 1.0191e-02 | 75.4 |
| 52 | `body.8.conv1` | 64×64×3×3 | 1.1811e-01 | 4.5607e-04 | 1.3440e+00 | 1.0582e-02 | 75.5 |
| 53 | `body.4.conv1` | 64×64×3×3 | 7.9549e-02 | 4.5437e-04 | 1.1639e+00 | 9.1644e-03 | 76.8 |

## Glossary

| Term | Definition |
|---|---|
| W amax | max(abs(weight)) — full FP32 dynamic range |
| W scale (mean) | mean per-channel INT8 scale = amax_per_ch / 127 |
| Act amax | max-abs activation seen during calibration (DIV2K val) |
| Act scale | activation amax / 127 — the INT8 quantization step |
| Isolated PSNR | PSNR(FP32 output, output with only this layer quantized) |
