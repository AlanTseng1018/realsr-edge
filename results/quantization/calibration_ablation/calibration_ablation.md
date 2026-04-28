# Calibration Method Ablation

## What was tested

- **Generated**: 2026-04-28T12:33:05
- **Checkpoint**: `results\runs\20260427_143542_ep200_b16_scale2_realistic\checkpoints\best.pt`
  - last modified: 2026-04-28T10:19:45, size: 15.75 MB
- **Model**: EDSR(scale_factor=2, n_resblocks=16, n_feats=64) -- 1,369,859 params
- **Device**: cuda (NVIDIA GeForce RTX 3060 Laptop GPU), PyTorch 2.6.0+cu124
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` -- 100 images, realistic degradation, LR patch 96x96
- **Calibration**: 8 batches (64 LR samples)
- **Histogram bins**: 2048
- **Quantization scheme**: symmetric per-tensor INT8 (activations) + symmetric per-channel INT8 (weights)

## Calibration scheme shootout

All four schemes share the **same calibration pass** -- the histogram is collected once, and each scheme just chooses a different summary of it (running max for `max-abs`, percentile cutoff for the other three).

| Scheme | PSNR (dB) | Drop vs FP32 | Latency (ms) |
|---|---:|---:|---:|
| max-abs | 27.361 | +0.077 | 13.19 +/- 2.62 |
| percentile-99.99 | 27.358 | +0.080 | 12.35 +/- 2.59 |
| percentile-99.9 | 26.975 | +0.464 | 15.08 +/- 6.96 |
| percentile-99.0 | 25.071 | +2.367 | 17.13 +/- 8.02 |

## Per-layer chosen `amax` per scheme

This is the value that drives `scale = amax / 127` for each layer's input quantizer. Smaller values = tighter clipping = the tail of the activation distribution gets saturated.

| Layer | max-abs | percentile-99.99 | percentile-99.9 | percentile-99.0 |
|---|---:|---:|---:|---:|
| `head` | 1.0000 | 1.0000 | 0.9999 | 0.9801 |
| `tail` | 1.9936 | 1.1900 | 0.9742 | 0.7036 |
| `body.16` | 2.8580 | 1.5147 | 1.0470 | 0.5715 |
| `body.0.conv1` | 1.4732 | 1.3451 | 1.1518 | 0.8445 |
| `body.0.conv2` | 1.3181 | 0.7344 | 0.6115 | 0.4473 |
| `body.1.conv1` | 1.2136 | 0.8612 | 0.7235 | 0.4945 |
| `body.1.conv2` | 1.5933 | 0.6711 | 0.4112 | 0.2764 |
| `body.2.conv1` | 1.1992 | 0.7375 | 0.5445 | 0.3543 |
| `body.2.conv2` | 1.1804 | 0.6860 | 0.4163 | 0.1847 |
| `body.3.conv1` | 1.4851 | 0.7337 | 0.5327 | 0.3286 |
| `body.3.conv2` | 1.7200 | 0.7219 | 0.4393 | 0.1983 |
| `body.4.conv1` | 1.3005 | 0.7604 | 0.5472 | 0.3328 |
| `body.4.conv2` | 1.1257 | 0.5436 | 0.3603 | 0.1746 |
| `body.5.conv1` | 1.3610 | 0.7812 | 0.5551 | 0.3306 |
| `body.5.conv2` | 1.5813 | 0.6944 | 0.3862 | 0.1820 |
| `body.6.conv1` | 1.4015 | 0.7971 | 0.5686 | 0.3389 |
| `body.6.conv2` | 1.5073 | 0.7317 | 0.4281 | 0.1795 |
| `body.7.conv1` | 1.5146 | 0.8233 | 0.5855 | 0.3469 |
| `body.7.conv2` | 1.3768 | 0.6248 | 0.3424 | 0.1697 |
| `body.8.conv1` | 1.5545 | 0.8447 | 0.5954 | 0.3510 |
| `body.8.conv2` | 1.5857 | 0.7864 | 0.3889 | 0.1808 |
| `body.9.conv1` | 1.7315 | 0.9011 | 0.6362 | 0.3711 |
| `body.9.conv2` | 1.4081 | 0.6993 | 0.4119 | 0.1891 |
| `body.10.conv1` | 1.7531 | 0.9527 | 0.6754 | 0.3957 |
| `body.10.conv2` | 1.5539 | 0.6943 | 0.3840 | 0.1823 |
| `body.11.conv1` | 1.8397 | 0.9989 | 0.7070 | 0.4094 |
| `body.11.conv2` | 1.5828 | 0.7823 | 0.4055 | 0.1965 |
| `body.12.conv1` | 1.8615 | 1.0506 | 0.7462 | 0.4300 |
| `body.12.conv2` | 1.7914 | 0.9117 | 0.4699 | 0.2226 |
| `body.13.conv1` | 2.2329 | 1.1492 | 0.7985 | 0.4529 |
| `body.13.conv2` | 2.0726 | 1.0194 | 0.5123 | 0.2386 |
| `body.14.conv1` | 2.3161 | 1.3016 | 0.8800 | 0.4883 |
| `body.14.conv2` | 1.8776 | 0.9322 | 0.5364 | 0.2534 |
| `body.15.conv1` | 2.7716 | 1.4158 | 0.9648 | 0.5256 |
| `body.15.conv2` | 2.2785 | 1.1466 | 0.6364 | 0.2742 |
| `upsampler.0` | 2.5938 | 1.5623 | 1.2166 | 0.8747 |

## How to read the histogram figure

`histograms.png` plots the activation-magnitude histogram (collected during calibration) for six representative layers. Vertical dashed lines mark where each scheme places its `amax`:

- **Red (max-abs)**: at the largest `|x|` ever seen. Most conservative (no clipping) but exposed to single outliers.
- **Orange (99.99)**: cut off the very last 0.01% of the tail.
- **Green (99.9)**: typical TensorRT-style aggressive choice.
- **Blue (99.0)**: aggressive clipping; saturates more than just outliers.

Y-axis is log-scale because activation distributions are heavily long-tailed -- a linear axis would render the tail invisible. The amax differences look small numerically, but the resulting INT8 scale (`amax / 127`) is what determines the bin resolution for the bulk of values, which is what shows up in the PSNR table above.

## Takeaway

On this checkpoint and val set, **`max-abs`** wins the shootout (27.361 dB). The spread across schemes is 2.290 dB.

Interpret carefully:

- A small spread (< 0.05 dB) means activations are NOT outlier-heavy for this model -- the calibration choice is mostly cosmetic.
- A large spread (> 0.2 dB) means there ARE outliers that max-abs is exposed to. In that case, percentile clipping is a real win.
- If `percentile-99.0` is best, the tail isn't useful; consider an even tighter cutoff or a learned clipping threshold (PAMS-style).
- If `max-abs` is best, the tail IS informative; percentile clipping is throwing away useful signal.

All four numbers are produced from the same FP32 weights and the same calibration histogram. The infrastructure is in place to add KL-div or MSE-optimal calibration as additional schemes by adding entries to `SCHEMES` in `calibration_ablation.py`.
