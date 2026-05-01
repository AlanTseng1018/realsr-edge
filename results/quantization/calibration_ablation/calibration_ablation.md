# Calibration Method Ablation

## What was tested

- **Generated**: 2026-05-01T21:54:05
- **Checkpoint**: `results\runs\20260427_143542_ep200_b16_scale2_realistic\checkpoints\best.pt`
  - last modified: 2026-04-27T15:27:43, size: 15.75 MB
- **Model**: EDSR(scale_factor=2, n_resblocks=16, n_feats=64) -- 1,369,859 params
- **Device**: cuda (NVIDIA GeForce RTX 3090), PyTorch 2.6.0+cu124
- **Validation set**: `data\DIV2K\DIV2K_valid_HR` -- 100 images, realistic degradation, LR patch 96x96
- **Calibration**: 8 batches (64 LR samples)
- **Histogram bins**: 2048
- **Quantization scheme**: symmetric per-tensor INT8 (activations) + symmetric per-channel INT8 (weights)

## Calibration scheme shootout

All four schemes share the **same calibration pass** -- the histogram is collected once, and each scheme just chooses a different summary of it (running max for `max-abs`, percentile cutoff for the other three).

| Scheme | PSNR (dB) | Drop vs FP32 | Latency (ms) |
|---|---:|---:|---:|
| max-abs | 27.362 | +0.077 | 16.42 +/- 2.09 |
| percentile-99.99 | 27.351 | +0.087 | 15.33 +/- 1.20 |
| percentile-99.9 | 26.937 | +0.502 | 15.41 +/- 1.52 |
| percentile-99.0 | 25.135 | +2.304 | 15.32 +/- 1.13 |

## Per-layer chosen `amax` per scheme

This is the value that drives `scale = amax / 127` for each layer's input quantizer. Smaller values = tighter clipping = the tail of the activation distribution gets saturated.

| Layer | max-abs | percentile-99.99 | percentile-99.9 | percentile-99.0 |
|---|---:|---:|---:|---:|
| `head` | 1.0000 | 1.0000 | 0.9999 | 0.9841 |
| `tail` | 1.9936 | 1.2206 | 1.0019 | 0.7235 |
| `body.16` | 2.8725 | 1.5118 | 1.0375 | 0.5454 |
| `body.0.conv1` | 1.4329 | 1.3773 | 1.1879 | 0.8766 |
| `body.0.conv2` | 1.1014 | 0.7466 | 0.6217 | 0.4691 |
| `body.1.conv1` | 1.2062 | 0.8678 | 0.7411 | 0.5065 |
| `body.1.conv2` | 1.1916 | 0.5944 | 0.4077 | 0.2772 |
| `body.2.conv1` | 1.1538 | 0.7399 | 0.5378 | 0.3517 |
| `body.2.conv2` | 1.1563 | 0.6550 | 0.3961 | 0.1822 |
| `body.3.conv1` | 1.2463 | 0.7182 | 0.5171 | 0.3197 |
| `body.3.conv2` | 1.2421 | 0.6246 | 0.3918 | 0.1820 |
| `body.4.conv1` | 1.2936 | 0.7378 | 0.5244 | 0.3190 |
| `body.4.conv2` | 1.1257 | 0.5206 | 0.3356 | 0.1612 |
| `body.5.conv1` | 1.3159 | 0.7674 | 0.5323 | 0.3139 |
| `body.5.conv2` | 1.2322 | 0.6629 | 0.3472 | 0.1716 |
| `body.6.conv1` | 1.4015 | 0.7876 | 0.5460 | 0.3192 |
| `body.6.conv2` | 1.2362 | 0.6583 | 0.3727 | 0.1654 |
| `body.7.conv1` | 1.4608 | 0.8014 | 0.5551 | 0.3251 |
| `body.7.conv2` | 1.3302 | 0.5890 | 0.3069 | 0.1594 |
| `body.8.conv1` | 1.5375 | 0.8212 | 0.5684 | 0.3305 |
| `body.8.conv2` | 1.4275 | 0.6650 | 0.3312 | 0.1745 |
| `body.9.conv1` | 1.6386 | 0.8794 | 0.6055 | 0.3496 |
| `body.9.conv2` | 1.2992 | 0.7006 | 0.4002 | 0.1791 |
| `body.10.conv1` | 1.7041 | 0.9350 | 0.6475 | 0.3726 |
| `body.10.conv2` | 1.5172 | 0.6457 | 0.3453 | 0.1699 |
| `body.11.conv1` | 1.8397 | 0.9839 | 0.6780 | 0.3841 |
| `body.11.conv2` | 1.5828 | 0.7695 | 0.3797 | 0.1830 |
| `body.12.conv1` | 1.9173 | 1.0457 | 0.7197 | 0.4035 |
| `body.12.conv2` | 1.7914 | 0.8717 | 0.4436 | 0.2095 |
| `body.13.conv1` | 2.2329 | 1.1530 | 0.7757 | 0.4264 |
| `body.13.conv2` | 2.0545 | 1.0572 | 0.4916 | 0.2243 |
| `body.14.conv1` | 2.3425 | 1.3110 | 0.8636 | 0.4630 |
| `body.14.conv2` | 2.0186 | 0.9351 | 0.5087 | 0.2369 |
| `body.15.conv1` | 2.6271 | 1.4320 | 0.9537 | 0.5013 |
| `body.15.conv2` | 2.2803 | 1.1156 | 0.5947 | 0.2588 |
| `upsampler.0` | 2.6470 | 1.5797 | 1.2382 | 0.8977 |

## How to read the histogram figure

`histograms.png` plots the activation-magnitude histogram (collected during calibration) for six representative layers. Vertical dashed lines mark where each scheme places its `amax`:

- **Red (max-abs)**: at the largest `|x|` ever seen. Most conservative (no clipping) but exposed to single outliers.
- **Orange (99.99)**: cut off the very last 0.01% of the tail.
- **Green (99.9)**: typical TensorRT-style aggressive choice.
- **Blue (99.0)**: aggressive clipping; saturates more than just outliers.

Y-axis is log-scale because activation distributions are heavily long-tailed -- a linear axis would render the tail invisible. The amax differences look small numerically, but the resulting INT8 scale (`amax / 127`) is what determines the bin resolution for the bulk of values, which is what shows up in the PSNR table above.

## Takeaway

On this checkpoint and val set, **`max-abs`** wins the shootout (27.362 dB). The spread across schemes is 2.227 dB.

Interpret carefully:

- A small spread (< 0.05 dB) means activations are NOT outlier-heavy for this model -- the calibration choice is mostly cosmetic.
- A large spread (> 0.2 dB) means there ARE outliers that max-abs is exposed to. In that case, percentile clipping is a real win.
- If `percentile-99.0` is best, the tail isn't useful; consider an even tighter cutoff or a learned clipping threshold (PAMS-style).
- If `max-abs` is best, the tail IS informative; percentile clipping is throwing away useful signal.

All four numbers are produced from the same FP32 weights and the same calibration histogram. The infrastructure is in place to add KL-div or MSE-optimal calibration as additional schemes by adding entries to `SCHEMES` in `calibration_ablation.py`.
