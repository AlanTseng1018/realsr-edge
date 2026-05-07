# ADR-004: 為何選擇純 L1 而非 Perceptual + Adversarial Loss

## Status
Accepted

## Context

訓練後評估,observe 兩個現象:
1. 材質感不足(sample 5 動物毛、sample 3 葉脈)
2. 邊緣銳化不足(sample 4 皮帶邊緣)

理論上可以加 perceptual / edge / adversarial loss 改善。

## Decision

**Keep pure L1 loss for now**.

## Reasoning

### Why not improve

1. **TV scenario alignment**:過度銳化 / 強烈材質會被觀眾感知為「**假**」
2. **Time budget**:perceptual loss 訓練成本 1-2 天,GAN 5+ 天
3. **Trade-off cost**:改善視覺 = 犧牲 PSNR 0.5-3 dB

### What we trade away

- 視覺豐富感
- benchmark 上的 perceptual scores (LPIPS)

### What we keep

- 純 fidelity 行為(不幻想細節)
- High PSNR/SSIM
- Edge deployment 友善(沒有額外 VGG 推論成本)
- Quantization 友善

## Future work

如果時間允許:
- Track D: EDSR + edge loss(0.5 天)
- Track E: EDSR + perceptual loss(1-2 天)
- 不在這次 scope
