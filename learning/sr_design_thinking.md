# SR 模型設計思路：從問題本質到 EDSR

> 這份筆記記錄的不是「EDSR 長什麼樣」，而是「為什麼它要長這樣」。
> 理解設計動機，才能在遇到新問題時知道從哪裡下手。

---

## 核心問題定義

Super-Resolution 要解決的問題：

> **給一張模糊、低解析度的圖（LR），還原成清晰的高解析度圖（HR）。**

這個定義裡藏著兩個關鍵字：**還原**（不是創造）、**清晰**（pixel-level fidelity）。

這兩個字決定了後面所有設計選擇的方向。

---

## 思路一：SR 是 fidelity 任務，不是分類任務

### 分類任務 vs. SR 任務的本質差異

| | 分類（ImageNet） | SR |
|---|---|---|
| 目標 | 輸出一個類別標籤 | 輸出每個像素的正確值 |
| 損失函數 | Cross-entropy | L1 / L2（pixel-level） |
| 在乎絕對數值嗎？ | 不在乎 | **非常在乎** |
| 特徵需要 scale 一致嗎？ | 是（有助於泛化） | **否（破壞 fidelity）** |

### BatchNorm 為什麼對 SR 有害

BatchNorm 的設計目的：讓每層 activation 的分布在 batch 之間保持穩定，消除 internal covariate shift。

對分類任務很好，但對 SR 是毒藥：

```
BN 在 batch 內計算 mean/var → normalize → 強制所有圖的 feature 到同一個 scale
                                              ↓
                               但每張圖的亮度、對比度本來就不同
                               把 feature scale 拉平 = 把圖的「個性」抹掉
                                              ↓
                                  pixel-level fidelity 下降 → PSNR 掉
```

**SRResNet（2016）踩到這個坑，EDSR（2017）把 BN 拿掉，PSNR 就上去了。**

這是 EDSR 的核心貢獻，也是最重要的 insight。

```python
# ResBlock 裡沒有 BN，只有 Conv → ReLU → Conv
def forward(self, x):
    residual = self.conv2(self.relu(self.conv1(x)))
    return x + residual
```

---

## 思路二：沒有 BN，那訓練穩定性怎麼辦？

拿掉 BN 之後，深網路的訓練容易梯度爆炸。EDSR 用兩個機制解決：

### 機制 A：Residual Scaling（res_scale）

讓每個 ResBlock 的輸出乘一個小係數，避免信號在深層累積過大：

```python
residual = self.conv2(self.relu(self.conv1(x)))
residual = residual * self.res_scale   # 深層版用 0.1
return x + residual
```

效果：就算有 32/64 層 ResBlock，梯度也不會爆炸。

### 機制 B：Global Residual（Long Skip Connection）

```
Input
  │
head ──────────────────────────────┐
  │                                │ long skip
body (ResBlock × 16 + Conv)        │
  │                                │
  └──────── x + res ───────────────┘
                │
           upsampler
                │
             tail
```

**讓 body 只學 LR 和 HR 之間的「差異（residual）」，不需要從頭學整個映射。**

為什麼這樣更好？LR 和 HR 其實非常相似（結構、色彩都差不多），residual 接近 0。
學一個接近 0 的函數比學一個複雜的映射容易很多。

```python
def forward(self, x):
    x = self.head(x)
    res = self.body(x)
    x = x + res        # global residual：body 只學 delta
    x = self.upsampler(x)
    x = self.tail(x)
    return x
```

---

## 思路三：upsampling 要放在哪裡？

### 早期（錯誤）的做法

```
LR (64×64) → bicubic 放大 → (128×128) → CNN → HR (128×128)
```

問題：CNN 在 128×128 上做所有計算，但 bicubic 放大的那部分根本不需要學，算力浪費。

### EDSR 的做法：在 feature space 做所有學習，最後才 upsample

```
LR (64×64) → CNN 在小尺寸做所有特徵萃取 → upsampler → HR (128×128)
```

**先把所有複雜的學習放在低解析度 feature map 上做（便宜），最後才擴大（貴）。**

同樣的模型深度，計算量少很多。

### 為什麼選 PixelShuffle 而不是 Transposed Conv？

| | Transposed Conv | PixelShuffle |
|---|---|---|
| Checkerboard artifact | 容易出現 | 不會 |
| 可學習參數 | 有（upsampling kernel） | 前面的 Conv 負責 |
| ONNX 支援 | 一般 | **全平台支援（DepthToSpace）** |
| 直覺理解 | 反向卷積 | 把 channel 重新排列成空間 |

```python
# scale=2 的 PixelShuffle upsampler
Conv(64 → 256)    # 把 channel 擴大 scale² 倍
PixelShuffle(2)   # [B, 256, H, W] → [B, 64, 2H, 2W]，純重排，不失真
```

---

## 整條設計思路線

```
「SR 是 fidelity 任務，不是分類任務」
        ↓
BN 破壞 pixel fidelity → 拿掉 BN
        ↓
沒有 BN → 用 residual scaling + global residual 穩定訓練
        ↓
「在低解析度做計算更有效率」
        ↓
CNN 在 LR feature space 完成所有學習 → 最後才 upsample
        ↓
PixelShuffle 是最乾淨、最相容的 upsample 方式
        ↓
EDSR
```

---

## 一個重要的 meta 觀察

**好的架構設計往往是「減法」，不是「加法」。**

EDSR 的核心進步是把 SRResNet 裡有害的東西（BN）拿掉，而不是加了什麼新東西。
得到的模型更簡單、更快、PSNR 更高、量化更友善。

這個思路在深度學習研究裡很常見：
- EDSR：去掉 BN
- MobileNet：去掉普通 conv，換成 depthwise separable
- ViT：去掉 CNN 的所有 inductive bias

> 每次你想加一個新東西之前，先問：「有沒有什麼可以拿掉？」

---

## 對應到這個專案的實際驗證

| 設計選擇 | 在這個專案的體現 |
|---|---|
| 無 BN | 量化分析：body 32 層幾乎 0 drop（weight distribution 平滑）|
| Global residual | 訓練曲線：收斂穩定，無震盪 |
| LR-space 學習 + PixelShuffle | ONNX export 全相容，無需特殊 op |
| 純 L1 loss（無 perceptual） | fidelity 優先，PSNR 27.44 dB，量化友善 |

---

## 延伸閱讀

- **EDSR 原始論文**：Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution", CVPRW 2017
- **SRResNet（EDSR 的前身）**：Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a GAN", CVPR 2017
- **PixelShuffle 原始論文**：Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN", CVPR 2016
- **為什麼 BN 在 SR 有害**：EDSR 論文 §4.1 有詳細的 ablation study
