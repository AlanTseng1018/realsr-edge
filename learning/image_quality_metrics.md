# 影像品質指標：PSNR 與 SSIM

## PSNR（Peak Signal-to-Noise Ratio）

### 公式

```
PSNR = 10 · log₁₀( MAX² / MSE )     單位：dB

MSE  = (1 / N) · Σ (A_i - B_i)²
```

- `MAX`：像素最大值（uint8 = 255；float [0,1] = 1.0）
- `N`：像素總數（H × W × C）
- `A`：參考圖（原始／GT）；`B`：重建圖

展開後常見的等價寫法：

```
PSNR = 20 · log₁₀(255) - 10 · log₁₀(MSE)
     ≈ 48.13 - 10 · log₁₀(MSE)        ← 以 uint8 為例
```

### 圖片情境

- 對整張圖所有像素（H × W × C）一次計算 MSE，再代入公式
- RGB 三通道通常一起算（Y channel 版本見下方）
- 學術 SR benchmark（DIV2K、Set5、Set14）標準做法：轉到 **YCbCr，只算 Y channel**

```python
# [0, 1] float，Y channel only（學術標準）
mse  = ((y_pred - y_gt) ** 2).mean()
psnr = 10 * math.log10(1.0 / mse)
```

### 影片情境

**做法 A — 逐幀 PSNR 再平均（最常見）**

```
PSNR_video = (1/T) · Σ PSNR(frame_t)
```

優點：可繪製每幀曲線，找出劣化集中在哪些場景  
缺點：對異常幀敏感（一幀極差會拉低整體）

**做法 B — 合併 MSE 再算（等同把所有幀串成一張大圖）**

```
MSE_video  = (1/T) · Σ MSE(frame_t)
PSNR_video = 10 · log₁₀( MAX² / MSE_video )
```

優點：對幀間一致性較穩定  
缺點：異常幀影響被稀釋，不易察覺局部問題

> **業界慣例**：做法 A 為主，回報時需說明是 per-frame mean 還是 global MSE。

### PSNR 的限制

- 本質是 MSE 的對數變換，對人眼感知非線性
- 結構性失真（模糊、振鈴）與隨機噪點的 MSE 相同，但感知差異很大
- 同樣 PSNR 的圖，人眼評分可能差異懸殊

---

## SSIM（Structural Similarity Index）

### 公式

SSIM 分三個分量：**亮度 (l)**、**對比 (c)**、**結構 (s)**

```
l(x,y) = (2·μ_x·μ_y + C1) / (μ_x² + μ_y² + C1)
c(x,y) = (2·σ_x·σ_y + C2) / (σ_x² + σ_y² + C2)
s(x,y) = (σ_xy + C3)       / (σ_x·σ_y   + C3)

SSIM(x,y) = l · c · s
```

合併後常用的單一公式（C3 = C2/2）：

```
SSIM(x,y) = (2·μ_x·μ_y + C1)(2·σ_xy + C2)
            ─────────────────────────────────
            (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)
```

- `μ`：局部均值（11×11 Gaussian window）
- `σ²`：局部方差
- `σ_xy`：局部協方差
- `C1 = (0.01·MAX)²`，`C2 = (0.03·MAX)²`（防止分母為零）

### 圖片情境

- 在每個 **11×11 sliding window** 上算 SSIM，再對全圖平均 → **MSSIM（Mean SSIM）**
- 值域 [-1, 1]，完全一致 = 1
- 學術 SR benchmark 同樣在 Y channel 計算

```python
from skimage.metrics import structural_similarity as ssim

score = ssim(img_gt, img_pred,
             data_range=1.0,        # float [0,1]
             channel_axis=-1,       # RGB → 3 channel
             win_size=11,
             gaussian_weights=True)
```

### 影片情境

**做法 A — 逐幀 SSIM 再平均**

```
SSIM_video = (1/T) · Σ SSIM(frame_t)
```

與 PSNR 做法 A 對應，最常回報。

**做法 B — T-SSIM / Video-SSIM（加入時間維度）**

在時間軸上額外計算幀間一致性，懲罰閃爍（flickering）：

```
T-SSIM 將相鄰幀的差作為第三個比較維度
```

適用情境：SR 影片、影片壓縮品質評估。單純圖片任務不需要。

### MS-SSIM（Multi-Scale SSIM）

對圖片做多次下採樣，在每個尺度各算 SSIM 再加權合併：

```
MS-SSIM = Π SSIM_scale_j ^ w_j
```

更接近人眼感知（人眼在不同觀看距離下判斷品質），是 SSIM 的強化版。
LPIPS 更進一步，用深度學習特徵取代手工特徵。

---

## 使用情境對照

| 情境 | 推薦指標 | 原因 |
|---|---|---|
| 學術 SR benchmark | PSNR + SSIM（Y channel）| 領域標準，方便比較 |
| 量化精度驗證（FP32 vs INT8）| PSNR | 計算快，數值差距小時高靈敏度 |
| 壓縮品質評估 | MS-SSIM | 人眼感知更準確 |
| 影片 SR / 超解析 | PSNR + T-SSIM | 需評估時間連貫性 |
| 感知品質研究 | LPIPS | 最接近人眼，但需深度模型 |
| 硬體驗收測試（edge 部署）| PSNR | 快速、可程式化、有明確閾值 |

---

## 本專案中的對應

| 腳本 | 使用指標 | 說明 |
|---|---|---|
| `export_pipeline.py` | PSNR | FP32 ONNX vs PyTorch 等價驗證 |
| `benchmark_onnx.py` | PSNR | 各精度（FP32/FP16/INT8）品質比較 |
| `benchmark_trt.py` | PSNR | Native TRT engine 品質驗證 |
| `analyze_layers.py` | PSNR | FP32 vs fake-INT8 逐層敏感度 |

SSIM 未納入目前 pipeline，原因：量化誤差分析場景中 FP32 vs INT8 的差距非常小
（PSNR > 48 dB），PSNR 在這個精度區間的解析度已足夠，SSIM 增加的資訊有限。
若要擴充為完整影像品質報告（對比 HR ground truth），加入 SSIM 是自然的下一步。
