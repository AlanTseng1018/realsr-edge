# SR scale 倍率對應的解析度任務

> 這份筆記只回答一個問題：`scale_factor=2/3/4` 在實務上對應到什麼解析度任務?
> 重點是把抽象的「倍率」對應到 deployment 場景,避免訓練時用錯 patch 設定、demo 時講錯應用情境。

---

## TL;DR

- **scale=2**:消費級 TV upscaling 主戰場(**1080p → 4K UHD**)
- **scale=3**:罕見,通常是 720p → 4K 的 hack,實務上少用
- **scale=4**:下世代 / 低訊源 / paper benchmark 主戰場(540p → 4K 或 1080p → 8K)

EDSR 架構**只支援整數倍率**,因為 `PixelShuffle(scale)` 要求 scale 是整數 attribute,且 channel 數要被 `scale²` 整除。所以 720p → 1080p(1.5×)在這個架構下做不到。

---

## 解析度對應表

| scale | LR → HR | 解析度 (W×H) | 應用場景 |
|---|---|---|---|
| **2** | 1080p → **4K UHD** | 1920×1080 → 3840×2160 | TV upscaling(主戰場) |
| 2 | 540p → 1080p | 960×540 → 1920×1080 | 串流低碼率還原 |
| 2 | 720p → 1440p (QHD) | 1280×720 → 2560×1440 | 顯示器 upscaling |
| 3 | 720p → ~4K | 1280×720 → 3840×2160 | 廣播訊源到 4K 面板 |
| 4 | 540p → 4K | 960×540 → 3840×2160 | 串流到 4K |
| 4 | 1080p → 8K | 1920×1080 → 7680×4320 | 下世代 8K TV |

---

## 為什麼非整數倍率(例如 720p → 1080p)做不到?

`PixelShuffle(scale)` 是個純靜態 reshape op,要求:
1. `scale` 是 Python int,不是 tensor
2. 輸入 channel 數能被 `scale²` 整除

1.5 沒辦法滿足這兩個條件。實務上要做 720p → 1080p 有三條路:

| 方法 | 做法 | 缺點 |
|---|---|---|
| 先 resize 再 refine | bicubic 拉到 1080p,過 scale=1 的 SR refiner | resize 在某些 NPU fallback 到 CPU |
| 過 2× 再 downscale | 720p × 2 = 1440p,再縮回 1080p | 浪費算力 |
| Implicit neural rep | LIIF / Local Implicit 任意倍率 | 對 edge SoC 太重,export 不乾淨 |

整數倍率是 **edge deployment 友善 vs. 任意倍率彈性**的取捨。consumer SR pipeline 幾乎都選整數,因為 PixelShuffle / DepthToSpace 在所有 edge runtime 上都原生支援。

---

## 實務記法(面試 / demo 時可以這樣講)

> 「我這個 RealSR-Quant 用 ×2 是因為要對齊 TV 量產情境—— 訊源 1080p、面板 4K,倍率剛好整數。
> 學界 benchmark 慣用 ×4 是為了讓任務難度拉開、能看出模型差異;
> ×3 在產業上幾乎用不到,因為它是質數,upsampler 一次要膨脹 9 倍 channel,參數比 ×4(拆兩段)還貴。」

這段話的三個 anchor:
- **產業 ↔ ×2**(1080p→4K,消費級 TV)
- **學界 ↔ ×4**(Set5/Set14/B100/Urban100 PSNR benchmark)
- **×3 是孤兒**(質數,實務罕用,參數還貴)

---

## 訓練端配套(patch size 跟 scale 綁在一起)

LR patch 餵進去,HR patch = LR × scale。所以 scale 越大,固定 HR patch size 下 LR patch 越小、context 越少:

| scale | LR patch | HR patch |
|---|---|---|
| 2 | 96 | 192 |
| 3 | 64 | 192 |
| 4 | 48 | 192 |

切換 scale 時記得改 dataset 的 patch_size,不然 dataloader 會吐錯誤的對。

---

## 對量化 / deployment 的影響

scale 的選擇會影響 upsampler 那一層的 activation 動態範圍,進而影響 INT8 PTQ 難度:

| scale | upsampler conv channel 膨脹 | 量化難度 |
|---|---|---|
| 2 | 一段 ×4 | 最友善 |
| 3 | 一段 ×9(一次膨脹太多) | activation range 寬,常是 PTQ bottleneck |
| 4 | 兩段 ×4 | 中等(分散到兩段) |

如果 sensitivity 報告顯示 upsampler 那層 PSNR drop 嚴重,**×3 比 ×2/×4 更明顯**就是這個原因。

---

## ONNX export 是 scale-specific

`scale_factor` 在 `__init__` 階段 bake 進 graph,export 出來的 ONNX 是 scale-specific 的:
- `model_x2.onnx`、`model_x3.onnx`、`model_x4.onnx` 各一個
- runtime 不能切換倍率
- 對 TV SoC 部署是合理選擇——靜態 graph 換最大相容性

如果產品要支援多倍率,deploy 端就是多個 ONNX 並存,搭配上層調度決定要用哪個。
