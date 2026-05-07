# INT8 LPIPS Finding —— 從 「**Paradox**」 到 「**Perceptually Equivalent**」 的 interpretation 升級

> 這份筆記記錄 RealSR-Edge **Page 3 Card 2 「PSNR / LPIPS Triangulation」** 的最終 narrative,以及它從 「**puzzling paradox**」 升級成 「**perceptually equivalent within noise**」 的 reasoning 過程。
>
> 這個升級**避免 over-claim 「INT8 perceptually 比 FP32 好」**,同時把 finding 從 「警告型」 變 「驗證型」 —— actionability 大升。

---

## 1. 觀察:三 metric 的方向 split

INT8 PTQ vs FP32 baseline,**99/100 val images** 一致:

| Metric | 方向 | 變化 |
|---|---|---|
| PSNR ↑ better | 變差 | -0.080 dB |
| SSIM ↑ better | 變差 | -0.0044 |
| **LPIPS ↓ better** | **「變好」** | **distance 縮短 0.0153** |

(直覺反應:**「**INT8 PSNR 變差但 LPIPS 反而更接近 GT?**」**)

## 2. 機制(主流假設,理論 anchor: Blau & Michaeli, CVPR 2018)

### 機制鏈

```
L1 loss                                    [step 1]
  ↓ 數學上最小化解 = conditional median of HR | LR
  ↓ median 比 typical sample 平滑(median 數學特性,非 explicit 設計)
FP32 SR 過度平滑                          [step 2]
  ↓ 缺自然影像的 high-frequency texture
LPIPS feature space 中,FP32 SR 距離 GT 較遠   [step 3]
  ↓
INT8 quantization 加 broadband 雜訊        [step 4]
  ↓ 雜訊 spectral 性質接近自然 high-freq texture
INT8 SR output 在 feature space 推回 natural manifold  [step 5]
  ↓
LPIPS distance 縮短                        [observation]
```

### 用詞精度提醒

| 弱講法(會被 catch) | 強講法(senior) |
|---|---|
| 「L1 以平滑訊號為優化方向」 | **「L1 收斂到 conditional median,median 比 typical sample 平滑」** |
| 「INT8 LPIPS 變高 / 變好」 | **「INT8 LPIPS distance 縮短」** 或 **「LPIPS 從 0.21 降到 0.20」**(避開方向詞) |
| 「INT8 perceptually 比 FP32 好」 | **「INT8 在 LPIPS feature space 較接近 GT,但 magnitude 在感知 noise 內 → perceptually equivalent」**(避免 over-claim) |

---

## 3. 關鍵:**Magnitude 檢查 — 這個改變人眼看得出來嗎?**

### Reference 門檻

LPIPS 在 SR / image quality 圈的 「**人眼感知門檻**」 rule of thumb 約為 **0.05**:
- LPIPS Δ < 0.05 ≈ 「仔細看才看得出」
- LPIPS Δ > 0.10 ≈ 「一眼看出差別」
- LPIPS Δ > 0.30 ≈ 「明顯不同的圖」

(注意:**這個門檻不是嚴格定義**,不同 paper 浮動。但 0.05 是普遍 rule of thumb。)

### 你 RealSR-Edge 的實際數字

| Comparison | LPIPS Δ | vs 0.05 門檻 | 解讀 |
|---|---|---|---|
| FP32 vs INT8 PTQ | -0.0153 | **30% 門檻** | **below perceptibility** |
| FP32 vs INT8 QAT | -0.0208 | **41% 門檻** | **below perceptibility** |
| FP32 vs FP16 | ±0.0000 | 0% | 完全等價 |

→ **兩個 INT8 variant 的 LPIPS 改變都在 「**人眼勉強或看不出來**」 的範圍**——statistical signal 有(99/100 images 一致),但 practical magnitude 在 noise 內。

---

## 4. 升級後的 narrative —— 「**Triangulation 確認等價**」 而非 「**Paradox 揭露**」

### 之前 framing(divergence-led,弱 actionability)

> 「INT8 PSNR/SSIM 變差,LPIPS 反而變好(99/100 images)——perception-distortion paradox」

問題:
- Reader 可能 over-claim 解讀成 「**INT8 perceptually 變好了**」
- 對 deploy 決策不直接(「**所以 INT8 該不該 ship?**」 沒答案)

### 升級 framing(magnitude-aware,強 actionability)

> 「INT8 PSNR drop 0.08 dB(微小),**LPIPS distance 縮短 0.015,遠低於人眼感知門檻 0.05**—**multi-metric stack 的結論是 INT8 perceptual quality 跟 FP32 在感知 noise 內等價;PSNR-only 框架會 false-alarm,multi-metric triangulation 防止這個誤判**」

✅ 這版本:
- **不誇大**(沒說 「INT8 比 FP32 更好」)
- **直接 drive deploy 決策**(「INT8 perceptually 安全 → 可 ship」)
- **多 metric stack 的價值更明確**(triangulation 確認 「等價」,非 single metric 誤判)

---

## 5. 三個 Caveat(senior 細節,被深問用)

### Caveat 1:**0.05 門檻不是嚴格定義**

LPIPS 沒官方 「**Just-Noticeable Difference (JND)**」 threshold。0.05 是 SR / image quality 圈口耳相傳,**不同 paper 數字會浮動 0.03-0.08**。

→ 嚴格說 「below perceptibility」 要 hedge:**「**below typical perceptibility threshold ~0.05**」**(加 「typical」)

### Caveat 2:**Aggregate 看 OK,worst-case 可能不一樣**

99/100 images aggregate -0.015,**但 worst-case image 可能 -0.05 以上**(從 [distribution.png](../results/quantization/200ep_with_report/lpips_heatmaps/distribution.png) histogram 可見)。

→ 嚴格 framing:**「**aggregate magnitude below threshold;per-image worst-case may exceed**」**

### Caveat 3:**LPIPS 不等於人眼**

LPIPS 是 「**用 CNN feature 模擬人眼**」,**不是真實 2AFC test**(human panel review)。

→ 嚴格 production 場景**還是該做 panel review**,不能 100% 信 LPIPS。

---

## 6. 完整 ready answer(60 秒版,可背)

被問 「**為什麼 INT8 LPIPS 變好?**」:

> 「**機制**:L1 loss 收斂到 conditional median,median 比 typical sample 平滑——所以 FP32 SR 缺自然影像的 high-frequency texture。INT8 quantization 引入 broadband 雜訊,雜訊的 spectral 性質接近自然 high-freq texture,**把 over-smooth output 在 LPIPS feature space 推回 natural manifold**,LPIPS distance 縮短。
>
> **但 magnitude reality check 很關鍵**:LPIPS 改善只有 0.015,**遠低於人眼感知門檻 0.05**——所以 「**INT8 perceptually 比 FP32 好**」 是 over-claim,**正確解讀是 「INT8 perceptually 跟 FP32 在感知 noise 內等價」**。
>
> **對 vendor 的 deploy 決策**:**multi-metric triangulation 確認 INT8 perceptual quality 等價 FP32**;**PSNR-only 框架看到 0.08 dB drop 會 false-alarm,multi-metric 防止這個誤判**。」

---

## 7. 對 Page 3 Card 2 的具體 implication

### 之前 「PSNR/LPIPS Divergence」 的弱點

- **不直接 drive deploy 決策**(critique 對)
- 比 Page 3 其他 card actionable 等級弱(QDQ paradox / Roofline / Cross-language 都有 deploy-direct 結論)

### 升級後 「PSNR/LPIPS Triangulation」 的強點

- **Drive deploy 決策**:「INT8 perceptually 安全可 ship」
- **跟其他 card actionable 等級對齊**
- **避免 over-claim**(不講 「INT8 更好」,講 「等價」)
- **demo multi-metric stack 的具體價值**(triangulation prevents false alarm)

---

## 8. 跟其他 finding 的關係(narrative 整體位置)

| Finding | 性質 | Drive 什麼 |
|---|---|---|
| QDQ Paradox | 直接 deploy 決策 | 用 Native TRT path |
| **PSNR/LPIPS Triangulation** | **直接 deploy 決策(磁轉化後)** | **INT8 deploy 安全(perceptual 確認)** |
| HW Utilization | 直接 deploy 決策 | sm86 用 FP16,NPU 重測 |
| Cross-language Fidelity | 驗證型 | C++ deploy 不需 retest |

升級後 4 個 cards **actionable 等級對齊**——這是 Page 3 narrative 內聚的關鍵。

---

## 9. 一句話 takeaway

> **同一個現象(INT8 LPIPS 縮短),用 「**direction-only**」 視角是 paradox(puzzling),用 「**magnitude-aware**」 視角是 perceptually equivalent(actionable)**。
>
> **後者比前者**(a) 更誠實(不過度宣稱)(b) 更 actionable(deploy 直接 OK)(c) 更 demo multi-metric stack 的具體價值(prevents false alarm)。
>
> **這個 reframe 是 RealSR-Edge narrative 最大的一次 self-correction**,從 「我發現 paradox」 升級到 「我用 multi-metric 確認 INT8 perceptual deploy 安全」——senior signal 顯著增強。
