# Perceptual metric for SR：LPIPS、它跟 PSNR/SSIM 為什麼會不同調

> 這份筆記記錄的不只是「LPIPS 是什麼」,更重要的是 **「在這個專案的 INT8 PTQ 數據上,PSNR / SSIM 跟 LPIPS 講了相反的故事」**——以及為什麼這個不同調反而比一致還更有訊息量。

---

## 為什麼 SR 量化研究需要 perceptual metric

PSNR 是 pixel-wise MSE 的對數形式,SSIM 是 11×11 window 的 local statistics。兩個都是**手工設計的公式**,跟人眼感知對齊度有限。

特別是在 SR 量化情境:

- **PSNR drop 0.1 dB** 看起來「在可接受範圍」,但完全沒告訴你 drop 是分散在紋理區還是集中在天空(平滑區)
- **SSIM drop 0.005** 也是 window 平均後的數字,類似的盲點
- **平滑區的 banding artifact** 對人眼非常明顯,但 error magnitude 很小,PSNR/SSIM 統計上幾乎看不到

LPIPS(Learned Perceptual Image Patch Similarity, Zhang et al., CVPR 2018)補上這個缺口:用預訓練 CNN 的中層 feature 算距離,因為「人眼看圖也是某種 feature extraction」,CNN feature 距離跟人眼判斷高度 correlated。

實作上幾乎免費:`pip install lpips`、輸入 `[-1, 1]` 範圍的 image pair,輸出一個距離 scalar(或 spatial map)。

---

## 我們在 RealSR-Edge 上的實際發現:三個 metric 不同調

200 epoch EDSR-baseline,INT8 PTQ vs FP32 baseline,DIV2K val 100 張(realistic degradation):

| Metric | FP32 → INT8 變化 | 說的是什麼 |
|---|---|---|
| PSNR | 27.439 → 27.359(**−0.080 dB**) | INT8 pixel error 變大 |
| SSIM | 0.7907 → 0.7863(**−0.0044**) | INT8 local structure 略劣 |
| **LPIPS** | 0.2108 → 0.1955(**−0.0154**) | **INT8 perceptual 反而更接近 GT** |

第三行的方向跟前兩行**相反**。在 99 / 100 val image 都成立(per-image distribution 全部在負值)。

直覺反轉的原因是 **perception-distortion tradeoff** (Blau & Michaeli, CVPR 2018)。

---

## 這個 tradeoff 的物理意義

### L1 loss 訓練的 model 容易 over-smooth

L1 loss 收斂到的是 conditional median;對自然影像而言,這個 median 比 GT 的「典型 sample」更平滑。所以 FP32 SR 跟 GT 比起來,**少了一些高頻紋理**。

PSNR 量到的 FP32 vs GT 距離,主要來自這個「過度平滑」的 systematic gap,而不是隨機 noise。

### INT8 quantization noise 是 broadband 的

每個 layer 的 quantization 都引入 small,寬頻譜的擾動。這些擾動加總起來,在 SR output 上看起來像低能量的「紋理 noise」。

### 在 feature space,broadband noise 比 over-smooth 更像「自然影像」

Pretrained CNN(SqueezeNet / AlexNet / VGG)在 ImageNet 上學到的中層 feature,對「自然紋理 statistics」是敏感的。一張**有點 noise 的圖** 跟 **過度平滑的圖** 比,前者的 feature 分佈更接近自然影像 manifold。

所以 INT8 SR(noisier)在 LPIPS feature space 上**距離 GT(natural)更近**,即使 pixel-wise error 更大。

### 直覺類比

> 想像給一張 8K 風景照,FP32 模型輸出像「過度去噪的數位修圖」(乾淨但生硬),INT8 模型輸出像「保留底片顆粒的版本」(有點 noise 但更像真實照片)。 PSNR 會偏好前者,但人眼/CNN feature 偏好後者。

---

## 重要的 nuance:全圖統計 vs 空間局部

LPIPS aggregate 不漲,但**不代表「INT8 沒引入任何 perceptual artifact」**。空間 LPIPS map(INT8 SR vs FP32 SR)講出另一個故事:

- 在 [heatmap_0879.png](../results/quantization/200ep_with_report/lpips_heatmaps/heatmap_0879.png) 的大教堂圖上
- **天空(平滑區)亮紅** — 這正是「banding hypothesis」會預期看到的位置
- **大教堂的雕飾與穹頂(紋理區)是冷色** — quantization noise 在這裡看不見

也就是說,quantization 確實**把 perceptual error 從「全圖均勻分佈」重分配到「平滑區集中」**,只是這個重分配沒有大到讓 aggregate LPIPS 反而上升。

對 deployment 的實際影響:

| 場景 | INT8 是否安全? |
|---|---|
| 紋理為主的內容(草地、人群、城市街景) | ✅ 安全,perception-distortion tradeoff 反而 favor INT8 |
| **平滑為主的內容(天空、漸層、單色背景)** | ⚠️ aggregate metric 說 OK,但局部仍可能有 banding,需要 **output dithering** 或 **mixed precision tail** |

---

## 對「SR 量化」研究方法論的 takeaway

1. **PSNR drop ≠ 感知傷害**。L1-trained 模型的 PSNR 進步常常是「更平滑」,而不是「更接近自然影像」
2. **單一 metric 騙人,三 metric 互相 cross-check 才安全**
   - PSNR (pixel) + SSIM (structure) + LPIPS (perceptual) 同方向 → 結論信度高
   - 三者**不同方向** → 訊號比一致時還多,值得深究
3. **Aggregate LPIPS + spatial LPIPS map 要一起看**。前者告訴你「整體是否變糟」,後者告訴你「變化集中在哪些區域」
4. **Perception-distortion tradeoff 不是 bug,是 feature**。如果你的 SR 應用追求 perceptual quality(消費級 TV),INT8 quantization 反而可能是 free lunch

---

## 怎麼跟面試官講

「我量了 INT8 的 LPIPS,**結果比 PSNR 預期的更複雜**:PSNR 跟 SSIM 都顯示 INT8 變差(分別 −0.08 dB、−0.0044),但 LPIPS 反而**下降** −0.0154,也就是 INT8 在 feature space 比 FP32 更接近 GT。99/100 val image 都成立。

這是 perception-distortion tradeoff:L1-trained 模型 over-smooth,INT8 quantization 加的 broadband noise 反而推 output 回到自然影像 distribution。

但這不代表 INT8 全面更好——空間 LPIPS heatmap 顯示 quantization 引發的 perceptual delta **集中在平滑區**(天空、漸層),所以 sky-heavy 內容仍然需要 dithering 或 mixed precision 保護。aggregate metric 看不到這個 redistribution。

對 TV deployment 而言,這個發現的價值是:**PSNR-based pessimism 對 INT8 可能過度悲觀**——對 perceptual quality 為主的應用,INT8 可能是 free lunch。」

這段話的三個 anchor:
- **三 metric 不同調的硬數據**(99/100 image 收斂)
- **能引用學術理論**(Blau & Michaeli 2018)
- **能延伸到 deployment 決策**(平滑區 vs 紋理區、是否需要 dithering)

---

## 附:LPIPS 如何接到 evaluation pipeline

在 [src/quantization/analyze.py](../src/quantization/analyze.py) 的 `evaluate_metrics()` 加上 optional `lpips_model` 參數,輸入 `(SR, GT)` pair 都先 rescale 到 `[-1, 1]` 再餵進去:

```python
if lpips_model is not None:
    d = lpips_model(sr * 2.0 - 1.0, hr * 2.0 - 1.0)
    lpips_sum += d.flatten().sum().item()
```

Spatial 版本則用 `lpips.LPIPS(net='squeeze', spatial=True)`,輸出 `(B, 1, H', W')` 的距離 map,upscale 後跟 SR output overlay 視覺化(見 [src/quantization/lpips_heatmap.py](../src/quantization/lpips_heatmap.py))。

Backbone 選擇:
- **SqueezeNet**(我們的 default):~5MB、最快、weights 下載穩定
- **AlexNet**(SR 文獻 standard):稍慢但更廣泛被引用,torchvision weights 下載偶爾 hash 失敗
- **VGG**:最準但 ~500MB、最慢,只在 paper 比賽用

對 production 評估而言,squeeze 已經足夠 — 跟 alex / vgg 的 ranking 一致,絕對值略不同但相對 ordering 不變。
