# RealSR-Quant Pipeline Anatomy — pre-handoff validation 各站清單

> **這個專案的 deliverable 是 pre-handoff validation pipeline,不是 SR model**。SR (EDSR-baseline) 是 stress-test 載體;pipeline 的目的是在模型交給 NPU vendor 前,系統性把可預期的 deploy-time failure mode 用 detector 全部攔下,讓 vendor 接過去**不需要打回來重訓**。
>
> 中間量到的 findings(QDQ paradox / perception-distortion / Tensor Core 利用率 / DLL 衝突)**不是專案目的,是 pipeline 真的有抓到東西的證據**——它們間接證明這條 infrastructure 不是 over-engineering。

下面 9 個站 + 一張總表是這個 pipeline 的完整 anatomy。每站列:**Script / 做什麼 / 產出 / 攔到什麼 deploy-time risk**。

---

## A. 訓練側(站 0-2)

### 站 0 — 資料準備

| 元件 | 內容 |
|---|---|
| Script | [src/data/dataset.py](../src/data/dataset.py) (`SRDataset`) + [src/data/degradation.py](../src/data/degradation.py) (`RealisticDegradation`) |
| Source | DIV2K(train 800 / val 100 張 2K HR images) |
| LR 生成 | **Realistic degradation** pipeline:blur (50%) → 確定 downsample → banding (50%) → noise (50%) → JPEG (50%) |
| 採樣方式 | Random crop 192×192 HR / 96×96 LR(training);center crop with deterministic seed(val) |
| Patch 對 scale 對齊 | `hr_patch_size % scale == 0` 強制檢查 |

**確保的事**:val LR 在每個 epoch / 每次 evaluation 完全一致(deterministic seed),才能公平比較不同 format 的 PSNR。

---

### 站 1 — 模型訓練

| 元件 | 內容 |
|---|---|
| 架構 | [src/models/edsr.py](../src/models/edsr.py) — EDSR-baseline(16 ResBlocks × 64 feats × scale 2)~1.37M params |
| 子模組 | [src/models/common.py](../src/models/common.py) — ResBlock(**no BN** 是 EDSR 的定義性選擇) |
| Loss | L1(EDSR-baseline 標準,實證上比 L2 給更高 PSNR) |
| Optimizer | Adam, lr=1e-4, scheduler step 100 / gamma 0.5 |
| Epochs / Batch | 200 / 16 |
| Loss 在哪算 | LR 96×96 in → SR 192×192 out vs HR 192×192 patch |
| Script | [src/training/train.py](../src/training/train.py) |
| Checkpoint | `best.pt`(每次 val PSNR 創新高存)、`final.pt`、metadata 記錄完整 args(scale / patch / batch / degradation 等) |

**產出**:`results/runs/[timestamp]/checkpoints/best.pt` + `metrics.csv` + `curves.png` + `val_samples/`

---

### 站 2 — 訓練期 validation(loop 內)

| 元件 | 內容 |
|---|---|
| 驗證頻率 | 每 5 epoch 跑一次 val |
| Metrics | PSNR + SSIM(skimage),over 100 val patches |
| 視覺 | 每次 val 存 5 張 SR sample 進 `val_samples/` |
| 收斂 | 200 epoch 收到 val PSNR ≈ 27.44 dB(realistic degradation track) |
| 產出 | `metrics.csv`(epoch / train_loss / val_psnr / val_ssim)+ `curves.png` |

**這站只負責**確認「模型訓練本身沒爛掉」——**不是 deploy validation**。

---

## B. PyTorch 端 pre-handoff 驗證(站 3a–3f)

下面 6 站都在 PyTorch domain,**ONNX export 之前**——產出的是「**這個 model 量化後會怎樣**」的 fidelity / sensitivity / perceptual reference。所有結論**跨 hardware portable**(NPU 端等價)。

### 站 3a — Format shootout(精度比較)

| 元件 | 內容 |
|---|---|
| Script | [src/quantization/analyze.py](../src/quantization/analyze.py) `run_shootout()` |
| 對象 | FP32 / FP16 (autocast) / BF16 (autocast) / INT8 (fake-quant) |
| Metrics | PSNR + SSIM + **LPIPS**(三 metric triangulation) |
| INT8 實作 | [src/quantization/fake_quant.py](../src/quantization/fake_quant.py) `CalibratingConv2d`,per-tensor activation symmetric / per-channel weight symmetric |
| Calibration | 8 batches × 8 LR samples = 64 calibration samples,max-abs |
| 產出 | `results/quantization/200ep_with_report/shootout.csv` / `shootout.md` |

**攔到的 risk**:量化導致 fidelity 不可接受時提早警告;**且**三 metric 不同調揭露 perceptual gap(PSNR 看不到 banding)。

### 站 3b — Per-layer sensitivity sweep(逐層敏感度)

| 元件 | 內容 |
|---|---|
| Script | [src/quantization/analyze.py](../src/quantization/analyze.py) `run_sensitivity()` |
| 方法 | 每次只把**一個** Conv2d 量化 INT8、其他保 FP32,測 PSNR drop |
| 範圍 | 36 個 Conv 全掃 |
| 產出 | `sensitivity.csv` / `sensitivity.md` |
| 關鍵發現 | top-3 = `tail` / `upsampler.0` / `head`,佔 PSNR drop 的 **82%** |

**攔到的 risk**:給 NPU vendor 一份「**這幾層必須保 FP16,其他 INT8 安全**」的 mixed precision recipe——他們不需要自己重做 sensitivity 研究。

### 站 3c — Calibration method ablation(校準法選擇)

| 元件 | 內容 |
|---|---|
| Script | [src/quantization/calibration_ablation.py](../src/quantization/calibration_ablation.py) |
| 對比 | max-abs vs percentile 99 / 99.9 / 99.99 |
| 額外 | 收 amax histogram,visualize 每層 activation 分佈 |
| 產出 | `calibration_ablation.md` + `histograms.png` + `per_layer_amax.csv` |

**攔到的 risk**:NPU vendor SDK 通常給 calibration method 多個選項,這份 ablation 直接告訴他們**哪個對這個 model 影響顯著、哪個 noise level**——不需自己摸索。

### 站 3d — Per-layer weight statistics(權重統計)

| 元件 | 內容 |
|---|---|
| Script | [src/deployment/analyze_layers.py](../src/deployment/analyze_layers.py) |
| 從哪裡讀 | 直接從 `best.pt` 讀 weight tensor |
| 算什麼 | 每層 weight min / max / mean / std / percentile / dynamic range |
| 產出 | `results/layer_analysis/edsr_200ep/` |

**攔到的 risk**:NPU memory layout 規劃時,看哪些層 weight magnitude 落在哪個範圍——影響 INT8 scale 設定 + 哪些層 outlier 多。

### 站 3e — Perceptual deep dive(LPIPS spatial)

| 元件 | 內容 |
|---|---|
| Script | [src/quantization/lpips_heatmap.py](../src/quantization/lpips_heatmap.py) |
| 三層輸出 | (1) per-image LPIPS scalar,(2) distribution histogram across val set,(3) spatial heatmap on chosen image |
| LPIPS net | SqueezeNet(alex 下載 flaky 改用 squeeze;結論不變) |
| 比較對象 | INT8 vs FP32(隔離量化引入的 perceptual delta) |
| 產出 | `per_image_lpips.csv` + `distribution.png` + `heatmap_<image>.png` |

**攔到的 risk**:PSNR 看不到的 banding-style perceptual artifact——讓 vendor 不會 ship 出 PSNR 過關但 customer 看著難看的版本。

### 站 3f — Mixed precision sweep(混合精度配方)

| 元件 | 內容 |
|---|---|
| Script | [src/deployment/mixed_precision.py](../src/deployment/mixed_precision.py) |
| 方法 | 根據 sensitivity 排名,top-N 層保 FP32、其他 INT8,sweep N 看 quality vs size trade-off |
| 產出 | `results/mixed_precision/edsr_200ep/` — `mixed_precision_sweep.csv` / `.json` / `.png` |

**攔到的 risk**:給 NPU vendor 一份「**保哪幾層 PSNR 還可接受,size 縮多少**」的對照表,不需自己 search。

---

## C. ONNX / TRT 端 pre-handoff 驗證(站 4-7)

下面 4 站從 ONNX export 開始,進入 backend-specific 的世界。產出針對「**ONNX 進到實際 runtime 後會怎樣**」的 reference。

### 站 4 — ONNX export + 跨格式驗證

| 元件 | 內容 |
|---|---|
| Script | [src/deployment/export_pipeline.py](../src/deployment/export_pipeline.py) |
| 產出三精度 | FP32 / FP16 / INT8 (QDQ format) |
| INT8 設定 | `quantize_static`,**ActivationSymmetric=True / WeightSymmetric=True / quant_pre_process** |
| Verification | PyTorch model vs ONNX 同樣 input 比 numerical drift |
| 產出 | `results/onnx_exports/edsr_200ep/` — 三個 .onnx + `verification.md` + `metadata.json` + `README.md` |

**攔到的 risk**:**PyTorch → ONNX 的 numerical drift**(常見:某 op behavior 差異)——若 verify 不過,在 export 階段 catch,而不是 vendor SDK 端 catch。

### 站 5 — Cross-backend latency benchmark(ORT × 3 EP)

| 元件 | 內容 |
|---|---|
| Script | [src/deployment/benchmark_onnx.py](../src/deployment/benchmark_onnx.py) |
| 矩陣 | 3 ONNX(FP32/FP16/INT8) × 3 EP(CPU/CUDA/TensorRT) = 9 點 |
| 量什麼 | latency mean ± std + PSNR + active_provider(實際生效的 EP) |
| Bench shape | `1×3×96×96`(matched 訓練 patch) |
| 產出 | `results/onnx_benchmark/edsr_200ep_full/benchmark.csv` + `benchmark.md` |
| 聚合 | [src/deployment/deploy_summary.py](../src/deployment/deploy_summary.py) → `deploy_summary.md` |

**攔到的 risk**:**Backend-specific INT8 失效**——this is where the QDQ paradox surfaced。Active provider 欄位記錄真實 fall back 行為,讓你看到 ORT 是否成功用 TRT EP。

### 站 6 — Native TRT engine + INT8 calibrator(繞過 QDQ)

| 元件 | 內容 |
|---|---|
| Script | [src/deployment/benchmark_trt.py](../src/deployment/benchmark_trt.py) |
| 三精度 | FP32(builder default) / FP16(BuilderFlag.FP16) / **INT8 from FP32 ONNX + IInt8EntropyCalibrator2** |
| Calibrator | `ValSetCalibrator`,64 val LR patches |
| 為什麼不用 INT8 QDQ ONNX | TRT 10 ONNX parser **拒絕** ORT 產的 INT32 bias DequantizeLinear |
| 產出 | `results/trt_benchmark/edsr_200ep/benchmark.csv` + `benchmark.md` + `engines/` 三個 .engine + `int8_calib.cache` |

**攔到的 risk**:**ORT layer-specific 失效跟 INT8 fundamental 失效要分清楚**——這站證明 INT8 在 native TRT 上 work(1.93 ms),前一站的 4.33 ms 是 ORT layer 的 bug。

### 站 7 — Roofline + kernel profile(硬體利用率)

| 元件 | 內容 |
|---|---|
| Script | [src/deployment/profile_trt.py](../src/deployment/profile_trt.py) |
| 計算 | 從 checkpoint 算 FLOPs(torchinfo)+ 估 memory traffic → 算 arithmetic intensity |
| Profile | `torch.profiler` CUDA activity → kernel-level latency breakdown |
| Roofline | 三精度 ceiling(GPU spec auto-detect by device name)+ 三個 measured 點 |
| 產出 | `results/trt_profile/edsr_200ep/` — `roofline.png` + `profile_report.md` + `metadata.json` |
| 關鍵發現 | 三精度全 compute-bound;INT8 利用率 **64%**(FP16 84%) |

**攔到的 risk**:**「INT8 為什麼不快過 FP16」的 root cause**——讓 NPU vendor team 看了知道是 GPU sm86 scheduling 問題,**不是 INT8 path 本身有問題**。

---

## D. Cross-language + Handoff bundle(站 8-9)

### 站 8 — Cross-language inference(C++ 端 fidelity)

| 元件 | 內容 |
|---|---|
| Script | [cpp_inference/edsr_runner.cpp](../cpp_inference/edsr_runner.cpp) |
| Build | [build.bat](../cpp_inference/build.bat) — VS 2022 BuildTools cl.exe + ORT GPU prebuilt + stb single-header IO |
| 三 EP | CPU / CUDA / TensorRT |
| 雙模式 | `--crop-hr 192`(matches Python 96 LR)和全圖 |
| Verify | 同樣 input 跨 EP / 跨語言比 PSNR(全部 29.45 dB ±0.004) |
| 產出 | `build/edsr_runner.exe` + `sr_cpu.png` / `sr_cuda.png` / `sr_trt.png` + [README.md](../cpp_inference/README.md) |

**攔到的 risk**:**C++ 端 numerical drift / DLL 衝突 / op behavior 差異**——vendor SDK 是 C++,Python 端通過不代表 C++ 端通過。Cross-language PSNR 對齊是 deploy 必過的關卡。

順帶**catch 到的 deployment 真實坑**:Windows `System32\onnxruntime.dll` 1.17.1(Windows ML 內建)會在 DLL search 時 override prebuilt 1.25.0。Build script 要 copy DLL 到 exe 同層覆蓋。

### 站 9 — 聚合 / 視覺化 / 報告(handoff bundle)

| 元件 | 內容 |
|---|---|
| [deploy_summary.py](../src/deployment/deploy_summary.py) | 把 ORT benchmark 聚合成 deployment markdown |
| [visualize_results.py](../src/deployment/visualize_results.py) | Single-PNG summary |
| [mindmap_validation.py](../src/deployment/mindmap_validation.py) | Validation mindmap PNG |
| [run_pipeline.py](../src/deployment/run_pipeline.py) | 一鍵跑整條 pipeline |
| 報告主軸 | `results/onnx_benchmark/edsr_200ep_full/deploy_summary.md` — 含 verified / hypothesized / cannot verify 三層 appendix |

**這站的 deliverable** 就是給 NPU vendor team 看的 handoff bundle:**model checkpoint + ONNX × 3 + .engine × 3 + sensitivity recipe + calibration ablation + perceptual eval + cross-backend reference + roofline 分析 + C++ 驗證**。

---

## 全 pipeline 一張總表

| 站 | 階段 | Script | 攔到什麼 deploy-time risk |
|---|---|---|---|
| 0 | 資料準備 | `dataset.py` + `degradation.py` | 確保 val deterministic,跨 format 比較公平 |
| 1 | 模型訓練 | `train.py` | 收斂 baseline + 完整 args metadata |
| 2 | Training-time val | (loop 內) | 模型本身沒爛掉 |
| **3a** | Format shootout(PSNR/SSIM/LPIPS) | `analyze.py` | 量化 fidelity 失效;三 metric 不同調揭露 perceptual gap |
| **3b** | Per-layer sensitivity | `analyze.py` | 哪幾層必須保 FP32;mixed precision recipe |
| **3c** | Calibration ablation | `calibration_ablation.py` | calibration method 選擇影響;outlier 處理 |
| **3d** | Weight statistics | `analyze_layers.py` | NPU memory layout 規劃輸入 |
| **3e** | LPIPS perceptual | `lpips_heatmap.py` | PSNR 看不到的 banding-style artifact |
| **3f** | Mixed precision sweep | `mixed_precision.py` | quality / size trade-off table |
| 4 | ONNX export + verify | `export_pipeline.py` | PyTorch → ONNX numerical drift |
| 5 | ORT cross-EP | `benchmark_onnx.py` | Backend-specific INT8 失效(QDQ paradox) |
| 6 | Native TRT + calibrator | `benchmark_trt.py` | ORT-layer bug 跟 INT8 fundamental 區分 |
| 7 | Roofline + profile | `profile_trt.py` | HW utilization gap;compute vs memory bound |
| 8 | C++ cross-language | `edsr_runner.cpp` | C++ numerical drift / DLL 衝突 / op 差異 |
| 9 | 聚合 / handoff bundle | `deploy_summary.py` + `visualize_results.py` | 給 vendor team 完整 handoff package |

---

## 一句話 takeaway

> **9 個站、~12 個 detector tool、~10 種 deploy-time failure mode 的 coverage**。每個 detector 對應一個「**vendor 拿到 model 可能踩、但被我提前攔下**」的具體 risk class。**這就是 thesis 講的 pre-handoff validation pipeline 的具體 anatomy**——不是 abstract framework,是 12 個實際可跑的 script 構成的 pipeline。

---

## 對應到 narrative 三層

回到專案 thesis 的三層結構:

| Thesis 層 | 對應到本文檔的哪幾站 |
|---|---|
| **WHAT**(deliverable = pre-handoff validation pipeline) | 站 3a–3f + 站 4-7 + 站 8-9 |
| **WHY**(minimize NPU vendor retrain probability) | 每站「攔到什麼 risk」欄位的累積效果 |
| **HOW PROVEN**(findings as evidence) | QDQ paradox(站 5)/ perception-distortion(站 3e)/ Tensor Core 64% utilization(站 7)/ DLL 衝突(站 8) |

面試時對方要你 navigate 哪一站,**這份文檔配 [project_thesis_realsredge memory](../../.claude/projects/c--Users-start-RealSR-Quant/memory/project_thesis_realsredge.md) 一起翻,1 秒就能定位**。
