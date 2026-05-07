# Senior Deliverable Framing —— 「Decision Support」 vs 「Method Demo」 vs 「Coverage Report」

> 這份筆記回答一個面試會被問、但很少 candidate 想清楚的問題:
> **「同樣做 SR 量化專案,為什麼學生 / junior / senior 會做出三種完全不同的 deliverable?」**
>
> 這也是 RealSR-Edge **Page 2 (Quantization Recipes)** 的關鍵精神——不是「我做了多少 method」, 是「**我給 vendor 一個 decision package: recipe + reasoning + boundary**」。

---

## 三種 archetype 的核心差異

| 面向 | 🎓 學生 / 研究生 | 👶 Junior 工程師 | 💎 Senior 工程師 |
|---|---|---|---|
| **賣點** | 「我**發現 / 提出**了什麼」 | 「我**做了多少**東西」 | 「我**為什麼這樣決策**」 |
| **Framing** | Method invention | Coverage breadth | Decision support |
| **Customer 是誰?** | 學界 reviewer | (不清楚)| **下游 team(vendor / deploy team)** |
| **典型句型** | 「I propose X」 | 「I tried 8 methods」 | 「我選 X 因為 Y;沒選 Z 因為 W」 |
| **失敗模式** | 跟產業需求脫節 | 廣度沒收斂成 actionable | (相對少)over-confidence |
| **Slide 看起來** | Equation 多 / cite paper / contribution 段落 | 5 頁 8 個 ablation table 找不到主角 | 結構化(system/recipes/findings)+ 「→ Vendor input:」收尾 |

---

## 三種人對 「同一個 SR 量化問題」 會做什麼

### 🎓 學生作法

- Pick 一個特定 method demo(LSQ / SmoothQuant / 自己改 fake-quant)
- 對 SOTA papers 的 PSNR 數字 ablation
- Claim contribution / future work
- **不接 deployment 端**

→ Interviewer 反應:「**學術思維,不是 deploy engineer**」

### 👶 Junior 作法

- PTQ + QAT + LSQ + AWQ + SmoothQuant + 17 種 calibration ...
- **Coverage 廣但沒選邊**
- 「8 個 method 各有優缺點,看 case」
- 沒 surface 「**vendor 該 deploy 哪一個**」

→ Interviewer 反應:「**想堆更多東西的人,沒判斷力**」

### 💎 Senior 作法(你 RealSR-Edge 走的路)

- KPI-driven scope cut(minimize NPU vendor retrain probability)
- 4 條主軸**有 reasoning 地選擇**(PTQ/QAT、calibration、granularity、mixed precision)
- **3 類主動承認 out-of-scope**(architecture / bit-width / edge case)
- 每個 recipe 帶 「→ Vendor input:」 actionable output
- Honest scope boundary 顯式標註(verified / hypothesized / cannot-verify / out-of-scope)

→ Interviewer 反應:「**有 scope-cutting + 業務感,可以一起 work**」

---

## Page 2 「Quantization Recipes」 怎麼具體實現 senior framing

### 一、副標 wording

❌ 「**5 Optimization Methods I Implemented**」(junior 心態)
✅ 「**PyTorch-side Quantization Recipes**」(deliverable-oriented)

### 二、每個 visual 配 「→ Vendor input」

不是 「我做了 sensitivity sweep」,是:

> **Per-Layer Sensitivity** → Vendor: Keep `tail / upsampler.0 / head` in FP32; rest INT8

不是 「我比較 4 種 calibration」,是:

> **Calibration Ablation** → Vendor: Default to **max-abs** for PSNR-leaning, **percentile-99.99** for SSIM-leaning

每個 「→ Vendor:」 同時 demo:**recipe**(具體建議)+ **reasoning**(背後 trade-off)。

### 三、Decision trail 比 final config 重要

只給 vendor 「**用 top-2 FP32 + QAT**」 = 他遇到自己 NPU 不支援 FP32 fallback 就死掉。

給 vendor「**top-2 是基於這個 sensitivity ranking;ranking 來自 36 層 isolated INT8 swept;PTQ vs QAT 對照看**」 = 他可以**自己 adapt**。

> **「optimization path / 決策軌跡」 比 「optimization output / 最終 config」 對 vendor 更有用**——
> 拿到 path 可以 self-adapt,拿到 output 只能照抄。

---

## 兩個 「decision support」 的隱含元素

### A. **顯式承認 scope 邊界** = senior

```
✓ VERIFIED: ...
⚠ HYPOTHESIZED: ...
✗ CANNOT VERIFY: NPU dev board / vendor SDK
🚫 DELIBERATELY OUT-OF-SCOPE: lightweight architecture
   choice (vendor knows silicon constraint better);
   INT4 / mixed-bit (no consumer GPU support);
   edge-case robustness (needs vendor's customer
   distribution, not DIV2K)
```

學生 / junior 不會主動標 「**out-of-scope**」——他們覺得標出來像在認輸。

Senior 知道 「**主動承認 「我為什麼不做 X」 的 reasoning**」 比 「**真的去做 X**」 更展現判斷力。

### B. **典型句型** 的差異

「**我選 max-abs 作為 calibration default,因為 spread < 0.05 dB——但我把 trade-off 表給你,你的客戶 spec 偏 SSIM 就換 percentile-99.99**」

這句話同時 demo 了:
- **Recipe**(max-abs 是預設)
- **Reasoning**(spread < 0.05 dB 所以 robust)
- **Adaptability**(vendor 可換 percentile)
- **Boundary**(這個判斷不適用所有應用場景)

四件事**同一句**——這是 senior 的句型 muscle memory。

---

## Interview 場景的 ready answer

### 被問 「**你這個 SR 量化專案跟學生 paper 有什麼不同?**」

> 「**學生 paper 賣 method invention,我這個賣 vendor decision support**。我的每個 detector 都在回答 vendor 端會踩的 specific failure mode,不是 academic 量化基準。所以我**刻意 narrowed scope**——4 條主軸 + 3 類 out-of-scope reasoning,而不是把 paper 上的 method 全 demo 一次。**這是 production engineer 跟 PhD candidate 的 deliverable 差異**。」

### 被問 「**你跟其他工程師 candidate 有什麼差別?**」

> 「多數 candidate 用 「coverage 廣度」 證明能力——我用 「decision rationale 透明度」 證明。我的 deliverable 是 「vendor 拿到 4 個 recipe + 4 個對應 reasoning + 3 個 scope 邊界」 的 **decision package**,不是 「**我做了 N 件事**」 的 list。對 hand-off 場景,**vendor 拿前者可以 adapt,拿後者只能照抄**。」

### 被問 「**你這個 pipeline 完整嗎?**」

> 「**對 pre-handoff scope 我 cover 4 條主軸**(PTQ/QAT、calibration、granularity、mixed precision)。**3 類刻意 out-of-scope:** lightweight architecture(vendor knows silicon better)、INT4/mixed-bit(no consumer GPU support)、edge-case robustness(needs vendor's customer distribution)。**真實 small gap 是 calibration data distribution sensitivity** — is next-step,但不影響主軸。」

---

## 應用到未來其他 project 的判斷準則

任何技術 deliverable 設計時,問自己這三題:

1. **Customer 是誰?**
   - 學界 reviewer? → method invention 路線(學生模式)
   - 沒明確 customer? → 容易掉進 coverage breadth 陷阱(junior 模式)
   - **下游 team / vendor / 業務**? → **decision support 路線(senior 模式)**

2. **Deliverable 是 「我做了什麼」 還是 「他們需要什麼」?**
   - 「我做了 N 件事」 = junior 心態
   - 「他們拿到 X 可以 execute / adapt」 = senior 心態

3. **Scope 邊界顯式還是隱式?**
   - 「我都做了」(隱式邊界,被 fact-check 會崩)
   - 「我做了 X,沒做 Y 因為 Z」(顯式邊界,**這個 reasoning 才是 senior signal**)

---

## 一句話總結

> **學生賣 「我發現了什麼」,junior 賣 「我做了多少」,senior 賣 「我的決策路徑為什麼這樣設計」。**
>
> RealSR-Edge Page 2 「Quantization Recipes」 的關鍵精神:**不是 「5 個 method 」 的 demo,是 「給 vendor 的 4 個 recipe + 對應 reasoning + 3 條 boundary」 的 decision package**。
>
> 這個 framing 一旦練到 muscle memory,**任何 deliverable 設計都自動偏向 「decision support」 而非 「method demo」 / 「coverage report」**——這是 senior judgment 的本質。
