# Reading Calibration Histograms

A workflow for using activation histograms (collected during calibration)
to choose between calibration schemes (max-abs, percentile-99.X,
KL-divergence, ...). The output of this workflow is a **defensible
decision** — "for this model, scheme X is the right choice, here's the
visual + numerical evidence" — rather than a heuristic guess.

The doc is generic. The companion code in this repo
(`src/quantization/calibration_ablation.py`) produces the kind of
histogram figure described below; the workflow applies to any
fake-quant calibration pipeline.

---

## What's in a calibration-histogram figure

A typical figure has one subplot per representative layer (head, body,
upsampler, tail for an SR-style model; head, several blocks, classifier
head for classification, etc.). Each subplot shows:

| Element | What it means |
|---|---|
| **Gray bars** | Histogram of `\|x\|` (activation magnitude) seen during calibration. Built online with rescaling, so it covers `[0, max(\|x\|)]`. |
| **Y-axis: log scale** | Activation distributions are typically heavy-tailed. A linear axis makes the tail invisible; log makes it readable. |
| **Vertical dashed lines** | The `amax` chosen by each calibration scheme. `amax` is what `scale = amax / 127` is computed from for symmetric INT8. |
| **Color per scheme** | Convention used in this repo: red=max-abs, orange=99.99%ile, green=99.9%ile, blue=99.0%ile. |

Each subplot answers the question: **"For this layer, where does each
scheme place the cut-off, and what does the activation distribution
look like?"**

---

## The four typical distribution shapes

Vision / signal models tend to produce one of four shapes. The shape
determines which calibration scheme is correct.

### Shape A — Compact / saturated (no tail)

```
count (linear)
 │ ▆▆▆▆▆▆▆▆
 │ ▆▆▆▆▆▆▆▆
 │ ▆▆▆▆▆▆▆▆
 └────────── |x|
   0      max
```

- All values fall in a narrow range, distribution is roughly uniform or
  bell-shaped, drops to 0 quickly past some natural ceiling.
- Common after sigmoid / hard-sigmoid / clip layers; or for image
  inputs in `[0, 1]`.
- All four scheme lines cluster in nearly the same place.
- **Decision**: any scheme works. Default to max-abs (simplest).

### Shape B — Exponential-decay long tail (mass continues into the tail)

```
count (log)
 │ ▆
 │ ▆▆
 │   ▆▆▆
 │       ▆▆▆▆▆▆▆▆
 └──────────────── |x|
   0          max
```

- Most mass concentrates near 0; long tail extends out, decaying
  smoothly. **Tail bins are non-empty in log scale** (10^1, 10^2, ...
  counts) — i.e., the tail represents real, frequent values.
- Common in models without BN (SR EDSR-style), or in any layer where
  activation magnitude reflects content importance (high-contrast edges,
  high-frequency texture).
- The four scheme lines spread out from right (max-abs) leftward.
- **Decision**: **max-abs**. Percentile clipping cuts into real signal,
  hurting accuracy. The tail is informative, not noise.

### Shape C — Bimodal / true outliers

```
count (log)
 │ ▆▆▆▆▆
 │ ▆▆▆▆▆            ▆
 │ ▆▆▆▆▆            ▆ (isolated cluster)
 │ ▆▆▆▆▆▆▆▆        ▆
 │           ▆ ▆  ▆
 └──────────────── |x|
   0   main mass    max
```

- Main distribution has a clear cutoff, then **isolated** outlier mass
  much further to the right. Visible "gap" in the histogram.
- Common in BN-free Transformer activations (BERT/GPT outliers), or in
  models with training instabilities, or in models where a few feature
  channels carry disproportionate magnitude.
- Percentile-99.9 lines neatly between the main mass and the outlier
  cluster.
- **Decision**: **percentile clipping wins**. This is exactly the
  scenario percentile is designed for. Try 99.99 first (close to max-abs
  but kills the very-tail outliers), then 99.9 if outliers are large
  cluster.

### Shape D — Tall spike near 0 + long tail

```
count (log)
 │▆
 │ ▆
 │  ▆
 │   ▆▆
 │     ▆▆▆▆▆▆▆▆▆▆▆
 └────────────────── |x|
   0              max
```

- Massive spike at small values (most of the distribution), with a
  steady long tail extending far. Looks like B but more extreme.
- Common in residual-block interior layers; the residual stream tends
  to be small in magnitude with rare large excursions.
- The amax spread between schemes can look dramatic (5x, 8x, 10x), but
  this layer might also be **quantization-robust** because it carries
  little of the final output.
- **Decision**: cross-reference with sensitivity analysis. If this
  layer's PSNR-drop contribution is large, check whether the tail is
  continuous (Shape B) or has gaps (Shape C). If small, calibration
  choice barely matters — keep max-abs.

---

## Workflow: 4 steps to a decision

### Step 1 — Classify the shape

Look at each subplot. Match to A / B / C / D. The whole model rarely
falls into one shape — head/tail/output convs often differ from middle
layers.

### Step 2 — Inspect tail mass under max-abs

For each layer, look at the histogram bins **between max-abs and
percentile-99.9** (i.e. the "tail" region):

| What you see | Interpretation |
|---|---|
| Tail bins still show 10² – 10⁴ counts (log scale) | Tail is **real signal** — many samples land there. Don't clip. |
| Tail is sparse, 10⁰ – 10¹ scattered counts | Possibly outliers. Percentile clipping is a candidate. |
| Tail is empty until a sudden cluster | Definite **outliers**. Percentile clipping should help. |

### Step 3 — Measure the spread

Compare the gap between scheme lines. Useful ratios:

| Ratio | Meaning |
|---|---|
| `max-abs / percentile-99.99 < 1.05` | Tail is so thin it's almost nothing. All schemes equivalent. |
| `1.05 < max-abs / percentile-99.99 < 1.5` | Tail is real but mild. Either scheme reasonable. |
| `max-abs / percentile-99.99 > 1.5` | Tail extends far. Inspect Step 2 to decide. |
| `max-abs / percentile-99.0 > 3` | Massive spread. Either Shape D (tail = signal) or strong outliers (Shape C). Step 2 disambiguates. |

### Step 4 — Cross-reference with sensitivity analysis

A histogram-based decision is informed but not final. Confirm via:

1. **Run an ablation** (one PSNR per scheme on the val set). The relative
   ordering of PSNR drops is ground truth.
2. **Cross with the sensitivity sweep** (each layer's individual
   contribution to total INT8 PSNR drop). A layer that looks like Shape
   D but has near-zero sensitivity contribution doesn't need careful
   calibration — even bad clipping won't hurt.

If the ablation result agrees with the histogram intuition: ship it.
If the ablation result disagrees: trust the ablation, but
**investigate** — usually means the histogram aggregation across
calibration was misleading (too few samples, or unrepresentative
samples).

---

## Decision matrix (quick reference)

| Shape | Tail-mass signal? | Outliers? | Recommended scheme |
|---|---|---|---|
| A — Compact | n/a | n/a | max-abs (anything works) |
| B — Long tail with continuous mass | Yes | No | **max-abs** |
| C — Bimodal with isolated cluster | No | Yes | **percentile-99.9 or 99.99** |
| D — Spike + long tail | Depends — inspect bins | Possibly | Match to B or C after inspection |

If the layer is **quantization-robust** (low sensitivity), the right
column collapses to "any scheme; pick what's simplest to maintain
across the model".

---

## When percentile clipping is the wrong tool

Percentile clipping was popularized in LLM activation quantization
(BERT / GPT have well-documented outlier features). The implicit
assumption is **the tail is noise**.

For models where **the tail is signal**, percentile clipping is
actively destructive:

- **SR / restoration models** (no BN, residual streams): activation
  magnitude correlates with edge strength / texture frequency. Cutting
  the tail throws away high-detail information.
- **Detection backbone with focal-loss training**: rare-class
  activations live in the tail; cutting them hurts hard-example
  detection.
- **Generative models**: tail values often encode the diversity of
  generated samples; clipping reduces output variety.
- **Small models with regular weight distributions**: typically don't
  develop outliers in the first place; percentile is a non-fix.

**Rule of thumb**: if the model has BN at every block (most ImageNet
classifiers), percentile clipping is usually safe. If not (SR, GAN
generators, modern Transformers without LayerNorm in critical paths),
**measure first** with an ablation before defaulting to percentile.

---

## When percentile clipping is the right tool

Concrete scenarios where the histogram will show Shape C and percentile
will win:

1. **Transformer activations** with explicit outlier features (the
   "BERT outlier channel" problem; addressed properly by SmoothQuant,
   AWQ, etc., but percentile clipping helps as a first pass).
2. **Models trained with mixed precision** where occasional FP16
   overflow has produced a few extreme weight values that propagate to
   activations.
3. **Channels with rare-but-extreme values** (e.g., a feature channel
   that fires only on a specific input class).
4. **Per-channel weight outliers** when only per-tensor weight
   quantization is available — though this is more usually addressed by
   per-channel weight quantization itself.

In each of these, the histogram will visibly show **a gap** between the
main mass and the outlier cluster. If you don't see a gap, percentile
clipping is unlikely to help.

---

## Common reading mistakes

| Mistake | What goes wrong |
|---|---|
| Reading on linear y-axis | Tail is invisible; you'll think the distribution is Shape A when it's actually Shape B/C/D. |
| Looking at only one layer | Different layers have different shapes. Decision should be per-layer or at least per-block. |
| Trusting the histogram without ablation | Histograms are static stats; effective accuracy after quantization depends on **how the scale is used downstream**. Always validate with PSNR. |
| Using too few calibration samples | A small calibration set may miss the true tail (rare large activations). Use enough samples to cover the input distribution; for vision typically 50-500 images. |
| Ignoring sensitivity ranking | A Shape-D layer with near-zero sensitivity is fine to clip aggressively; a Shape-A layer with high sensitivity might still need careful per-channel scales. |

---

## Talking points

For portfolio / interview / report writing, two ready sentences:

> "I read calibration histograms before choosing a scheme. The four
> typical shapes (compact, long-tail, bimodal, spike+tail) each map to a
> different recommended scheme. For SR-style models without BN, the
> long-tail shape predominates and the tail carries signal — so max-abs
> usually beats percentile clipping. The decision is validated by an
> ablation across schemes; the histogram tells you *why* one wins."

> "Percentile clipping is the right tool when the histogram shows a
> visible gap between the main distribution and an outlier cluster. If
> there's no gap, the 'outliers' are real signal, and clipping them
> hurts."

---

## Cross-references

- [`quantization_terminology.md`](quantization_terminology.md) — formal
  terms for "calibration", "max-abs", "percentile", "fake-quant".
- [`deployment_methodology.md`](deployment_methodology.md) — Stage 2
  (Pre-deployment Analysis) is where this workflow sits.
- `src/quantization/fake_quant.py` — the histogram and percentile
  primitives that produce the data.
- `src/quantization/calibration_ablation.py` — the runner that produces
  the histogram figure plus a 4-scheme PSNR comparison.
