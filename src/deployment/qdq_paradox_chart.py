"""Generate the QDQ INT8 paradox bar chart for §3.3.

Plots, for each backend (ORT CPU EP / ORT CUDA EP / ORT TRT EP / Native TRT),
FP32 / FP16 / INT8 latency side-by-side. INT8 bars are colored red where they
are SLOWER than the same backend's FP32 (the paradox), green where INT8 finally
wins (Native TRT calibrator path). FP16 is included to give the full GPU-side
picture — on this consumer Ampere GPU, FP16 is actually the optimum across all
GPU backends (see §3.4 roofline). Log Y-axis fits CPU's ~50 ms next to GPU's
~1-7 ms range.

Run example::

    python -m src.deployment.qdq_paradox_chart \\
        --output results/qdq_paradox_chart.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Hard-coded latencies (mean, ms) from existing CSVs.
# Source: results/onnx_benchmark/edsr_200ep_full/benchmark.csv
#       + results/trt_benchmark/edsr_200ep/benchmark.csv
DATA = [
    # (backend label, FP32 ms, FP16 ms, INT8 ms)
    ("ORT CPU EP",   49.17, 50.55, 56.25),
    ("ORT CUDA EP",   5.28,  4.05,  6.57),
    ("ORT TRT EP",    3.28,  1.28,  4.33),
    ("Native TRT",    3.46,  1.50,  1.93),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    backends = [d[0] for d in DATA]
    fp32 = np.array([d[1] for d in DATA])
    fp16 = np.array([d[2] for d in DATA])
    int8 = np.array([d[3] for d in DATA])
    int8_ratios = int8 / fp32

    x = np.arange(len(backends))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12.5, 6.6))

    # FP32 bars (neutral grey, leftmost in each group)
    ax.bar(
        x - width, fp32, width,
        color="#7d8a99", edgecolor="#3a4654", linewidth=0.8,
        label="FP32 (per backend)",
    )

    # FP16 bars (orange, matching §3.2 PREC_COLOR convention; middle)
    ax.bar(
        x, fp16, width,
        color="#e8872a", edgecolor="#3a4654", linewidth=0.8,
        label="FP16 (per backend — GPU optimum on this HW)",
    )

    # INT8 bars: red if slower than FP32 (paradox), green if faster (bypass)
    int8_colors = ["#c44a3a" if r > 1 else "#2a8c4a" for r in int8_ratios]
    ax.bar(
        x + width, int8, width,
        color=int8_colors, edgecolor="#3a4654", linewidth=0.8,
    )

    # Log Y axis — CPU at 50 ms vs GPU at 1-7 ms otherwise crushes GPU bars
    ax.set_yscale("log")
    ax.set_ylim(1, 100)

    # Annotate INT8 bars with ratio + arrow direction (the paradox story)
    for i, (t, r) in enumerate(zip(int8, int8_ratios)):
        if r > 1:
            label = f"INT8 = {r:.2f}× FP32 ↑ slower"
            color = "#a83820"
        else:
            label = f"INT8 = {r:.2f}× FP32 ↓ faster"
            color = "#1d6535"
        ax.annotate(
            label,
            xy=(x[i] + width, t),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", fontsize=8.5, color=color, weight="bold",
        )

    # Numerical labels on FP32 / FP16 bars
    for i in range(len(backends)):
        ax.text(x[i] - width, fp32[i] * 1.06, f"{fp32[i]:.2f}",
                ha="center", fontsize=7.5, color="#3a4654")
        ax.text(x[i], fp16[i] * 1.06, f"{fp16[i]:.2f}",
                ha="center", fontsize=7.5, color="#7a4310")

    # Star-mark the absolute optimum (FP16 / ORT TRT EP at 1.28 ms)
    # Placed above the FP16 bar's numerical label so it doesn't clip on log scale.
    opt_idx = next(i for i, b in enumerate(backends) if b == "ORT TRT EP")
    ax.text(x[opt_idx], fp16[opt_idx] * 1.7, "★",
            ha="center", va="center", fontsize=15, color="#d4a017", zorder=5)
    ax.annotate(
        "absolute optimum",
        xy=(x[opt_idx], fp16[opt_idx] * 1.7),
        xytext=(28, 0), textcoords="offset points",
        ha="left", va="center",
        fontsize=8, color="#7a4310", style="italic",
    )

    # Background shading: ORT × 3 EP paradox region vs Native TRT bypass
    ax.axvspan(-0.5, 2.5, color="#c44a3a", alpha=0.06)
    ax.axvspan( 2.5, 3.5, color="#2a8c4a", alpha=0.08)
    # Region labels: top of plot, away from data
    ax.text(1.0, 78, "ORT × 3 EP region\nINT8 paradox",
            color="#a83820", fontsize=10, ha="center", weight="bold", alpha=0.9)
    ax.text(3.0, 78, "Native TRT region\ncalibrator bypass",
            color="#1d6535", fontsize=10, ha="center", weight="bold", alpha=0.9)

    # Legend (custom, since INT8 bars have mixed colors)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#7d8a99", edgecolor="#3a4654", label="FP32 (per backend)"),
        Patch(facecolor="#e8872a", edgecolor="#3a4654", label="FP16 (per backend; ★ marks the absolute optimum)"),
        Patch(facecolor="#c44a3a", edgecolor="#3a4654", label="INT8 (slower than FP32 — paradox)"),
        Patch(facecolor="#2a8c4a", edgecolor="#3a4654", label="INT8 (faster than FP32 — bypass)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9, framealpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(backends, fontsize=10)
    ax.set_ylabel("Latency (ms · log scale, lower better)")
    ax.set_title(
        "QDQ INT8 Paradox · ORT × 3 EP all slower than FP32 · Native TRT bypass via calibrator · FP16 wins on every GPU backend",
        fontsize=11.5,
    )
    ax.grid(True, axis="y", which="both", alpha=0.25)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
