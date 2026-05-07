"""Generate the QDQ INT8 paradox bar chart for Page 3 Card 1.

Plots, for each backend (ORT CPU EP / ORT CUDA EP / ORT TRT EP / Native TRT),
the FP32 vs INT8 latency. INT8 bars are colored red where they are SLOWER than
the same backend's FP32 (the paradox), green where INT8 finally wins (Native TRT
calibrator path). Log Y-axis to fit CPU's ~50 ms next to GPU's ~1-7 ms range.

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
    # (backend label, FP32 ms, INT8 ms)
    ("ORT CPU EP",   49.17, 56.25),
    ("ORT CUDA EP",   5.28,  6.57),
    ("ORT TRT EP",    3.28,  4.33),
    ("Native TRT",    3.46,  1.93),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    backends = [d[0] for d in DATA]
    fp32 = np.array([d[1] for d in DATA])
    int8 = np.array([d[2] for d in DATA])
    ratios = int8 / fp32

    x = np.arange(len(backends))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    # FP32 bars (neutral grey)
    fp32_bars = ax.bar(
        x - width/2, fp32, width,
        label="FP32 baseline (this backend)",
        color="#7d8a99", edgecolor="#3a4654", linewidth=0.8,
    )

    # INT8 bars: red if slower than FP32 (paradox), green if faster (bypass)
    int8_colors = ["#c44a3a" if r > 1 else "#2a8c4a" for r in ratios]
    int8_bars = ax.bar(
        x + width/2, int8, width,
        color=int8_colors, edgecolor="#3a4654", linewidth=0.8,
    )

    # Log Y axis — CPU at 50 ms vs GPU at 1-7 ms otherwise crushes GPU bars
    ax.set_yscale("log")
    ax.set_ylim(1, 100)

    # Annotate INT8 bars with ratio + arrow direction
    for i, (f, t, r) in enumerate(zip(fp32, int8, ratios)):
        if r > 1:
            label = f"INT8 = {r:.2f}× FP32 ↑ slower"
            color = "#a83820"
        else:
            label = f"INT8 = {r:.2f}× FP32 ↓ faster"
            color = "#2a8c4a"
        ax.annotate(
            label,
            xy=(x[i] + width/2, t),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", fontsize=9, color=color, weight="bold",
        )

    # Background shading: ORT × 3 EP paradox region vs Native TRT bypass
    ax.axvspan(-0.5, 2.5, color="#c44a3a", alpha=0.06)
    ax.axvspan( 2.5, 3.5, color="#2a8c4a", alpha=0.08)
    # Region labels: shifted down, away from y-axis top where legend sits
    ax.text(1.0, 28, "ORT × 3 EP region\nINT8 paradox",
            color="#a83820", fontsize=10, ha="center", weight="bold", alpha=0.9)
    ax.text(3.0, 28, "Native TRT region\ncalibrator bypass",
            color="#1d6535", fontsize=10, ha="center", weight="bold", alpha=0.9)

    # Numerical labels on top of FP32 bars (small, grey)
    for i, f in enumerate(fp32):
        ax.text(x[i] - width/2, f * 1.05, f"{f:.2f}",
                ha="center", fontsize=8, color="#3a4654")

    # Legend (custom, since INT8 bars have mixed colors)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#7d8a99", edgecolor="#3a4654", label="FP32 (per backend)"),
        Patch(facecolor="#c44a3a", edgecolor="#3a4654", label="INT8 (slower than FP32 — paradox)"),
        Patch(facecolor="#2a8c4a", edgecolor="#3a4654", label="INT8 (faster than FP32 — bypass)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=9, framealpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(backends, fontsize=10)
    ax.set_ylabel("Latency (ms · log scale, lower better)")
    ax.set_title(
        "QDQ INT8 Paradox · ORT × 3 EP all slower than FP32 · Native TRT bypass via calibrator",
        fontsize=12,
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
