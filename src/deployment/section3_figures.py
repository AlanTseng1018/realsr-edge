"""Generate three figures for README §3 (cross-stack deployment).

Outputs:
    results/deploy_overview/validation_pipeline_layers.png   §3.1 — concept
    results/onnx_benchmark/edsr_200ep_full/precision_ep_scatter.png   §3.2
    cpp_inference/cross_language_panel.png   §3.5

Run:
    python -m src.deployment.section3_figures
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

PREC_COLOR = {"FP32": "#4c8cbf", "FP16": "#e8872a", "INT8": "#2ca02c"}
PROVIDER_MARKER = {"tensorrt": "s", "cuda": "o", "cpu": "^"}


# ---------------------------------------------------------------------------
# §3.1 — three-layer cross-stack validation diagram
# ---------------------------------------------------------------------------
def make_validation_pipeline_layers(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    layers = [
        {
            "title": "Layer 1 · EXPORT",
            "subtitle": "PyTorch → ONNX",
            "bullets": [
                "Numerical fidelity (max pixel diff vs PyTorch reference)",
                "Symmetric vs asymmetric calibration (TRT EP requires symmetric)",
                "Static vs dynamic shape, opset compatibility",
            ],
            "y": 8.6,
            "color": "#dceaf5",
        },
        {
            "title": "Layer 2 · RUNTIME",
            "subtitle": "ORT (CPU / CUDA / TensorRT EP) + Native TensorRT",
            "bullets": [
                "Per-EP × per-precision latency × accuracy matrix",
                "QDQ fusion failures, INT32-bias paradox",
                "Tensor-core utilization, roofline classification",
            ],
            "y": 4.8,
            "color": "#fde8d0",
        },
        {
            "title": "Layer 3 · LANGUAGE",
            "subtitle": "Python ↔ C++ binary (cpp_inference/edsr_runner.cpp)",
            "bullets": [
                "Bit-parity check (Python output vs C++ output)",
                "Cross-EP correctness (CPU = CUDA = TensorRT to float noise)",
                "Vendor SDK port skeleton (ORT → SNPE / TRT C++ / RKNN)",
            ],
            "y": 1.0,
            "color": "#d5ebd8",
        },
    ]

    for layer in layers:
        rect = mpatches.FancyBboxPatch(
            (0.4, layer["y"]),
            9.2,
            3.0,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            linewidth=1.5,
            edgecolor="#333",
            facecolor=layer["color"],
        )
        ax.add_patch(rect)
        ax.text(
            0.7, layer["y"] + 2.45, layer["title"],
            fontsize=14, fontweight="bold", color="#222",
        )
        ax.text(
            0.7, layer["y"] + 2.0, layer["subtitle"],
            fontsize=10.5, fontstyle="italic", color="#444",
        )
        for i, bullet in enumerate(layer["bullets"]):
            ax.text(
                0.85, layer["y"] + 1.4 - i * 0.45,
                f"•  {bullet}",
                fontsize=10, color="#222",
            )

    # Arrows between layers
    for y_top, y_bot in [(8.6, 7.8), (4.8, 4.0)]:
        ax.annotate(
            "",
            xy=(5, y_bot), xytext=(5, y_top),
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#555",
                             mutation_scale=25),
        )

    ax.set_title(
        "Cross-stack deployment: three validation layers",
        fontsize=14, fontweight="bold", pad=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# §3.2 — PSNR vs latency scatter (3 precision × 3 EP = 9 points)
# ---------------------------------------------------------------------------
def make_precision_ep_scatter(csv_path: Path, out_path: Path) -> None:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "precision": r["precision"],
                "provider": r["provider"],
                "psnr": float(r["psnr_db"]),
                "latency": float(r["latency_ms_mean"]),
            })

    # Per-point label offsets (in points) to avoid overlap in tight clusters.
    LABEL_OFFSETS = {
        ("FP16", "tensorrt"): (12, 8),     # bottom-left, room to the right
        ("FP32", "tensorrt"): (-12, 14),   # top-left
        ("FP16", "cuda"):     (-10, -18),  # bottom-left of point
        ("FP32", "cuda"):     (12, 8),     # top-right
        ("INT8", "tensorrt"): (-12, -18),  # below-left
        ("INT8", "cuda"):     (12, 8),     # above-right
        ("FP32", "cpu"):      (-10, 14),   # above-left
        ("FP16", "cpu"):      (-10, -18),  # below-left
        ("INT8", "cpu"):      (-10, -18),  # below-left
    }

    fig, ax = plt.subplots(figsize=(13, 7))

    for row in rows:
        ax.scatter(
            row["latency"], row["psnr"],
            c=PREC_COLOR[row["precision"]],
            marker=PROVIDER_MARKER[row["provider"]],
            s=220,
            edgecolors="#222", linewidths=1.3,
            zorder=3,
        )
        offset = LABEL_OFFSETS[(row["precision"], row["provider"])]
        ax.annotate(
            f"{row['precision']}/{row['provider']}",
            (row["latency"], row["psnr"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=PREC_COLOR[row["precision"]],
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.8, 95)
    ax.set_ylim(27.245, 27.465)
    ax.set_xlabel("Latency per per-tile (ms, log scale)  ↓", fontsize=11.5)
    ax.set_ylabel("Validation PSNR (dB)  ↑", fontsize=11.5)
    ax.set_title(
        "ONNX × 3 EP × 3 precision: PSNR vs latency  (bottom-left corner = best)",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    # Highlight the best corner with a downward arrow from text to FP16/TRT point.
    ax.annotate(
        "FP16 / TensorRT  →  optimal\n(this consumer Ampere GPU)",
        xy=(1.28, 27.437), xytext=(2.0, 27.355),
        fontsize=9.5, color="#444",
        arrowprops=dict(arrowstyle="->", color="#666", lw=1.2),
        ha="left",
    )

    # Cluster annotations
    ax.text(
        50, 27.300,
        "CPU cluster\n(another regime, ~10× slower)",
        ha="center", fontsize=9, color="#555", style="italic",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb",
                   boxstyle="round,pad=0.3"),
    )

    # Two legends — placed OUTSIDE the plot to keep the data area clean.
    prec_handles = [
        plt.Line2D([], [], marker="o", linestyle="",
                    color=PREC_COLOR[p], markersize=11, label=p)
        for p in ["FP32", "FP16", "INT8"]
    ]
    prov_handles = [
        plt.Line2D([], [], marker=PROVIDER_MARKER[p], linestyle="",
                    color="#666", markersize=11, label=p)
        for p in ["tensorrt", "cuda", "cpu"]
    ]

    leg1 = ax.legend(
        handles=prec_handles, loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        title="Precision",
        fontsize=10, title_fontsize=10.5, framealpha=0.95,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=prov_handles, loc="upper left",
        bbox_to_anchor=(1.015, 0.65),
        title="Provider",
        fontsize=10, title_fontsize=10.5, framealpha=0.95,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# §3.5 — C++ vs Python latency parity + cross-EP PSNR triangulation
# ---------------------------------------------------------------------------
def make_cross_language_panel(out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- Left: per-tile latency (96x96 LR), C++ vs Python ---
    eps = ["CUDA EP", "TensorRT EP"]
    py_lat = [5.28, 1.28]
    cpp_lat = [4.18, 1.41]
    cpp_std = [0.04, 0.06]

    x = np.arange(len(eps))
    width = 0.36

    bars_py = ax1.bar(
        x - width / 2, py_lat, width,
        label="Python (ORT)",
        color="#4c8cbf", edgecolor="#222", linewidth=0.9,
    )
    bars_cpp = ax1.bar(
        x + width / 2, cpp_lat, width,
        label="C++ (edsr_runner.exe)",
        color="#e8872a", edgecolor="#222", linewidth=0.9,
        yerr=cpp_std, capsize=4,
    )

    for bar, val in zip(bars_py, py_lat):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.12,
                  f"{val:.2f}", ha="center", fontsize=9.5)
    for bar, val in zip(bars_cpp, cpp_lat):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.12,
                  f"{val:.2f}", ha="center", fontsize=9.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(eps)
    ax1.set_ylabel("Latency per per-tile (ms)  ↓", fontsize=11)
    ax1.set_title(
        "Per-tile latency — C++ vs Python  (96×96 LR → 192×192 SR)",
        fontsize=11.5, fontweight="bold", pad=8,
    )
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(py_lat + cpp_lat) * 1.32)

    ax1.text(
        0.02, 0.86,
        "CUDA: C++ 21% faster (less wrapper overhead)\n"
        "TRT: within noise (engine kernel dominates)",
        transform=ax1.transAxes,
        fontsize=8.8, color="#444", style="italic",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    # --- Right: full-frame cross-EP PSNR (3 bars, zoomed) ---
    eps_full = ["CPU", "CUDA", "TensorRT"]
    psnr_full = [29.450, 29.450, 29.446]
    bar_colors = ["#888888", "#4c8cbf", "#cc4444"]

    bars = ax2.bar(
        eps_full, psnr_full,
        color=bar_colors, edgecolor="#222", linewidth=0.9, width=0.55,
    )
    for bar, val in zip(bars, psnr_full):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.0008,
                  f"{val:.3f}", ha="center", fontsize=10.5,
                  fontweight="bold")

    ax2.set_ylim(29.430, 29.462)
    ax2.set_ylabel("PSNR vs HR (dB)  ↑", fontsize=11)
    ax2.set_title(
        "Full-frame cross-EP PSNR — same C++ binary, three backends\n"
        "(0879.png, 1020×936 LR → 2040×1872 SR)",
        fontsize=11.5, fontweight="bold", pad=8,
    )
    ax2.grid(axis="y", alpha=0.3)

    ax2.text(
        0.5, 0.06,
        "Spread = 0.004 dB  →  three EPs match within float-rounding noise.\n"
        "C++ binary is correct AND backend-invariant.",
        ha="center",
        transform=ax2.transAxes,
        fontsize=9, color="#444", style="italic",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
    )

    fig.suptitle(
        "Cross-language correctness: C++ matches Python, and is consistent across EPs",
        fontsize=13, fontweight="bold", y=1.02,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    csv_path = REPO / "results/onnx_benchmark/edsr_200ep_full/benchmark.csv"

    make_validation_pipeline_layers(
        REPO / "results/deploy_overview/validation_pipeline_layers.png"
    )
    make_precision_ep_scatter(
        csv_path,
        REPO / "results/onnx_benchmark/edsr_200ep_full/precision_ep_scatter.png",
    )
    make_cross_language_panel(
        REPO / "cpp_inference/cross_language_panel.png"
    )


if __name__ == "__main__":
    main()
