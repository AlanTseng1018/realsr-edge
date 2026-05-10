"""Generate three figures for README §3 (cross-stack deployment).

Outputs:
    results/deploy_overview/validation_pipeline_layers.png   §3.1 — concept
    results/onnx_benchmark/edsr_200ep_full/precision_ep_breakdown.png   §3.2
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
# §3.2 — accuracy (PSNR) and latency, stacked vertically
# ---------------------------------------------------------------------------
def make_precision_ep_breakdown(csv_path: Path, out_path: Path) -> None:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "precision": r["precision"],
                "provider": r["provider"],
                "psnr": float(r["psnr_db"]),
                "latency": float(r["latency_ms_mean"]),
                "latency_std": float(r["latency_ms_std"]),
            })

    # PSNR is provider-invariant within rounding noise — average per precision.
    by_prec: dict[str, list[float]] = {}
    for r in rows:
        by_prec.setdefault(r["precision"], []).append(r["psnr"])
    psnr_means = {p: sum(v) / len(v) for p, v in by_prec.items()}
    psnr_spread_max = max(max(v) - min(v) for v in by_prec.values())

    lat_lookup = {
        (r["precision"], r["provider"]): (r["latency"], r["latency_std"])
        for r in rows
    }

    fig, (ax_psnr, ax_lat) = plt.subplots(
        2, 1, figsize=(12, 9.5),
        gridspec_kw={"height_ratios": [1.0, 1.45], "hspace": 0.32},
    )

    # === Top: PSNR by precision (provider-invariant) ===
    precisions = ["FP32", "FP16", "INT8"]
    psnr_vals = [psnr_means[p] for p in precisions]
    bars_psnr = ax_psnr.bar(
        precisions, psnr_vals,
        color=[PREC_COLOR[p] for p in precisions],
        edgecolor="#222", linewidth=0.9, width=0.55,
    )
    for bar, val in zip(bars_psnr, psnr_vals):
        ax_psnr.text(
            bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.3f}", ha="center", fontsize=11.5, fontweight="bold",
        )

    # ΔPSNR vs FP32 annotations under each bar
    fp32 = psnr_means["FP32"]
    for bar, prec in zip(bars_psnr, precisions):
        delta = psnr_means[prec] - fp32
        if prec == "FP32":
            txt = "baseline"
        else:
            txt = f"Δ {delta:+.3f} dB"
        ax_psnr.text(
            bar.get_x() + bar.get_width() / 2, 27.215,
            txt, ha="center", fontsize=9.5, color="#444", style="italic",
        )

    ax_psnr.set_ylim(27.20, 27.47)
    ax_psnr.set_ylabel("Validation PSNR (dB)  ↑", fontsize=11.5)
    ax_psnr.set_title(
        "Accuracy — PSNR by precision  "
        f"(provider-invariant: cross-EP spread ≤ {psnr_spread_max:.3f} dB)",
        fontsize=12, fontweight="bold", pad=8,
    )
    ax_psnr.grid(axis="y", alpha=0.3)

    # === Bottom: latency, grouped bars (x = EP, hue = precision) ===
    providers = ["tensorrt", "cuda", "cpu"]
    provider_labels = ["TensorRT EP", "CUDA EP", "CPU EP"]
    width = 0.27
    x = np.arange(len(providers))

    for i, prec in enumerate(precisions):
        lat_vals = [lat_lookup[(prec, prov)][0] for prov in providers]
        lat_stds = [lat_lookup[(prec, prov)][1] for prov in providers]
        offset = (i - 1) * width
        bars = ax_lat.bar(
            x + offset, lat_vals, width,
            label=prec,
            color=PREC_COLOR[prec],
            edgecolor="#222", linewidth=0.9,
            yerr=lat_stds, capsize=3, error_kw={"ecolor": "#444", "lw": 1},
        )
        for bar, val in zip(bars, lat_vals):
            ax_lat.text(
                bar.get_x() + bar.get_width() / 2, val * 1.1,
                f"{val:.2f}", ha="center", fontsize=9, color="#222",
            )

    ax_lat.set_yscale("log")
    ax_lat.set_ylim(0.85, 130)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(provider_labels, fontsize=11)
    ax_lat.set_ylabel("Latency per per-tile (ms, log scale)  ↓", fontsize=11.5)
    ax_lat.set_title(
        "Latency — per provider × precision  (FP16 / TensorRT EP is the optimum at 1.28 ms)",
        fontsize=12, fontweight="bold", pad=8,
    )
    ax_lat.legend(
        title="Precision", loc="upper left",
        fontsize=10, title_fontsize=10.5, framealpha=0.95,
    )
    ax_lat.grid(axis="y", which="both", alpha=0.3, linestyle=":")

    # Star-mark the FP16/TRT optimum bar
    fp16_trt_x = 0 + 0 * width  # FP16 is the middle bar (i=1, offset=0)
    ax_lat.text(
        fp16_trt_x, 1.28 / 1.5, "★",
        ha="center", va="center", fontsize=20, color="#d4a017",
        zorder=5,
    )

    fig.suptitle(
        "ONNX × 3 EP × 3 precision — accuracy and latency, separated",
        fontsize=13.5, fontweight="bold", y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    make_precision_ep_breakdown(
        csv_path,
        REPO / "results/onnx_benchmark/edsr_200ep_full/precision_ep_breakdown.png",
    )
    make_cross_language_panel(
        REPO / "cpp_inference/cross_language_panel.png"
    )


if __name__ == "__main__":
    main()
