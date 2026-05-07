"""Deployment results visualizer — single PNG summary.

Reads benchmark and profiling outputs and writes one large PNG with
5 subplots:

  [row 0]  Accuracy vs Latency  |  ORT TRT EP vs Native TRT
  [row 1]  Roofline model        |  Kernel breakdown
  [row 2]  Size vs Accuracy (centred)

Run example::

    python -m src.deployment.visualize_results --ort-benchmark  results/onnx_benchmark/edsr_200ep_trt_v2 --trt-benchmark  results/trt_benchmark/edsr_200ep_v2 --trt-profile    results/trt_profile/edsr_200ep --output-dir     results/deploy_report/edsr_200ep
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np


PREC_COLOR  = {"FP32": "#4c8cbf", "FP16": "#e8872a", "INT8": "#2ca02c"}
PREC_MARKER = {"FP32": "o", "FP16": "s", "INT8": "D"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_kernel_breakdown(path: Path) -> dict[str, dict]:
    """Parse profile_report.md for kernel breakdown percentages."""
    md = path / "profile_report.md"
    if not md.exists():
        return {}
    result: dict[str, dict] = {}
    current = None
    with md.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("### "):
                current = line[4:].strip()
                result[current] = {}
            if current and "Compute kernels" in line:
                result[current]["compute"] = float(line.split(":")[1].strip().rstrip("%"))
            if current and "Memory transfer" in line:
                result[current]["memory_transfer"] = float(line.split(":")[1].strip().rstrip("%"))
            if current and "Other / overhead" in line:
                result[current]["other"] = float(line.split(":")[1].strip().rstrip("%"))
    return result


# ---------------------------------------------------------------------------
# Individual subplot drawers
# ---------------------------------------------------------------------------

def draw_psnr_latency(ax, trt_rows: list[dict]) -> None:
    for row in trt_rows:
        prec = row["precision"]
        psnr = _f(row.get("psnr_db"))
        lat  = _f(row.get("latency_ms_mean"))
        if psnr is None or lat is None:
            continue
        ax.scatter(lat, psnr, c=PREC_COLOR[prec], marker=PREC_MARKER[prec],
                   s=160, zorder=5)
        ax.annotate(f"  {prec}", (lat, psnr), fontsize=9,
                    color=PREC_COLOR[prec], va="center")

    ax.set_xlabel("Latency (ms) — CUDA Events")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Accuracy vs Latency\n(Native TRT Engine)")
    patches = [mpatches.Patch(color=c, label=p) for p, c in PREC_COLOR.items()]
    ax.legend(handles=patches, fontsize=8)


def draw_ort_vs_trt(ax, ort_rows: list[dict], trt_rows: list[dict]) -> None:
    precs = ["FP32", "FP16", "INT8"]

    ort_lat = {}
    for row in ort_rows:
        prec = row.get("precision", "").upper()
        if "tensorrt" in row.get("provider", "").lower():
            ort_lat[prec] = _f(row.get("latency_ms_mean"))

    trt_lat = {row["precision"]: _f(row.get("latency_ms_mean")) for row in trt_rows}

    x = np.arange(len(precs))
    w = 0.35
    colors = [PREC_COLOR[p] for p in precs]

    b1 = ax.bar(x - w/2, [ort_lat.get(p) or 0 for p in precs],
                w, color=colors, alpha=0.45, label="ORT TRT EP", edgecolor="none")
    b2 = ax.bar(x + w/2, [trt_lat.get(p) or 0 for p in precs],
                w, color=colors, alpha=1.0,  label="Native TRT", edgecolor="none")

    for bars, vals in [(b1, [ort_lat.get(p) for p in precs]),
                       (b2, [trt_lat.get(p) for p in precs])]:
        for bar, val in zip(bars, vals):
            if val:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.04,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(precs)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("ORT TRT EP vs Native TRT\n(faded = ORT, solid = native)")
    ax.legend(fontsize=8)


def draw_roofline(ax, profile_meta: dict) -> None:
    gpu      = profile_meta.get("gpu", {})
    rows     = profile_meta.get("results", [])
    mem_bw   = gpu.get("mem_bw_gbs",       936.2)
    peak_fp32= gpu.get("peak_fp32_tflops",  35.58) * 1e3
    peak_fp16= gpu.get("peak_fp16_tflops", 142.3)  * 1e3
    peak_int8= gpu.get("peak_int8_tops",   284.6)  * 1e3

    ai = np.logspace(-1, 4, 500)
    ax.loglog(ai, np.minimum(mem_bw * ai, peak_fp32),
              color=PREC_COLOR["FP32"], lw=1.8, ls="-",
              label=f"FP32 ({peak_fp32/1e3:.0f} TFLOPS)")
    ax.loglog(ai, np.minimum(mem_bw * ai, peak_fp16),
              color=PREC_COLOR["FP16"], lw=1.8, ls="--",
              label=f"FP16 TC ({peak_fp16/1e3:.0f} TFLOPS)")
    ax.loglog(ai, np.minimum(mem_bw * ai, peak_int8),
              color=PREC_COLOR["INT8"], lw=1.8, ls=":",
              label=f"INT8 TC ({peak_int8/1e3:.0f} TOPS)")

    for r in rows:
        prec  = r.get("precision", "")
        ai_pt = _f(r.get("ai"))
        gf    = _f(r.get("achieved_gflops"))
        if ai_pt is None or gf is None:
            continue
        ax.scatter(ai_pt, gf, c=PREC_COLOR.get(prec, "k"),
                   s=120, marker="D", zorder=6)
        ax.annotate(f"  {prec}", (ai_pt, gf), fontsize=8,
                    color=PREC_COLOR.get(prec, "k"))

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title(f"Roofline — {gpu.get('name', 'GPU')}")
    ax.legend(fontsize=8)
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(10, peak_int8 * 3)


def draw_kernel_breakdown(ax, kernel_data: dict) -> None:
    precs = [p for p in ["FP32", "FP16", "INT8"] if p in kernel_data]
    if not precs:
        ax.text(0.5, 0.5, "No kernel data", ha="center", va="center",
                transform=ax.transAxes)
        return

    compute = [kernel_data[p].get("compute", 0)         for p in precs]
    memory  = [kernel_data[p].get("memory_transfer", 0) for p in precs]
    other   = [kernel_data[p].get("other", 0)           for p in precs]

    x = np.arange(len(precs))
    w = 0.5
    b1 = ax.bar(x, compute, w, label="Compute",  color="#4c8cbf")
    b2 = ax.bar(x, memory,  w, bottom=compute,   label="Memory transfer", color="#e8872a")
    b3 = ax.bar(x, other,   w,
                bottom=[c + m for c, m in zip(compute, memory)],
                label="Other / overhead", color="#cccccc")

    for bars, vals in [(b1, compute), (b2, memory), (b3, other)]:
        for bar, val in zip(bars, vals):
            if val > 4:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(precs)
    ax.set_ylabel("% of CUDA device time")
    ax.set_ylim(0, 115)
    ax.set_title("Kernel Breakdown\n(torch.profiler CUDA activity)")
    ax.legend(fontsize=8)


def draw_size_vs_psnr(ax, trt_rows: list[dict]) -> None:
    for row in trt_rows:
        prec = row["precision"]
        psnr = _f(row.get("psnr_db"))
        size = _f(row.get("engine_size_mb"))
        if psnr is None or size is None:
            continue
        ax.scatter(size, psnr, c=PREC_COLOR[prec], marker=PREC_MARKER[prec],
                   s=160, zorder=5)
        ax.annotate(f"  {prec} {size:.1f} MB", (size, psnr),
                    fontsize=9, color=PREC_COLOR[prec], va="center")

    ax.set_xlabel("Engine size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Size vs Accuracy Trade-off")
    patches = [mpatches.Patch(color=c, label=p) for p, c in PREC_COLOR.items()]
    ax.legend(handles=patches, fontsize=8)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_figure(
    trt_rows: list[dict],
    ort_rows: list[dict],
    profile_meta: dict,
    kernel_data: dict,
    output_path: Path,
    hardware: str,
    trt_version: str,
) -> None:
    fig = plt.figure(figsize=(16, 18))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.32,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.05)

    ax_psnr_lat   = fig.add_subplot(gs[0, 0])
    ax_ort_vs_trt = fig.add_subplot(gs[0, 1])
    ax_roofline   = fig.add_subplot(gs[1, 0])
    ax_kernel     = fig.add_subplot(gs[1, 1])
    ax_size_psnr  = fig.add_subplot(gs[2, :])   # full-width bottom row

    if trt_rows:
        draw_psnr_latency(ax_psnr_lat, trt_rows)
    if ort_rows and trt_rows:
        draw_ort_vs_trt(ax_ort_vs_trt, ort_rows, trt_rows)
    if profile_meta:
        draw_roofline(ax_roofline, profile_meta)
    if kernel_data:
        draw_kernel_breakdown(ax_kernel, kernel_data)
    if trt_rows:
        draw_size_vs_psnr(ax_size_psnr, trt_rows)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"EDSR Deployment Summary  |  {hardware}  |  TRT {trt_version}  |  {timestamp}",
        fontsize=13, fontweight="bold", y=0.97,
    )

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate deployment summary PNG.")
    p.add_argument("--ort-benchmark", type=str, default=None)
    p.add_argument("--trt-benchmark", type=str, required=True)
    p.add_argument("--trt-profile",   type=str, default=None)
    p.add_argument("--output-dir",    type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trt_rows    = load_csv(Path(args.trt_benchmark) / "benchmark.csv")
    ort_rows    = load_csv(Path(args.ort_benchmark) / "benchmark.csv") if args.ort_benchmark else []
    profile_meta= load_json(Path(args.trt_profile) / "metadata.json") if args.trt_profile else {}
    kernel_data = load_kernel_breakdown(Path(args.trt_profile)) if args.trt_profile else {}

    trt_meta = load_json(Path(args.trt_benchmark) / "metadata.json")
    hardware    = trt_meta.get("device_name", "GPU")
    trt_version = trt_meta.get("trt_version", "—")

    print(f"TRT rows: {len(trt_rows)}, ORT rows: {len(ort_rows)}, "
          f"profile: {'yes' if profile_meta else 'no'}")

    build_figure(
        trt_rows, ort_rows, profile_meta, kernel_data,
        output_path=out / "deploy_summary.png",
        hardware=hardware,
        trt_version=trt_version,
    )


if __name__ == "__main__":
    main()
