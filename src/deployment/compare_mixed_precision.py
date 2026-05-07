"""Overlay PTQ vs QAT mixed-precision sweep curves into a single comparison chart.

Reads two ``mixed_precision_sweep.csv`` files (PTQ-based and QAT-based) and
produces a single PNG showing both PSNR-vs-N curves, the underlying baselines
(FP32, all-INT8 for each), and the recovery story.

Run example::

    python -m src.deployment.compare_mixed_precision \\
        --ptq-csv  results/mixed_precision/edsr_200ep/mixed_precision_sweep.csv \\
        --qat-csv  results/mixed_precision/edsr_200ep_qat/mixed_precision_sweep.csv \\
        --output   results/mixed_precision/ptq_vs_qat_sweep.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path: Path) -> tuple[list[int], list[float], list[float]]:
    n, psnr, ssim = [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n.append(int(row["n_fp32_layers"]))
            psnr.append(float(row["psnr"]))
            ssim.append(float(row["ssim"]))
    order = sorted(range(len(n)), key=lambda i: n[i])
    return [n[i] for i in order], [psnr[i] for i in order], [ssim[i] for i in order]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ptq-csv", required=True)
    p.add_argument("--qat-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fp32-baseline", type=float, default=None,
                   help="Original FP32 baseline PSNR (e.g. 27.439). Drawn as a "
                        "reference line so the QAT-vs-FP32 cross-over is visible.")
    args = p.parse_args()

    ptq_n, ptq_psnr, _ = load(Path(args.ptq_csv))
    qat_n, qat_psnr, _ = load(Path(args.qat_csv))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(qat_n, qat_psnr, marker="^", linewidth=2.2, color="#2a8c4a",
            label="QAT-based mixed precision")
    ax.plot(ptq_n, ptq_psnr, marker="o", linewidth=2.2, color="#1f6fb8",
            label="PTQ-based mixed precision")

    # Reference baselines (per-INT8 starting point for each path)
    ax.axhline(qat_psnr[0], linestyle="--", linewidth=1.0, color="#2a8c4a", alpha=0.5,
               label=f"QAT All-INT8 baseline ({qat_psnr[0]:.3f} dB)")
    ax.axhline(ptq_psnr[0], linestyle="--", linewidth=1.0, color="#1f6fb8", alpha=0.5,
               label=f"PTQ All-INT8 baseline ({ptq_psnr[0]:.3f} dB)")

    # ★ Original FP32 baseline -- the headline anchor for "QAT exceeds FP32"
    if args.fp32_baseline is not None:
        ax.axhline(args.fp32_baseline, linestyle="-", linewidth=1.6, color="#c44a1a", alpha=0.7,
                   label=f"Original FP32 baseline ({args.fp32_baseline:.3f} dB)")

    # Annotations
    ax.annotate("PTQ plateaus ~ top-4",
                xy=(4, ptq_psnr[min(4, len(ptq_psnr)-1)]), xytext=(4.4, ptq_psnr[1]+0.005),
                fontsize=9, color="#1f6fb8",
                arrowprops=dict(arrowstyle="->", color="#1f6fb8", alpha=0.7))

    # ★ Highlight: entire QAT curve sits above original FP32 baseline
    if args.fp32_baseline is not None:
        ax.annotate("QAT curve exceeds FP32 baseline\nat every N (incl. all-INT8)",
                    xy=(0, qat_psnr[0]),
                    xytext=(0.6, qat_psnr[2] + 0.008),
                    fontsize=9, color="#2a8c4a", weight="bold",
                    arrowprops=dict(arrowstyle="->", color="#2a8c4a", alpha=0.8))

    ax.set_xlabel("Number of sensitive layers kept in FP32")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Mixed Precision Sweep: PTQ vs QAT — QAT regime exceeds FP32 baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
