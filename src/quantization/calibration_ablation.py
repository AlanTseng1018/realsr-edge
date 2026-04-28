"""Calibration-method ablation: max-abs vs percentile (99 / 99.9 / 99.99).

The same calibration pass collects max-abs running stats AND a histogram of
``|x|``. We then sweep the choice of "what amax to use" across four schemes,
re-running the val-set PSNR for each. No re-training, no extra calibration
pass -- it's just "use a different summary of the same data".

Outputs (under ``--output-dir``):

* ``calibration_ablation.md``  -- full report with all four schemes
* ``ablation.csv``             -- machine-readable shootout numbers
* ``per_layer_amax.csv``       -- (layer x scheme -> chosen amax)
* ``histograms.png``           -- 2x3 grid of representative layers,
                                  showing the activation magnitude
                                  histogram and where each scheme's amax
                                  lands. The visual evidence for *why*
                                  one scheme picks a different cutoff
                                  than another.

Run::

    python -m src.quantization.calibration_ablation \\
        --checkpoint results/runs/.../checkpoints/best.pt \\
        --output-dir results/quantization/calibration_ablation
"""

from __future__ import annotations

import argparse
import csv
import datetime
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.models.edsr import EDSR
from src.quantization.analyze import (
    benchmark_latency,
    calibrate_int8,
    evaluate_psnr,
)
from src.quantization.fake_quant import (
    CalibratingConv2d,
    apply_calibration_to_all,
    set_all_modes,
    wrap_convs,
)


# ---------------------------------------------------------------------------
# Schemes under test
# ---------------------------------------------------------------------------

# (label, method, percentile)  --  percentile is ignored for 'max-abs'
SCHEMES: list[tuple[str, str, float | None]] = [
    ("max-abs",          "max-abs",    None),
    ("percentile-99.99", "percentile", 0.9999),
    ("percentile-99.9",  "percentile", 0.999),
    ("percentile-99.0",  "percentile", 0.99),
]

# Color per scheme for the histogram plot
SCHEME_COLORS: dict[str, str] = {
    "max-abs":          "tab:red",
    "percentile-99.99": "tab:orange",
    "percentile-99.9":  "tab:green",
    "percentile-99.0":  "tab:blue",
}


# Representative layers for visualization. We pick a spread:
#   - quant-critical: head, tail, upsampler.0
#   - quant-robust:   body.0.conv1, body.8.conv1, body.15.conv2
# This is hard-coded for EDSR-baseline (16 ResBlocks). If the architecture
# changes, this list should change too.
REPRESENTATIVE_LAYERS: list[str] = [
    "head",
    "upsampler.0",
    "tail",
    "body.0.conv1",
    "body.8.conv1",
    "body.15.conv2",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_amax_per_layer(
    wrappers: dict[str, CalibratingConv2d],
) -> dict[str, float]:
    """Snapshot ``input_amax`` per layer (call after ``apply_calibration_to_all``)."""
    return {name: w.input_amax.item() for name, w in wrappers.items()}


def write_per_layer_amax_csv(
    path: Path,
    amax_per_scheme: dict[str, dict[str, float]],
    layer_order: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + list(amax_per_scheme.keys()))
        for layer in layer_order:
            row: list[str | float] = [layer]
            for scheme in amax_per_scheme:
                row.append(amax_per_scheme[scheme].get(layer, float("nan")))
            writer.writerow(row)


def write_ablation_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_ablation_md(
    path: Path,
    rows: list[dict],
    amax_per_scheme: dict[str, dict[str, float]],
    layer_order: list[str],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Calibration Method Ablation\n\n")

        # Metadata block
        f.write("## What was tested\n\n")
        f.write(f"- **Generated**: {metadata['datetime']}\n")
        f.write(f"- **Checkpoint**: `{metadata['checkpoint_path']}`\n")
        f.write(f"  - last modified: {metadata['checkpoint_mtime']}, "
                f"size: {metadata['checkpoint_size_mb']:.2f} MB\n")
        f.write(f"- **Model**: {metadata['model_arch']} "
                f"-- {metadata['model_params']:,} params\n")
        f.write(f"- **Device**: {metadata['device']} ({metadata['device_name']}), "
                f"PyTorch {metadata['torch_version']}\n")
        f.write(f"- **Validation set**: `{metadata['val_set_dir']}` -- "
                f"{metadata['val_set_size']} images, realistic degradation, "
                f"LR patch {metadata['val_lr_patch']}x{metadata['val_lr_patch']}\n")
        f.write(f"- **Calibration**: {metadata['calib_batches']} batches "
                f"({metadata['calib_samples']} LR samples)\n")
        f.write(f"- **Histogram bins**: {metadata['n_bins']}\n")
        f.write(f"- **Quantization scheme**: symmetric per-tensor INT8 (activations) "
                f"+ symmetric per-channel INT8 (weights)\n\n")

        # Shootout
        f.write("## Calibration scheme shootout\n\n")
        f.write("All four schemes share the **same calibration pass** -- the histogram "
                "is collected once, and each scheme just chooses a different summary of "
                "it (running max for `max-abs`, percentile cutoff for the other three).\n\n")
        f.write("| Scheme | PSNR (dB) | Drop vs FP32 | Latency (ms) |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows:
            lat = f"{r['latency_ms_mean']:.2f} +/- {r['latency_ms_std']:.2f}"
            f.write(f"| {r['scheme']} | {r['psnr_db']:.3f} | "
                    f"{r['psnr_drop_db']:+.3f} | {lat} |\n")
        f.write("\n")

        # Per-layer amax table
        f.write("## Per-layer chosen `amax` per scheme\n\n")
        f.write("This is the value that drives `scale = amax / 127` for each layer's "
                "input quantizer. Smaller values = tighter clipping = the tail of the "
                "activation distribution gets saturated.\n\n")
        f.write("| Layer | " + " | ".join(amax_per_scheme.keys()) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(amax_per_scheme)) + "|\n")
        for layer in layer_order:
            cells = []
            for scheme in amax_per_scheme:
                v = amax_per_scheme[scheme].get(layer, float("nan"))
                cells.append(f"{v:.4f}")
            f.write(f"| `{layer}` | " + " | ".join(cells) + " |\n")
        f.write("\n")

        # Reading guide
        f.write("## How to read the histogram figure\n\n")
        f.write("`histograms.png` plots the activation-magnitude histogram (collected "
                "during calibration) for six representative layers. Vertical dashed "
                "lines mark where each scheme places its `amax`:\n\n")
        f.write("- **Red (max-abs)**: at the largest `|x|` ever seen. Most conservative "
                "(no clipping) but exposed to single outliers.\n")
        f.write("- **Orange (99.99)**: cut off the very last 0.01% of the tail.\n")
        f.write("- **Green (99.9)**: typical TensorRT-style aggressive choice.\n")
        f.write("- **Blue (99.0)**: aggressive clipping; saturates more than just outliers.\n\n")
        f.write("Y-axis is log-scale because activation distributions are heavily "
                "long-tailed -- a linear axis would render the tail invisible. The "
                "amax differences look small numerically, but the resulting INT8 "
                "scale (`amax / 127`) is what determines the bin resolution for the "
                "bulk of values, which is what shows up in the PSNR table above.\n\n")

        # Reading guide for the table
        best_psnr = max(r["psnr_db"] for r in rows)
        best_scheme = next(r["scheme"] for r in rows if r["psnr_db"] == best_psnr)
        f.write("## Takeaway\n\n")
        f.write(f"On this checkpoint and val set, **`{best_scheme}`** wins the shootout "
                f"({best_psnr:.3f} dB). The spread across schemes is "
                f"{(max(r['psnr_db'] for r in rows) - min(r['psnr_db'] for r in rows)):.3f} dB.\n\n")
        f.write("Interpret carefully:\n\n")
        f.write("- A small spread (< 0.05 dB) means activations are NOT outlier-heavy "
                "for this model -- the calibration choice is mostly cosmetic.\n")
        f.write("- A large spread (> 0.2 dB) means there ARE outliers that max-abs is "
                "exposed to. In that case, percentile clipping is a real win.\n")
        f.write("- If `percentile-99.0` is best, the tail isn't useful; consider an "
                "even tighter cutoff or a learned clipping threshold (PAMS-style).\n")
        f.write("- If `max-abs` is best, the tail IS informative; percentile clipping "
                "is throwing away useful signal.\n\n")
        f.write("All four numbers are produced from the same FP32 weights and the same "
                "calibration histogram. The infrastructure is in place to add KL-div "
                "or MSE-optimal calibration as additional schemes by adding entries "
                "to `SCHEMES` in `calibration_ablation.py`.\n")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_histograms(
    wrappers: dict[str, CalibratingConv2d],
    layer_names: list[str],
    amax_per_scheme: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """2x3 subplot: activation histogram + amax markers per scheme.

    Y-axis is log to make long tails legible.
    """
    matplotlib.use("Agg")  # headless
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, layer_name in zip(axes.ravel(), layer_names):
        if layer_name not in wrappers:
            ax.set_title(f"{layer_name} (not found)")
            ax.axis("off")
            continue

        w = wrappers[layer_name]
        hist = w.hist.detach().cpu().numpy()
        hist_max = w.hist_max.item()
        n_bins = len(hist)

        if hist_max <= 0 or hist.sum() <= 0:
            ax.set_title(f"{layer_name} (no data)")
            ax.axis("off")
            continue

        bin_edges = np.linspace(0.0, hist_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = hist_max / n_bins

        # Replace zeros with NaN so log-scale doesn't render empty bins
        hist_for_plot = np.where(hist > 0, hist, np.nan)
        ax.bar(
            bin_centers, hist_for_plot, width=bin_width * 0.95,
            color="lightgray", edgecolor="none", label="|x| histogram",
        )
        ax.set_yscale("log")

        # Vertical lines per scheme
        for scheme_name, amaxes in amax_per_scheme.items():
            v = amaxes.get(layer_name)
            if v is None:
                continue
            color = SCHEME_COLORS.get(scheme_name, "black")
            ax.axvline(
                v, color=color, linestyle="--", linewidth=1.5,
                label=f"{scheme_name}: {v:.3f}",
            )

        ax.set_title(layer_name, fontsize=11)
        ax.set_xlabel("|x|", fontsize=10)
        ax.set_ylabel("count (log)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Activation magnitude histogram + chosen amax per calibration scheme",
        fontsize=14,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibration-method ablation (max-abs vs percentile)"
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument(
        "--output-dir", type=str,
        default="results/quantization/calibration_ablation",
    )
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--n-bins", type=int, default=2048,
                   help="Histogram resolution (TensorRT default: 2048)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Calibration Method Ablation")
    print("=" * 60)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {device}")
    print(f"  output     : {output_dir}")
    print(f"  schemes    : {[s[0] for s in SCHEMES]}")
    print()

    # --- Data ---
    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation="realistic",
        is_train=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    calib_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  val: {len(val_set)} images")

    # --- Models (FP32 baseline + wrapped quant copy) ---
    def build_model() -> torch.nn.Module:
        m = EDSR(
            scale_factor=args.scale,
            n_resblocks=args.n_resblocks,
            n_feats=args.n_feats,
        )
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model"])
        return m

    fp32_model = build_model().to(device).eval()
    quant_model = build_model().to(device).eval()
    n_params = sum(p.numel() for p in fp32_model.parameters())

    # Patch wrap_convs to use the desired n_bins
    wrappers: dict[str, CalibratingConv2d] = {}
    pairs = []
    for parent_name, parent in quant_model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, torch.nn.Conv2d) and not isinstance(child, CalibratingConv2d):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                pairs.append((parent, child_name, child, full))
    for parent, child_name, child, full in pairs:
        w = CalibratingConv2d(child, n_bins=args.n_bins)
        setattr(parent, child_name, w)
        wrappers[full] = w
    print(f"  wrapped {len(wrappers)} Conv2d layers ({args.n_bins} bins each)")
    print()

    # --- Calibrate (collects max-abs AND histogram) ---
    print("Calibrating ...")
    calibrate_int8(quant_model, wrappers, calib_loader, device,
                   n_batches=args.calib_batches)
    print()

    # --- FP32 baseline PSNR ---
    print("FP32 baseline PSNR ...")
    fp32_psnr = evaluate_psnr(fp32_model, val_loader, device)
    print(f"  FP32 = {fp32_psnr:.3f} dB")
    print()

    # --- Sweep schemes ---
    print("Running scheme shootout ...")
    bench_input = torch.randn(1, 3, args.patch_size, args.patch_size, device=device)
    rows: list[dict] = []
    amax_per_scheme: dict[str, dict[str, float]] = {}

    for label, method, percentile in SCHEMES:
        apply_calibration_to_all(
            wrappers,
            method=method,
            percentile=percentile if percentile is not None else 0.999,
        )
        amax_per_scheme[label] = collect_amax_per_layer(wrappers)

        set_all_modes(wrappers, "quantize")
        psnr = evaluate_psnr(quant_model, val_loader, device)
        lat_mean, lat_std = benchmark_latency(quant_model, bench_input)
        set_all_modes(wrappers, "fp32")

        rows.append({
            "scheme": label,
            "method": method,
            "percentile": "" if percentile is None else f"{percentile:.4f}",
            "psnr_db": psnr,
            "psnr_drop_db": fp32_psnr - psnr,
            "latency_ms_mean": lat_mean,
            "latency_ms_std": lat_std,
        })
        print(f"  [{label:18s}] PSNR {psnr:.3f} dB  "
              f"(drop {fp32_psnr - psnr:+.3f})  "
              f"latency {lat_mean:.2f} +/- {lat_std:.2f} ms")

    # --- Restore to default max-abs ---
    apply_calibration_to_all(wrappers, method="max-abs")

    # --- Write outputs ---
    print()
    print("Writing outputs ...")

    layer_order = list(wrappers.keys())
    write_ablation_csv(output_dir / "ablation.csv", rows)
    write_per_layer_amax_csv(
        output_dir / "per_layer_amax.csv", amax_per_scheme, layer_order,
    )

    # Histogram plot for representative layers
    plot_histograms(
        wrappers, REPRESENTATIVE_LAYERS, amax_per_scheme,
        output_dir / "histograms.png",
    )

    # Metadata for the report
    ckpt_path = Path(args.checkpoint)
    metadata = {
        "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_mtime": datetime.datetime.fromtimestamp(
            ckpt_path.stat().st_mtime
        ).isoformat(timespec="seconds"),
        "checkpoint_size_mb": ckpt_path.stat().st_size / (1024 * 1024),
        "model_arch": (
            f"EDSR(scale_factor={args.scale}, "
            f"n_resblocks={args.n_resblocks}, n_feats={args.n_feats})"
        ),
        "model_params": n_params,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda"
            else (platform.processor() or "CPU")
        ),
        "torch_version": torch.__version__,
        "val_set_dir": str(Path(args.data_root) / args.val_dir),
        "val_set_size": len(val_set),
        "val_lr_patch": args.patch_size,
        "calib_batches": args.calib_batches,
        "calib_samples": args.calib_batches * args.batch_size,
        "n_bins": args.n_bins,
    }
    write_ablation_md(
        output_dir / "calibration_ablation.md",
        rows=rows,
        amax_per_scheme=amax_per_scheme,
        layer_order=layer_order,
        metadata=metadata,
    )

    print(f"  ablation.csv               -> {output_dir / 'ablation.csv'}")
    print(f"  per_layer_amax.csv         -> {output_dir / 'per_layer_amax.csv'}")
    print(f"  histograms.png             -> {output_dir / 'histograms.png'}")
    print(f"  calibration_ablation.md    -> {output_dir / 'calibration_ablation.md'}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
