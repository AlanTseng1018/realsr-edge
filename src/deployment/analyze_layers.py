"""Per-layer quantization precision analysis — from PyTorch checkpoint.

Computes per-layer quantization statistics directly from best.pt using
fake-quantization (CalibratingConv2d). No ONNX files required.

Three analyses
--------------
1. **Weight stats** — per-channel INT8 scale and dynamic range for every Conv2d,
   read straight from the checkpoint weights.
2. **Activation calibration** — activation amax per layer, calibrated by running
   DIV2K validation patches through the FP32 model.
3. **Isolated sensitivity** — for each Conv2d, quantize only that layer (keep
   all others FP32), measure the PSNR drop vs pure FP32. Higher drop = that
   layer is more sensitive to INT8 quantization.

Outputs
-------
  layer_analysis.md   — per-layer precision table (Markdown)
  layer_analysis.csv  — same data, machine-readable
  weight_scales.png   — per-channel weight scale bar chart
  activation_amaxes.png — calibrated activation amax bar chart
  sensitivity.png     — isolated sensitivity (PSNR drop per layer)
  summary.json        — key numbers for downstream tools

Run::

    python -m src.deployment.analyze_layers \\
        --checkpoint results/runs/.../checkpoints/best.pt \\
        --output-dir results/layer_analysis/edsr_200ep \\
        --data-root  data/DIV2K --val-dir DIV2K_valid_HR
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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.models.edsr import EDSR
from src.quantization.fake_quant import (
    CalibratingConv2d,
    apply_calibration_to_all,
    per_channel_scale,
    set_all_modes,
    wrap_convs,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: Path, device: torch.device) -> EDSR:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state      = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    train_args = ckpt.get("args", {})

    # Infer arch from checkpoint or args
    n_feats     = (train_args.get("n_feats") if isinstance(train_args, dict) else None) \
                  or state["head.weight"].shape[0]
    n_resblocks = (train_args.get("n_resblocks") if isinstance(train_args, dict) else None) \
                  or 16
    scale       = (train_args.get("scale") if isinstance(train_args, dict) else None) \
                  or 2

    model = EDSR(scale_factor=scale, n_resblocks=n_resblocks, n_feats=n_feats)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Part 1: Weight stats from checkpoint
# ---------------------------------------------------------------------------

def extract_weight_stats(wrappers: dict[str, CalibratingConv2d]) -> list[dict]:
    """Return per-layer weight statistics from the wrapped Conv2d weights."""
    rows = []
    for name, wrapper in wrappers.items():
        w = wrapper.conv.weight.detach().cpu()          # (out_ch, in_ch, kH, kW)
        w_scales = per_channel_scale(w, ch_axis=0)      # (out_ch,)

        rows.append({
            "layer":          name,
            "shape":          list(w.shape),
            "out_ch":         w.shape[0],
            "in_ch":          w.shape[1],
            "kernel":         w.shape[2],
            "w_amax":         float(w.abs().amax()),
            "w_scale_mean":   float(w_scales.mean()),
            "w_scale_std":    float(w_scales.std()),
            "w_scale_max":    float(w_scales.max()),
            "w_dynamic_range": float(w_scales.max() * 127),  # max representable
        })
    return rows


# ---------------------------------------------------------------------------
# Part 2: Activation calibration
# ---------------------------------------------------------------------------

def calibrate(
    model: EDSR,
    wrappers: dict[str, CalibratingConv2d],
    loader: DataLoader,
    n_calib: int,
    device: torch.device,
) -> None:
    """Run a calibration pass over n_calib images; updates wrapper.input_amax."""
    set_all_modes(wrappers, "calibrate")
    with torch.no_grad():
        for i, (lr, _) in enumerate(loader):
            if i >= n_calib:
                break
            model(lr.to(device))
    apply_calibration_to_all(wrappers, method="max-abs")
    set_all_modes(wrappers, "fp32")


def collect_activation_stats(wrappers: dict[str, CalibratingConv2d]) -> dict[str, dict]:
    """Return {layer_name: {act_amax, act_scale}} after calibration."""
    out = {}
    for name, wrapper in wrappers.items():
        amax = float(wrapper.input_amax)
        out[name] = {
            "act_amax":  amax,
            "act_scale": amax / 127.0,
        }
    return out


# ---------------------------------------------------------------------------
# Part 3: Isolated sensitivity
# ---------------------------------------------------------------------------

def _psnr_fp32_vs_output(
    fp32_outputs: list[torch.Tensor],
    quant_outputs: list[torch.Tensor],
) -> float:
    """PSNR between paired FP32 and quantized model outputs (range [0, 1])."""
    mse_sum = 0.0
    n = 0
    for fp32, quant in zip(fp32_outputs, quant_outputs):
        diff = (fp32.float() - quant.float())
        mse_sum += float(diff.pow(2).mean())
        n += 1
    mse = mse_sum / max(n, 1)
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def _collect_outputs(
    model: EDSR,
    loader: DataLoader,
    n_samples: int,
    device: torch.device,
) -> list[torch.Tensor]:
    outputs = []
    with torch.no_grad():
        for i, (lr, _) in enumerate(loader):
            if i >= n_samples:
                break
            outputs.append(model(lr.to(device)).cpu())
    return outputs


def compute_isolated_sensitivity(
    model: EDSR,
    wrappers: dict[str, CalibratingConv2d],
    loader: DataLoader,
    n_samples: int,
    device: torch.device,
) -> dict[str, float]:
    """For each Conv2d, quantize only that layer and measure PSNR vs FP32 output.

    Returns {layer_name: psnr_dB}. Lower PSNR = more sensitive layer.
    """
    set_all_modes(wrappers, "fp32")
    fp32_outputs = _collect_outputs(model, loader, n_samples, device)

    sensitivity: dict[str, float] = {}
    layer_names = list(wrappers.keys())
    for i, name in enumerate(layer_names):
        print(f"    [{i+1:02d}/{len(layer_names)}] {name} ...", end="\r")
        set_all_modes(wrappers, "fp32")
        wrappers[name].set_mode("quantize")
        quant_outputs = _collect_outputs(model, loader, n_samples, device)
        sensitivity[name] = _psnr_fp32_vs_output(fp32_outputs, quant_outputs)

    set_all_modes(wrappers, "fp32")
    print()
    return sensitivity


def compute_e2e_fake_quant_psnr(
    model: EDSR,
    wrappers: dict[str, CalibratingConv2d],
    loader: DataLoader,
    n_samples: int,
    device: torch.device,
) -> float:
    """PSNR between FP32 and all-layers-quantized outputs."""
    set_all_modes(wrappers, "fp32")
    fp32_outputs = _collect_outputs(model, loader, n_samples, device)
    set_all_modes(wrappers, "quantize")
    int8_outputs = _collect_outputs(model, loader, n_samples, device)
    set_all_modes(wrappers, "fp32")
    return _psnr_fp32_vs_output(fp32_outputs, int8_outputs)


# ---------------------------------------------------------------------------
# Merge into per-layer table
# ---------------------------------------------------------------------------

def build_layer_table(
    weight_rows: list[dict],
    act_stats:   dict[str, dict],
    sensitivity: dict[str, float],
) -> list[dict]:
    """Merge weight stats, activation stats, and sensitivity into one table."""
    rows = []
    for r in weight_rows:
        name = r["layer"]
        act  = act_stats.get(name, {})
        sens = sensitivity.get(name, float("nan"))
        rows.append({
            "layer":          name,
            "shape":          r["shape"],
            "out_ch":         r["out_ch"],
            "w_amax":         r["w_amax"],
            "w_scale_mean":   r["w_scale_mean"],
            "w_scale_max":    r["w_scale_max"],
            "w_dynamic_range": r["w_dynamic_range"],
            "act_amax":       act.get("act_amax", float("nan")),
            "act_scale":      act.get("act_scale", float("nan")),
            "isolated_psnr":  sens,
        })
    # Sort by sensitivity ascending (most sensitive = lowest PSNR = quantize hurts most)
    rows.sort(key=lambda r: r["isolated_psnr"])
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_weight_scales(rows: list[dict], output_path: Path) -> None:
    labels = [r["layer"] for r in rows]
    means  = [r["w_scale_mean"] for r in rows]
    maxs   = [r["w_scale_max"]  for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Per-Layer Weight Quantization Scale (INT8, from best.pt)",
                 fontsize=13, fontweight="bold")

    for ax, vals, title, color in [
        (axes[0], means, "Weight scale — mean across output channels", "#4c8cbf"),
        (axes[1], maxs,  "Weight scale — max channel (determines clipping)", "#c0392b"),
    ]:
        x = range(len(labels))
        ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6.5)
        ax.set_ylabel("Scale")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {output_path.name}")


def plot_activation_amaxes(rows: list[dict], output_path: Path) -> None:
    labels = [r["layer"] for r in rows]
    amaxes = [r["act_amax"] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(labels))
    ax.bar(x, amaxes, color="#2ca02c", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6.5)
    ax.set_ylabel("Activation amax (max-abs calibration)")
    ax.set_title("Per-Layer Activation Dynamic Range (calibrated on DIV2K val)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {output_path.name}")


def plot_sensitivity(rows: list[dict], e2e_psnr: float, output_path: Path) -> None:
    """Bar chart of isolated PSNR (lower bar = more sensitive layer)."""
    # Sort by PSNR ascending so most sensitive is at top
    sorted_rows = sorted(rows, key=lambda r: r["isolated_psnr"])
    labels = [r["layer"] for r in sorted_rows]
    psnrs  = [r["isolated_psnr"] for r in sorted_rows]

    # Colour: below 60 dB = high sensitivity (red), else blue
    colors = ["#c0392b" if p < 60 else "#4c8cbf" for p in psnrs]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.28)))
    ax.barh(range(len(labels)), psnrs, color=colors, alpha=0.85)
    ax.axvline(e2e_psnr, color="#e07b2a", lw=1.5, ls="--",
               label=f"E2E fake-INT8 PSNR = {e2e_psnr:.1f} dB")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("PSNR (dB)  [FP32 output vs isolated-quantize output]\n"
                  "Lower = more sensitive to INT8 quantization")
    ax.set_title("Isolated Layer Sensitivity\n"
                 "(quantize one Conv2d at a time; rest stay FP32)\n"
                 "Red bars: PSNR < 60 dB (high sensitivity)")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {output_path.name}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_md_report(
    rows: list[dict],
    e2e_psnr: float,
    checkpoint: Path,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Per-Layer Quantization Precision Analysis\n\n")
        f.write(f"- **Source**: `{checkpoint}`\n")
        f.write(f"- **Generated**: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- **E2E fake-INT8 PSNR** (FP32 vs all-layers-quantized): "
                f"**{e2e_psnr:.2f} dB**\n\n")

        f.write("> Table sorted by **isolated PSNR** ascending — "
                "most sensitive layers appear first.\n")
        f.write("> Isolated PSNR: PSNR between FP32 output and output when "
                "*only this layer* is INT8 fake-quantized.\n\n")

        f.write("| # | Layer | Shape | W amax | W scale (mean) | "
                "Act amax | Act scale | Isolated PSNR (dB) |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for i, r in enumerate(rows, 1):
            shape_str = "×".join(str(d) for d in r["shape"])
            psnr_str  = f"**{r['isolated_psnr']:.1f}**" \
                        if r["isolated_psnr"] < 60 \
                        else f"{r['isolated_psnr']:.1f}"
            f.write(
                f"| {i} | `{r['layer']}` | {shape_str} | "
                f"{r['w_amax']:.4e} | {r['w_scale_mean']:.4e} | "
                f"{r['act_amax']:.4e} | {r['act_scale']:.4e} | "
                f"{psnr_str} |\n"
            )
        f.write("\n")

        f.write("## Glossary\n\n")
        f.write("| Term | Definition |\n|---|---|\n")
        f.write("| W amax | max(abs(weight)) — full FP32 dynamic range |\n")
        f.write("| W scale (mean) | mean per-channel INT8 scale = amax_per_ch / 127 |\n")
        f.write("| Act amax | max-abs activation seen during calibration (DIV2K val) |\n")
        f.write("| Act scale | activation amax / 127 — the INT8 quantization step |\n")
        f.write("| Isolated PSNR | PSNR(FP32 output, output with only this layer quantized) |\n")


def write_csv(rows: list[dict], e2e_psnr: float, output_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "layer", "out_ch", "w_amax", "w_scale_mean", "w_scale_max",
        "w_dynamic_range", "act_amax", "act_scale", "isolated_psnr",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-layer quantization analysis from best.pt")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to best.pt checkpoint")
    p.add_argument("--output-dir",  type=str, required=True)
    p.add_argument("--data-root",   type=str, default="data/DIV2K")
    p.add_argument("--val-dir",     type=str, default="DIV2K_valid_HR")
    p.add_argument("--scale",       type=int, default=2)
    p.add_argument("--patch-size",  type=int, default=96,
                   help="LR patch size for calibration/sensitivity passes")
    p.add_argument("--n-calib",     type=int, default=20,
                   help="Images used for activation calibration")
    p.add_argument("--n-sensitivity", type=int, default=5,
                   help="Images used per isolated sensitivity measurement")
    p.add_argument("--device",      type=str, default="cuda",
                   choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out     = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt    = Path(args.checkpoint)

    print("=" * 60)
    print("  Per-Layer Quantization Precision Analysis (from best.pt)")
    print("=" * 60)
    print(f"  checkpoint : {ckpt}")
    print(f"  device     : {device}")

    # Load model
    print("\n[1/5] Loading model ...")
    model = load_model(ckpt, device)
    wrappers = wrap_convs(model)
    print(f"  Conv2d layers found: {len(wrappers)}")
    for name, w in list(wrappers.items())[:3]:
        print(f"    {name}: {list(w.conv.weight.shape)}")
    print("    ...")

    # Build data loader (shared for calibration + sensitivity)
    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation="realistic",
        is_train=False,
    )
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # Weight stats (no data needed — straight from checkpoint)
    print("\n[2/5] Extracting weight stats from checkpoint ...")
    weight_rows = extract_weight_stats(wrappers)

    # Calibration
    print(f"\n[3/5] Calibrating activations ({args.n_calib} images) ...")
    calibrate(model, wrappers, loader, args.n_calib, device)
    act_stats = collect_activation_stats(wrappers)
    amax_vals = [v["act_amax"] for v in act_stats.values()]
    print(f"  act_amax range: [{min(amax_vals):.4e}, {max(amax_vals):.4e}]")

    # Isolated sensitivity
    print(f"\n[4/5] Isolated sensitivity ({args.n_sensitivity} images × "
          f"{len(wrappers)} layers) ...")
    sensitivity = compute_isolated_sensitivity(
        model, wrappers, loader, args.n_sensitivity, device
    )

    # E2E fake-INT8 PSNR
    print("\n[5/5] End-to-end fake-INT8 PSNR ...")
    e2e_psnr = compute_e2e_fake_quant_psnr(
        model, wrappers, loader, args.n_sensitivity, device
    )
    print(f"  E2E fake-INT8 PSNR (vs FP32): {e2e_psnr:.2f} dB")

    # Merge table and write outputs
    layer_table = build_layer_table(weight_rows, act_stats, sensitivity)

    print("\nWriting outputs ...")
    plot_weight_scales(layer_table, out / "weight_scales.png")
    plot_activation_amaxes(layer_table, out / "activation_amaxes.png")
    plot_sensitivity(layer_table, e2e_psnr, out / "sensitivity.png")
    write_md_report(layer_table, e2e_psnr, ckpt, out / "layer_analysis.md")
    write_csv(layer_table, e2e_psnr, out / "layer_analysis.csv")

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "datetime":       datetime.datetime.now().isoformat(timespec="seconds"),
            "checkpoint":     str(ckpt),
            "n_conv_layers":  len(wrappers),
            "e2e_fake_int8_psnr_db": e2e_psnr,
            "most_sensitive_layers": [
                {"layer": r["layer"], "isolated_psnr_db": r["isolated_psnr"]}
                for r in layer_table[:5]
            ],
        }, f, indent=2)

    print(f"\n  layer_analysis.md     -> {out / 'layer_analysis.md'}")
    print(f"  layer_analysis.csv    -> {out / 'layer_analysis.csv'}")
    print(f"  weight_scales.png     -> {out / 'weight_scales.png'}")
    print(f"  activation_amaxes.png -> {out / 'activation_amaxes.png'}")
    print(f"  sensitivity.png       -> {out / 'sensitivity.png'}")
    print(f"\n{'=' * 60}")
    print(f"  E2E fake-INT8 PSNR (FP32 vs all quantized): {e2e_psnr:.2f} dB")
    most_sensitive = layer_table[0]
    print(f"  Most sensitive layer : {most_sensitive['layer']} "
          f"({most_sensitive['isolated_psnr']:.1f} dB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
