"""Mixed-precision evaluation: keep the N most sensitive layers in FP32.

Reads layer_analysis.csv (produced by analyze_layers.py) to rank layers by
quantization sensitivity, then sweeps over how many sensitive layers to leave
in FP32 while quantizing the rest to INT8.

Run::

    python -m src.deployment.mixed_precision --checkpoint results/runs/20260427_143542_ep200_b16_scale2_realistic/checkpoints/best.pt --sensitivity results/layer_analysis/edsr_200ep/layer_analysis.csv --output-dir  results/mixed_precision/edsr_200ep --data-root   data/DIV2K --val-dir DIV2K_valid_HR
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.models.edsr import EDSR
from src.quantization.fake_quant import (
    CalibratingConv2d,
    apply_calibration_to_all,
    set_all_modes,
    wrap_convs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint: Path, device: torch.device) -> EDSR:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    train_args = ckpt.get("args", {})
    n_feats     = (train_args.get("n_feats") if isinstance(train_args, dict) else None) or state["head.weight"].shape[0]
    n_resblocks = (train_args.get("n_resblocks") if isinstance(train_args, dict) else None) or 16
    scale       = (train_args.get("scale") if isinstance(train_args, dict) else None) or 2
    model = EDSR(scale_factor=scale, n_resblocks=n_resblocks, n_feats=n_feats)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def calibrate(
    model: EDSR,
    wrappers: dict[str, CalibratingConv2d],
    loader: DataLoader,
    n_calib: int,
    device: torch.device,
) -> None:
    set_all_modes(wrappers, "calibrate")
    with torch.no_grad():
        for i, (lr, _) in enumerate(loader):
            if i >= n_calib:
                break
            model(lr.to(device))
    apply_calibration_to_all(wrappers, method="max-abs")
    set_all_modes(wrappers, "fp32")


@torch.no_grad()
def evaluate_metrics(
    model: EDSR,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Mean (PSNR dB, SSIM) over the loader -- same definition as the rest of
    the pipeline so the mixed-precision sweep can be cross-referenced with
    fake-quant analysis and ONNX deploy benchmark."""
    model.eval()
    psnr_sum, ssim_sum, count = 0.0, 0.0, 0
    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr).clamp(0.0, 1.0)
        mse = ((sr - hr) ** 2).mean(dim=(1, 2, 3))
        psnr_sum += (10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))).sum().item()

        sr_np = sr.cpu().numpy()
        hr_np = hr.cpu().numpy()
        for i in range(sr_np.shape[0]):
            # Cast to Python float so the running sum stays JSON-serialisable
            # (skimage SSIM returns numpy scalar types).
            ssim_sum += float(ssim_fn(
                hr_np[i].transpose(1, 2, 0),
                sr_np[i].transpose(1, 2, 0),
                data_range=1.0,
                channel_axis=2,
            ))

        count += lr.shape[0]
    n = max(count, 1)
    return psnr_sum / n, ssim_sum / n


def read_sensitivity_ranking(csv_path: Path) -> list[str]:
    """Return layer names sorted by isolated_psnr ascending (most sensitive first)."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["layer"], float(row["isolated_psnr"])))
    rows.sort(key=lambda x: x[1])
    return [name for name, _ in rows]


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # Dataset
    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=2,
        hr_patch_size=192,
        degradation="realistic",
        is_train=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=False, num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Model + calibration (done once, reused across all sweep points)
    if args.qat:
        # QAT path: build raw model, wrap it FIRST so the wrapper buffers
        # (input_amax, calibrated, hist) are part of the state-dict shape,
        # then load the QAT checkpoint into the wrapped model.
        # Calibration is skipped because QAT scales are already baked in.
        print("  QAT mode: loading wrapped checkpoint, skipping calibration")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        train_args = ckpt.get("args", {}) or {}
        n_feats     = train_args.get("n_feats", 64) or 64
        n_resblocks = train_args.get("n_resblocks", 16) or 16
        scale       = train_args.get("scale", 2) or 2
        model = EDSR(scale_factor=scale, n_resblocks=n_resblocks, n_feats=n_feats).to(device)
        wrappers = wrap_convs(model)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"  QAT load: missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()
        print(f"  wrapped {len(wrappers)} Conv2d layers")
    else:
        print("  loading model and calibrating...")
        model = load_model(Path(args.checkpoint), device)
        wrappers = wrap_convs(model)
        print(f"  wrapped {len(wrappers)} Conv2d layers")

        calib_loader = DataLoader(
            val_set, batch_size=8, shuffle=False, num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        calibrate(model, wrappers, calib_loader, n_calib=args.n_calib, device=device)

    # FP32 baseline
    set_all_modes(wrappers, "fp32")
    psnr_fp32, ssim_fp32 = evaluate_metrics(model, val_loader, device)
    print(f"\n  FP32 baseline : PSNR {psnr_fp32:.4f} dB | SSIM {ssim_fp32:.4f}")

    # All-INT8 baseline
    set_all_modes(wrappers, "quantize")
    psnr_int8, ssim_int8 = evaluate_metrics(model, val_loader, device)
    print(f"  All-INT8      : PSNR {psnr_int8:.4f} dB | SSIM {ssim_int8:.4f}  "
          f"(drop {psnr_fp32 - psnr_int8:.4f} dB / "
          f"{ssim_fp32 - ssim_int8:.4f} SSIM)\n")

    # Sensitivity ranking (most sensitive first)
    ranked = read_sensitivity_ranking(Path(args.sensitivity))
    print(f"  sensitivity ranking (most → least sensitive):")
    for i, name in enumerate(ranked[:8]):
        print(f"    [{i+1}] {name}")
    if len(ranked) > 8:
        print(f"    ... ({len(ranked)} total)")

    # Sweep: keep top-N sensitive layers in FP32
    n_fp32_candidates = sorted(set(
        [0] + list(range(1, min(args.max_fp32_layers + 1, len(ranked) + 1)))
    ))

    results = []
    print()
    psnr_recovery_denom = max(psnr_fp32 - psnr_int8, 1e-8)
    ssim_recovery_denom = max(ssim_fp32 - ssim_int8, 1e-8)
    for n_fp32 in n_fp32_candidates:
        set_all_modes(wrappers, "quantize")
        fp32_layers = ranked[:n_fp32]
        for name in fp32_layers:
            if name in wrappers:
                wrappers[name].set_mode("fp32")

        psnr, ssim = evaluate_metrics(model, val_loader, device)
        drop = psnr_fp32 - psnr
        ssim_drop = ssim_fp32 - ssim
        recovered = (psnr - psnr_int8) / psnr_recovery_denom * 100
        ssim_recovered = (ssim - ssim_int8) / ssim_recovery_denom * 100

        results.append({
            "n_fp32_layers": n_fp32,
            "fp32_layers": fp32_layers,
            "psnr": round(psnr, 4),
            "drop_vs_fp32": round(drop, 4),
            "recovered_pct": round(recovered, 1),
            "ssim": round(ssim, 6),
            "ssim_drop_vs_fp32": round(ssim_drop, 6),
            "ssim_recovered_pct": round(ssim_recovered, 1),
        })

        tag = f"FP32×{n_fp32:2d}" if n_fp32 > 0 else "All-INT8"
        print(f"  {tag} | PSNR {psnr:.4f} dB (drop {drop:.4f}, "
              f"recovered {recovered:.1f}%) | "
              f"SSIM {ssim:.4f} (drop {ssim_drop:.4f}, "
              f"recovered {ssim_recovered:.1f}%)"
              + (f"  ← {fp32_layers}" if fp32_layers else ""))

    # Save CSV
    csv_path = out_dir / "mixed_precision_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_fp32_layers", "psnr", "drop_vs_fp32", "recovered_pct",
                "ssim", "ssim_drop_vs_fp32", "ssim_recovered_pct",
                "fp32_layers",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({**r, "fp32_layers": "|".join(r["fp32_layers"])})
    print(f"\n  sweep saved → {csv_path}")

    # Save JSON
    json_path = out_dir / "mixed_precision_sweep.json"
    with open(json_path, "w") as f:
        json.dump({"fp32_baseline_psnr": round(psnr_fp32, 4),
                   "int8_baseline_psnr": round(psnr_int8, 4),
                   "fp32_baseline_ssim": round(ssim_fp32, 6),
                   "int8_baseline_ssim": round(ssim_int8, 6),
                   "sweep": results}, f, indent=2)

    # Plot -- twin y-axes so PSNR (left) and SSIM (right) read cleanly on
    # the same x-axis (number of FP32 layers). Without dual axes the SSIM
    # curve gets squashed into a flat line because its absolute range is tiny.
    ns    = [r["n_fp32_layers"] for r in results]
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    line_psnr, = ax.plot(
        ns, psnrs, marker="o", linewidth=2, color="steelblue",
        label="PSNR (mixed precision)",
    )
    ax.axhline(psnr_fp32, linestyle="--", color="green", linewidth=1.2,
               label=f"PSNR FP32  {psnr_fp32:.2f} dB")
    ax.axhline(psnr_int8, linestyle="--", color="tomato", linewidth=1.2,
               label=f"PSNR All-INT8 {psnr_int8:.2f} dB")
    ax.set_xlabel("Number of sensitive layers kept in FP32")
    ax.set_ylabel("PSNR (dB)", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    line_ssim, = ax2.plot(
        ns, ssims, marker="^", linewidth=2, color="darkorange",
        label="SSIM (mixed precision)",
    )
    ax2.axhline(ssim_fp32, linestyle=":", color="green", linewidth=1.0,
                label=f"SSIM FP32  {ssim_fp32:.4f}")
    ax2.axhline(ssim_int8, linestyle=":", color="tomato", linewidth=1.0,
                label=f"SSIM All-INT8 {ssim_int8:.4f}")
    ax2.set_ylabel("SSIM", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax.set_title("Mixed Precision: accuracy vs. INT8 layer count "
                 "(PSNR + SSIM)")
    # Merge legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
              fontsize=8)
    fig.tight_layout()
    plot_path = out_dir / "mixed_precision_sweep.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  plot saved    → {plot_path}")

    # Summary
    print("\n  ── Summary ──")
    print(f"  FP32 baseline  : PSNR {psnr_fp32:.4f} dB | SSIM {ssim_fp32:.4f}")
    print(f"  All-INT8       : PSNR {psnr_int8:.4f} dB | SSIM {ssim_int8:.4f}")
    best = max(results[1:], key=lambda r: r["recovered_pct"]) if len(results) > 1 else None
    if best:
        print(f"  Best PSNR-recovery : FP32×{best['n_fp32_layers']} layers"
              f" → PSNR {best['psnr']:.4f} dB  "
              f"({best['recovered_pct']:.1f}% recovered)  | "
              f"SSIM {best['ssim']:.4f} "
              f"({best['ssim_recovered_pct']:.1f}% recovered)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed-precision PSNR sweep")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--sensitivity", required=True,
                   help="layer_analysis.csv from analyze_layers.py")
    p.add_argument("--output-dir",  default="results/mixed_precision/edsr_200ep")
    p.add_argument("--data-root",   default="data/DIV2K")
    p.add_argument("--val-dir",     default="DIV2K_valid_HR")
    p.add_argument("--n-calib",     type=int, default=20)
    p.add_argument("--max-fp32-layers", type=int, default=8,
                   help="sweep from 0 to this many sensitive layers in FP32")
    p.add_argument("--qat", action="store_true",
                   help="Treat --checkpoint as QAT-trained (already wrapped, "
                        "calibration scales baked in). Skips fresh calibration.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
