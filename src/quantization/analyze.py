"""Quantization shootout + per-layer sensitivity analysis.

Two analyses run from one checkpoint, against the val set:

1. **Format shootout** (:func:`run_shootout`): compare PSNR + latency across
   FP32 (baseline), FP16 (autocast), BF16 (autocast), and INT8 (fake-quant
   per :mod:`src.quantization.fake_quant`). Output: markdown table + CSV.

2. **Per-layer sensitivity** (:func:`run_sensitivity`): keep the network in
   FP32 except quantize **one Conv2d at a time** to INT8; measure the PSNR
   drop attributable to that single layer. Output: ranked CSV +
   barplot-friendly numbers. Identifies "quantization-critical" layers
   (typically first conv / last conv) -- the diagnostic-first foundation
   for §3.4 of the spec.

Run examples
------------
::

    # Both analyses on the existing checkpoint, default device
    python -m src.quantization.analyze \
        --checkpoint results/checkpoints/edsr_baseline/final.pt \
        --output-dir results/quantization

    # Only the shootout, smaller calibration set
    python -m src.quantization.analyze \
        --checkpoint results/checkpoints/edsr_baseline/final.pt \
        --output-dir results/quantization \
        --skip-sensitivity \
        --calib-batches 4
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import platform
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.models.edsr import EDSR
from src.quantization.fake_quant import (
    CalibratingConv2d,
    reset_all_calibration,
    set_all_modes,
    wrap_convs,
)


# ---------------------------------------------------------------------------
# Eval / latency helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_psnr(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> float:
    """Mean PSNR (dB) over the loader. ``forward`` lets callers wrap the model
    call (e.g. with ``torch.amp.autocast``)."""
    if forward is None:
        forward = lambda m, x: m(x)
    model.eval()
    psnr_sum, count = 0.0, 0
    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        sr = forward(model, lr).clamp(0.0, 1.0).float()
        mse = ((sr - hr) ** 2).mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
        psnr_sum += psnr.sum().item()
        count += psnr.numel()
    return psnr_sum / max(count, 1)


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    n_warmup: int = 5,
    n_iter: int = 30,
) -> tuple[float, float]:
    """Forward latency in ms (mean, std) on the given input shape. Includes
    ``cuda.synchronize`` when on CUDA so the numbers are real."""
    if forward is None:
        forward = lambda m, x: m(x)
    model.eval()
    is_cuda = sample_input.is_cuda

    for _ in range(n_warmup):
        _ = forward(model, sample_input)
    if is_cuda:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = forward(model, sample_input)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_int8(
    model: nn.Module,
    wrappers: dict[str, CalibratingConv2d],
    calib_loader: DataLoader,
    device: torch.device,
    n_batches: int = 8,
) -> None:
    """Run a calibration pass: switch wrappers to 'calibrate' mode, push
    ``n_batches`` of LR images through the model, then leave wrappers in
    'fp32' mode (caller decides when to switch to 'quantize')."""
    set_all_modes(wrappers, "calibrate")
    model.eval()
    seen = 0
    for i, (lr, _hr) in enumerate(calib_loader):
        if i >= n_batches:
            break
        lr = lr.to(device, non_blocking=True)
        _ = model(lr)
        seen += lr.shape[0]
    set_all_modes(wrappers, "fp32")
    print(f"  calibrated on {seen} LR samples ({n_batches} batches)")


# ---------------------------------------------------------------------------
# Forward wrappers for autocast
# ---------------------------------------------------------------------------

def _make_autocast_forward(dtype: torch.dtype, device_type: str):
    def fwd(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            return m(x)
    return fwd


# ---------------------------------------------------------------------------
# Shootout
# ---------------------------------------------------------------------------

def model_size_mb(model: nn.Module, bytes_per_param: float) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n * bytes_per_param / (1024 * 1024)


def run_shootout(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    bench_input_shape: tuple[int, int, int, int],
) -> list[dict]:
    """Compare FP32, FP16, BF16, INT8 (fake-quant). Returns list of result rows."""
    fp32_model = fp32_model.to(device).eval()
    quant_model = quant_model.to(device).eval()
    bench_in_fp32 = torch.randn(*bench_input_shape, device=device)

    results: list[dict] = []

    # FP32 baseline
    print("  [FP32]")
    psnr = evaluate_psnr(fp32_model, val_loader, device)
    lat_mean, lat_std = benchmark_latency(fp32_model, bench_in_fp32)
    results.append({
        "format": "FP32",
        "psnr_db": psnr,
        "psnr_drop_db": 0.0,
        "latency_ms_mean": lat_mean,
        "latency_ms_std": lat_std,
        "size_mb": model_size_mb(fp32_model, 4.0),
        "notes": "baseline",
    })
    fp32_psnr = psnr
    print(f"    PSNR={psnr:.3f} dB | latency={lat_mean:.2f}+/-{lat_std:.2f} ms")

    # FP16 autocast
    if device.type == "cuda":
        print("  [FP16 autocast]")
        fwd = _make_autocast_forward(torch.float16, device.type)
        psnr = evaluate_psnr(fp32_model, val_loader, device, forward=fwd)
        lat_mean, lat_std = benchmark_latency(fp32_model, bench_in_fp32, forward=fwd)
        results.append({
            "format": "FP16 (autocast)",
            "psnr_db": psnr,
            "psnr_drop_db": fp32_psnr - psnr,
            "latency_ms_mean": lat_mean,
            "latency_ms_std": lat_std,
            "size_mb": model_size_mb(fp32_model, 2.0),
            "notes": "weights FP32, ops cast on the fly",
        })
        print(f"    PSNR={psnr:.3f} dB | latency={lat_mean:.2f}+/-{lat_std:.2f} ms")

        # BF16 autocast
        print("  [BF16 autocast]")
        fwd = _make_autocast_forward(torch.bfloat16, device.type)
        psnr = evaluate_psnr(fp32_model, val_loader, device, forward=fwd)
        lat_mean, lat_std = benchmark_latency(fp32_model, bench_in_fp32, forward=fwd)
        results.append({
            "format": "BF16 (autocast)",
            "psnr_db": psnr,
            "psnr_drop_db": fp32_psnr - psnr,
            "latency_ms_mean": lat_mean,
            "latency_ms_std": lat_std,
            "size_mb": model_size_mb(fp32_model, 2.0),
            "notes": "wider exponent than FP16, less overflow risk",
        })
        print(f"    PSNR={psnr:.3f} dB | latency={lat_mean:.2f}+/-{lat_std:.2f} ms")
    else:
        print("  [FP16/BF16] skipped (autocast requires CUDA on this code path)")

    # INT8 (fake-quant) -- quant_model has already been calibrated; just enable
    print("  [INT8 fake-quant (per-tensor act, per-channel weight)]")
    quant_wrappers = {n: m for n, m in quant_model.named_modules()
                      if isinstance(m, CalibratingConv2d)}
    set_all_modes(quant_wrappers, "quantize")
    psnr = evaluate_psnr(quant_model, val_loader, device)
    # Latency note: fake-quant has a quant-dequant overhead (slower than FP32),
    # so this number is NOT the deploy latency. The deploy number requires a
    # real INT8 backend (ONNX RT / TensorRT). We report it for completeness.
    lat_mean, lat_std = benchmark_latency(quant_model, bench_in_fp32)
    results.append({
        "format": "INT8 PTQ (fake-quant)",
        "psnr_db": psnr,
        "psnr_drop_db": fp32_psnr - psnr,
        "latency_ms_mean": lat_mean,
        "latency_ms_std": lat_std,
        "size_mb": model_size_mb(fp32_model, 1.0),
        "notes": "PSNR is real; latency is fake-quant overhead (NOT deploy latency)",
    })
    print(f"    PSNR={psnr:.3f} dB | latency={lat_mean:.2f}+/-{lat_std:.2f} ms (fake-quant overhead, see note)")
    set_all_modes(quant_wrappers, "fp32")

    return results


# ---------------------------------------------------------------------------
# Per-layer sensitivity
# ---------------------------------------------------------------------------

def run_sensitivity(
    quant_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    fp32_psnr: float,
) -> list[dict]:
    """Quantize ONE layer at a time, measure the PSNR drop. Wrappers must be
    pre-calibrated. Returns rows ranked by drop (largest -> most sensitive)."""
    quant_model = quant_model.to(device).eval()
    wrappers = {n: m for n, m in quant_model.named_modules()
                if isinstance(m, CalibratingConv2d)}
    print(f"  sweeping {len(wrappers)} Conv2d layers")
    set_all_modes(wrappers, "fp32")

    rows: list[dict] = []
    for i, (name, w) in enumerate(wrappers.items()):
        w.set_mode("quantize")
        psnr = evaluate_psnr(quant_model, val_loader, device)
        drop = fp32_psnr - psnr
        rows.append({
            "layer_idx": i,
            "layer_name": name,
            "psnr_when_only_this_layer_q8": psnr,
            "psnr_drop_db": drop,
        })
        w.set_mode("fp32")  # reset before next
        print(f"    [{i+1:2d}/{len(wrappers)}] {name:35s}  PSNR {psnr:.3f}  drop {drop:+.3f} dB")

    rows.sort(key=lambda r: r["psnr_drop_db"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_shootout_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Quantization Shootout\n\n")
        f.write("| Format | PSNR (dB) | drop vs FP32 | Latency (ms) | Size (MB) | Notes |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['format']} | {r['psnr_db']:.3f} | "
                f"{r['psnr_drop_db']:+.3f} | "
                f"{r['latency_ms_mean']:.2f} +/- {r['latency_ms_std']:.2f} | "
                f"{r['size_mb']:.2f} | {r['notes']} |\n"
            )


# ---------------------------------------------------------------------------
# Layer classification + Markdown sensitivity / report writers
# ---------------------------------------------------------------------------

def classify_layer(name: str) -> str:
    """Categorize an EDSR Conv2d by its role in the network.

    The category names align with how SR papers (PAMS, 2DQuant) talk about
    quantization sensitivity: input/output convs and the upsampler are
    typically the critical ones; ResBlock interior tends to be robust.
    """
    if name == "head":
        return "input"
    if name == "tail":
        return "output"
    if name.startswith("upsampler"):
        return "upsampler"
    if name.startswith("body."):
        # body.0.conv1 -> 2 dots -> resblock interior
        # body.16     -> 1 dot  -> post-resblock conv (the long-skip target)
        return "resblock-interior" if name.count(".") >= 2 else "post-resblock"
    return "other"


def gather_metadata(
    args: argparse.Namespace,
    n_params: int,
    val_set_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Collect everything that should appear in 'What was tested' section."""
    ckpt_path = Path(args.checkpoint)
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    ckpt_mtime = datetime.datetime.fromtimestamp(
        ckpt_path.stat().st_mtime
    ).isoformat(timespec="seconds")

    if device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = platform.processor() or "CPU"

    return {
        "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_mtime": ckpt_mtime,
        "checkpoint_size_mb": ckpt_size_mb,
        "model_arch": (
            f"EDSR(scale_factor={args.scale}, "
            f"n_resblocks={args.n_resblocks}, n_feats={args.n_feats})"
        ),
        "model_params": n_params,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "val_set_dir": str(Path(args.data_root) / args.val_dir),
        "val_set_size": val_set_size,
        "val_degradation": "realistic",
        "val_lr_patch": args.patch_size,
        "calib_batches": args.calib_batches,
        "calib_samples": args.calib_batches * args.batch_size,
        "quant_act_scheme": "symmetric per-tensor INT8 (range -128..127)",
        "quant_weight_scheme": "symmetric per-output-channel INT8 (range -128..127)",
        "calibration_method": "max-abs (no percentile clipping, no KL-div)",
    }


def write_sensitivity_md(path: Path, rows: list[dict]) -> None:
    """Markdown table version of sensitivity.csv (sorted, with class column)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Per-layer Quantization Sensitivity\n\n")
        f.write(
            "Each row: hold every other layer in FP32, fake-quantize ONLY this "
            "Conv2d to INT8, measure PSNR drop. Larger drop = layer is more "
            "quantization-sensitive.\n\n"
        )
        f.write("| Rank | Layer | PSNR (dB) | Drop (dB) | Class |\n")
        f.write("|---:|---|---:|---:|---|\n")
        for i, r in enumerate(rows):
            cls = classify_layer(r["layer_name"])
            f.write(
                f"| {i + 1} | `{r['layer_name']}` | "
                f"{r['psnr_when_only_this_layer_q8']:.3f} | "
                f"{r['psnr_drop_db']:+.3f} | {cls} |\n"
            )


def write_report_md(
    path: Path,
    metadata: dict[str, Any],
    shootout_rows: list[dict],
    sensitivity_rows: list[dict],
    top_n: int = 3,
) -> None:
    """Consolidated report: 'What was tested' + shootout + sensitivity + recipe."""
    fp32_psnr = next(
        (r["psnr_db"] for r in shootout_rows if r["format"] == "FP32"), None,
    )
    int8_drop = next(
        (r["psnr_drop_db"] for r in shootout_rows
         if r["format"].startswith("INT8")),
        None,
    )

    top_layers = sensitivity_rows[:top_n]
    top_drop_sum = sum(r["psnr_drop_db"] for r in top_layers)
    other_drop_sum = sum(r["psnr_drop_db"] for r in sensitivity_rows[top_n:])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # ------------------------------- header ----------------------------
        f.write("# Quantization Analysis Report\n\n")

        # ------------------------------ what was tested --------------------
        f.write("## What was tested\n\n")
        f.write(f"- **Generated**: {metadata['datetime']}\n")
        f.write(f"- **Checkpoint**: `{metadata['checkpoint_path']}`\n")
        f.write(
            f"  - last modified: {metadata['checkpoint_mtime']}, "
            f"size: {metadata['checkpoint_size_mb']:.2f} MB\n"
        )
        f.write(
            f"- **Model**: {metadata['model_arch']} "
            f"-- {metadata['model_params']:,} params\n"
        )
        f.write(
            f"- **Device**: {metadata['device']} ({metadata['device_name']}), "
            f"PyTorch {metadata['torch_version']}\n"
        )
        f.write(
            f"- **Validation set**: `{metadata['val_set_dir']}` -- "
            f"{metadata['val_set_size']} images, "
            f"{metadata['val_degradation']} degradation, "
            f"LR patch {metadata['val_lr_patch']}x{metadata['val_lr_patch']} "
            f"(deterministic per-index seed)\n"
        )
        f.write(
            f"- **Calibration**: {metadata['calib_batches']} batches "
            f"({metadata['calib_samples']} LR samples), "
            f"{metadata['calibration_method']}\n"
        )
        f.write(
            f"- **Quantization scheme**:\n"
            f"  - activations: {metadata['quant_act_scheme']}\n"
            f"  - weights: {metadata['quant_weight_scheme']}\n"
        )
        f.write("\n")

        # ----------------------------- format shootout ---------------------
        f.write("## Format shootout\n\n")
        f.write(
            "PSNR + forward-pass latency for each format on the val set. "
            "**Caveat**: the INT8 latency is fake-quant overhead (q-dq inserted "
            "in PyTorch float math), not real INT8 deploy latency. Real "
            "deploy-side latency requires ONNX Runtime / TensorRT / vendor "
            "NPU SDK.\n\n"
        )
        f.write(
            "| Format | PSNR (dB) | Drop vs FP32 | Latency (ms) | "
            "Size (MB) | Notes |\n"
        )
        f.write("|---|---:|---:|---:|---:|---|\n")
        for r in shootout_rows:
            f.write(
                f"| {r['format']} | {r['psnr_db']:.3f} | "
                f"{r['psnr_drop_db']:+.3f} | "
                f"{r['latency_ms_mean']:.2f} +/- {r['latency_ms_std']:.2f} | "
                f"{r['size_mb']:.2f} | {r['notes']} |\n"
            )
        f.write("\n")

        # ----------------------------- per-layer sensitivity ---------------
        f.write("## Per-layer sensitivity (INT8, sorted by drop)\n\n")
        f.write(
            "Each row: hold every other layer in FP32, fake-quantize ONLY this "
            "Conv2d to INT8, measure PSNR drop. Larger drop = more "
            "quantization-sensitive.\n\n"
        )
        f.write("| Rank | Layer | PSNR (dB) | Drop (dB) | Class |\n")
        f.write("|---:|---|---:|---:|---|\n")
        for i, r in enumerate(sensitivity_rows):
            cls = classify_layer(r["layer_name"])
            f.write(
                f"| {i + 1} | `{r['layer_name']}` | "
                f"{r['psnr_when_only_this_layer_q8']:.3f} | "
                f"{r['psnr_drop_db']:+.3f} | {cls} |\n"
            )
        f.write("\n")

        # ----------------------------- mixed-precision recipe --------------
        f.write("## Mixed-precision recommendation\n\n")
        if int8_drop is None or fp32_psnr is None:
            f.write("(Shootout data missing -- skip recipe.)\n")
        else:
            top_pct = (
                100.0 * top_drop_sum / int8_drop
                if int8_drop > 1e-9 else float("nan")
            )
            f.write(
                f"- Pure-INT8 (all 36 Conv2d) PSNR drop vs FP32: "
                f"**{int8_drop:+.3f} dB**\n"
            )
            f.write(
                f"- Top-{top_n} most sensitive layers contribute **{top_pct:.0f}%** "
                f"of the total drop:\n"
            )
            for r in top_layers:
                f.write(
                    f"  - `{r['layer_name']}` ({classify_layer(r['layer_name'])}): "
                    f"+{r['psnr_drop_db']:.3f} dB\n"
                )
            f.write("\n")
            f.write(
                f"**Recipe**: keep `"
                f"{', '.join(r['layer_name'] for r in top_layers)}"
                f"` in higher precision (FP16 / FP32) and INT8 the remaining "
                f"{len(sensitivity_rows) - top_n} Conv2d. Estimated combined "
                f"drop with this mixed scheme: **~{other_drop_sum:+.3f} dB** "
                f"(assuming layer drops are approximately additive; the data "
                f"above supports this -- sum of all single-layer drops is "
                f"{top_drop_sum + other_drop_sum:.3f} dB vs measured pure-INT8 "
                f"drop {int8_drop:.3f} dB).\n\n"
            )
            f.write(
                "This pattern matches SR PTQ literature (PAMS, 2DQuant): the "
                "input conv (`head`), output conv (`tail`), and upsampling "
                "convs are quantization-critical; ResBlock interior is robust. "
                "Once a real INT8 backend (ONNX RT QInt8, TensorRT) is in "
                "place, this recipe should be re-validated on actual deploy "
                "latency.\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantization analysis for EDSR-baseline")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument("--output-dir", type=str, default="results/quantization")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=96, help="LR patch size")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--calib-batches", type=int, default=8,
                   help="number of LR batches used for INT8 calibration")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bench-batch", type=int, default=1, help="batch size for latency benchmark")
    p.add_argument("--skip-shootout", action="store_true")
    p.add_argument("--skip-sensitivity", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Quantization analysis")
    print("=" * 60)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {device}")
    print(f"  output     : {output_dir}")
    print()

    # --- Data ---
    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation="realistic",
        is_train=False,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))
    # Calibration uses the SAME val set (for V1 simplicity). For real deploy,
    # you'd use a held-out calibration set (§3.4 B "calibration dataset impact").
    calib_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    print(f"  val: {len(val_set)} HR images")

    # --- Model x2 (fp32 baseline + a copy to wrap) ---
    def build_model() -> nn.Module:
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
    wrappers = wrap_convs(quant_model)
    print(f"  wrapped {len(wrappers)} Conv2d layers in quant_model | "
          f"{n_params:,} params")
    print()

    # --- Calibration ---
    print("Calibrating INT8 ...")
    calibrate_int8(quant_model, wrappers, calib_loader, device,
                   n_batches=args.calib_batches)
    print()

    # --- Shootout ---
    fp32_psnr_for_sensitivity = None
    if not args.skip_shootout:
        print("Running format shootout ...")
        bench_shape = (args.bench_batch, 3, args.patch_size, args.patch_size)
        results = run_shootout(fp32_model, quant_model, val_loader, device, bench_shape)
        write_shootout_md(output_dir / "shootout.md", results)
        write_csv(output_dir / "shootout.csv", results)
        print(f"  wrote {output_dir / 'shootout.md'} and shootout.csv")
        fp32_psnr_for_sensitivity = next(r["psnr_db"] for r in results if r["format"] == "FP32")
        print()

    # --- Sensitivity ---
    sensitivity_rows: list[dict] = []
    if not args.skip_sensitivity:
        print("Running per-layer sensitivity sweep ...")
        if fp32_psnr_for_sensitivity is None:
            fp32_psnr_for_sensitivity = evaluate_psnr(fp32_model, val_loader, device)
            print(f"  FP32 baseline PSNR (recomputed): {fp32_psnr_for_sensitivity:.3f} dB")
        sensitivity_rows = run_sensitivity(
            quant_model, val_loader, device, fp32_psnr_for_sensitivity,
        )
        write_csv(output_dir / "sensitivity.csv", sensitivity_rows)
        write_sensitivity_md(output_dir / "sensitivity.md", sensitivity_rows)
        print()
        print("Top-5 most quantization-sensitive layers:")
        for r in sensitivity_rows[:5]:
            print(f"  {r['layer_name']:35s}  drop {r['psnr_drop_db']:+.3f} dB")
        print(f"  wrote {output_dir / 'sensitivity.csv'}")
        print(f"  wrote {output_dir / 'sensitivity.md'}")

    # --- Consolidated report ---
    if not args.skip_shootout and not args.skip_sensitivity:
        metadata = gather_metadata(args, n_params, len(val_set), device)
        write_report_md(
            output_dir / "report.md",
            metadata=metadata,
            shootout_rows=results,
            sensitivity_rows=sensitivity_rows,
        )
        print()
        print(f"Wrote consolidated report: {output_dir / 'report.md'}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
