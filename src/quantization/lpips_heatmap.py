"""Per-image LPIPS distribution and spatial heatmap for INT8 quantization.

This is a perceptual deep-dive that complements the aggregate shootout
(:mod:`src.quantization.analyze`). Aggregate metrics (mean PSNR / SSIM /
LPIPS) average over the val set and can hide image-type-specific failure
modes -- in particular, INT8 quantization tends to introduce banding /
posterization in smooth regions that the per-image average smooths away.

Outputs (all under ``--output-dir``):
    per_image_lpips.csv        One row per val image: idx, name, fp32_lpips,
                               int8_lpips, lpips_rise, ssim_rise.
    distribution.png           Histogram of INT8 lpips_rise across val images,
                               with the target image's position marked.
    heatmap_<image_name>.png   3-panel: GT HR | INT8 SR | spatial LPIPS
                               overlay. Highlights *where* INT8 introduces
                               perceptual error (smooth regions vs textured).

Usage::

    python -m src.quantization.lpips_heatmap \\
        --checkpoint results/checkpoints/edsr_baseline/final.pt \\
        --output-dir results/quantization/200ep_with_report/lpips_heatmaps \\
        --target-image 0879.png

The spatial LPIPS shown in the heatmap is **INT8_SR vs FP32_SR**, not vs
GT. This isolates the *quantization-induced* perceptual delta from the
underlying SR reconstruction error, which is what the deployment decision
turns on.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.data.degradation import RealisticDegradation
from src.models.edsr import EDSR
from src.quantization._onnx_inference import OnnxSRRunner
from src.quantization.analyze import calibrate_int8
from src.quantization.fake_quant import (
    CalibratingConv2d,
    set_all_modes,
    wrap_convs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_lpips(net: str, device: torch.device, spatial: bool = False) -> nn.Module:
    import lpips as lpips_pkg
    m = lpips_pkg.LPIPS(net=net, spatial=spatial, verbose=False).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _to_chw(img_uint8: np.ndarray) -> torch.Tensor:
    """``(H, W, 3) uint8`` -> ``(3, H, W) float32`` in [0, 1]."""
    return torch.from_numpy(img_uint8).permute(2, 0, 1).contiguous().float() / 255.0


def _make_realistic_lr(
    hr: np.ndarray, scale: int, seed: int, deg: RealisticDegradation
) -> np.ndarray:
    """Deterministic version matching :meth:`SRDataset._make_realistic_lr`."""
    rng = np.random.default_rng(seed)
    out = hr
    if rng.random() < 0.5:
        out = deg.apply_blur(out, random_state=rng)
    out = deg.apply_downsample(out, scale=scale, random_state=rng)
    if rng.random() < 0.5:
        out = deg.apply_banding(out, random_state=rng)
    if rng.random() < 0.5:
        out = deg.apply_noise(out, random_state=rng)
    if rng.random() < 0.5:
        out = deg.apply_jpeg_compression(out, random_state=rng)
    return out


# ---------------------------------------------------------------------------
# Per-image LPIPS distribution
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_image_lpips(
    fp32_model: nn.Module | None,
    quant_model: nn.Module | None,
    quant_wrappers: dict[str, CalibratingConv2d] | None,
    val_set: SRDataset,
    device: torch.device,
    lpips_model: nn.Module,
    onnx_runner: OnnxSRRunner | None = None,
) -> list[dict]:
    """Per-image LPIPS for FP32 SR vs GT and INT8 SR vs GT.

    Iterates the val set one image at a time (batch=1) so we get one
    LPIPS scalar per image, suitable for distribution analysis. SSIM is
    NOT recomputed here -- it's already in the aggregate shootout.

    Backend selected by ``onnx_runner`` (same convention as
    :func:`make_heatmap`).
    """
    rows: list[dict] = []
    for idx in range(len(val_set)):
        lr, hr = val_set[idx]
        lr = lr.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)

        if onnx_runner is not None:
            sr_fp32 = torch.from_numpy(onnx_runner.run_fp32(lr)).to(device).float()
            sr_int8 = torch.from_numpy(onnx_runner.run_int8(lr)).to(device).float()
        else:
            sr_fp32 = fp32_model(lr).clamp(0.0, 1.0).float()
            set_all_modes(quant_wrappers, "quantize")
            sr_int8 = quant_model(lr).clamp(0.0, 1.0).float()
            set_all_modes(quant_wrappers, "fp32")

        d_fp32 = lpips_model(sr_fp32 * 2 - 1, hr * 2 - 1).flatten().item()
        d_int8 = lpips_model(sr_int8 * 2 - 1, hr * 2 - 1).flatten().item()
        # Also INT8-vs-FP32 distance: isolates the quantization-induced
        # perceptual delta from the underlying SR reconstruction error.
        d_q_only = lpips_model(sr_int8 * 2 - 1, sr_fp32 * 2 - 1).flatten().item()

        name = val_set.hr_paths[idx % len(val_set.hr_paths)].name
        rows.append({
            "idx": idx,
            "name": name,
            "fp32_lpips": d_fp32,
            "int8_lpips": d_int8,
            "lpips_rise": d_int8 - d_fp32,
            "int8_vs_fp32_lpips": d_q_only,
        })
        if (idx + 1) % 10 == 0:
            print(f"    [{idx+1}/{len(val_set)}] {name}  rise={d_int8 - d_fp32:+.4f}")

    return rows


# ---------------------------------------------------------------------------
# Spatial heatmap on a single full image
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_heatmap(
    fp32_model: nn.Module | None,
    quant_model: nn.Module | None,
    quant_wrappers: dict[str, CalibratingConv2d] | None,
    target_path: Path,
    target_idx: int,
    scale: int,
    device: torch.device,
    lpips_spatial: nn.Module,
    crop_hr: int = 1024,
    onnx_runner: OnnxSRRunner | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run FP32 + INT8 inference on a center-crop of the target HR image,
    return (gt_hr_rgb, int8_sr_rgb, lpips_overlay_rgb, lpips_scalar).

    Backend selected by ``onnx_runner``:
        * ``None``  -> Stage 2 PyTorch fake-quant (fp32_model / quant_model /
                       quant_wrappers must all be provided).
        * provided -> Stage 3 ONNX inference; PyTorch model args unused.

    The crop is square ``crop_hr x crop_hr`` HR pixels (i.e. the LR side
    is ``crop_hr // scale``). Cropping caps GPU memory and keeps the
    output a clean square for slide layout.
    """
    hr_full = cv2.imread(str(target_path))
    if hr_full is None:
        raise RuntimeError(f"cv2 failed to read {target_path}")
    hr_full = cv2.cvtColor(hr_full, cv2.COLOR_BGR2RGB)
    h, w = hr_full.shape[:2]
    if h < crop_hr or w < crop_hr:
        raise ValueError(f"Image {target_path.name} {h}x{w} too small for crop {crop_hr}")

    # Snap crop to scale grid so HR/LR sizes match exactly.
    cy, cx = h // 2, w // 2
    half = crop_hr // 2
    half -= half % scale
    hr = hr_full[cy - half:cy + half, cx - half:cx + half].copy()

    # Generate LR via the same deterministic pipeline as SRDataset val mode.
    deg = RealisticDegradation()
    lr = _make_realistic_lr(hr, scale=scale, seed=target_idx, deg=deg)

    hr_t = _to_chw(hr).unsqueeze(0).to(device)
    lr_t = _to_chw(lr).unsqueeze(0).to(device)

    if onnx_runner is not None:
        # ONNX returns numpy (1, 3, sH, sW); lift to torch on `device` so the
        # spatial LPIPS network (a torch module on `device`) can consume it.
        fp32_np = onnx_runner.run_fp32(lr_t)
        int8_np = onnx_runner.run_int8(lr_t)
        sr_fp32 = torch.from_numpy(fp32_np).to(device).float()
        sr_int8 = torch.from_numpy(int8_np).to(device).float()
    else:
        # FP32 SR (PyTorch)
        sr_fp32 = fp32_model(lr_t).clamp(0.0, 1.0).float()
        # INT8 SR (PyTorch fake-quant)
        set_all_modes(quant_wrappers, "quantize")
        sr_int8 = quant_model(lr_t).clamp(0.0, 1.0).float()
        set_all_modes(quant_wrappers, "fp32")

    # Spatial LPIPS: INT8 SR vs FP32 SR (isolates quantization-induced delta)
    # Output shape: (1, 1, H', W') with H'/W' a downsampled spatial map.
    sp = lpips_spatial(sr_int8 * 2 - 1, sr_fp32 * 2 - 1)
    sp_np = sp.squeeze().cpu().numpy()
    lp_scalar = float(sp_np.mean())

    # Upsample heatmap to HR size and overlay.
    H, W = hr.shape[:2]
    sp_full = cv2.resize(sp_np, (W, H), interpolation=cv2.INTER_LINEAR)
    # Normalize for display (preserve the absolute value via colorbar in caller).
    sp_norm = sp_full / max(sp_full.max(), 1e-8)
    cmap = plt.get_cmap("hot")
    heatmap_rgb = (cmap(sp_norm)[..., :3] * 255).astype(np.uint8)

    int8_sr_rgb = (
        sr_int8.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)

    # Overlay: 60% original, 40% heatmap, on the INT8 SR.
    overlay = (0.55 * int8_sr_rgb + 0.45 * heatmap_rgb).clip(0, 255).astype(np.uint8)
    return hr, int8_sr_rgb, overlay, lp_scalar


def save_heatmap_png(
    out_path: Path,
    gt_hr: np.ndarray,
    int8_sr: np.ndarray,
    overlay: np.ndarray,
    title: str,
    lpips_scalar: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(gt_hr); axes[0].set_title("GT HR"); axes[0].axis("off")
    axes[1].imshow(int8_sr); axes[1].set_title("INT8 SR"); axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(f"LPIPS heatmap (INT8 vs FP32)  mean={lpips_scalar:.4f}")
    axes[2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_distribution_png(
    out_path: Path,
    rows: list[dict],
    target_name: str,
) -> None:
    rises = np.array([r["lpips_rise"] for r in rows])
    target_rise = next(
        (r["lpips_rise"] for r in rows if r["name"] == target_name), None
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rises, bins=20, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(rises.mean(), color="darkblue", linestyle="--", linewidth=1.5,
               label=f"mean={rises.mean():.4f}")
    if target_rise is not None:
        ax.axvline(target_rise, color="red", linestyle="-", linewidth=2.0,
                   label=f"{target_name} rise={target_rise:.4f}")
    ax.set_xlabel("INT8 LPIPS rise vs FP32 (per image)")
    ax.set_ylabel("# val images")
    ax.set_title(
        "Per-image perceptual gap from INT8 fake-quant\n"
        "(higher = INT8 hurts perception more on this image)"
    )
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_per_image_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LPIPS spatial + per-image analysis")
    p.add_argument("--source", type=str, default="pytorch", choices=["pytorch", "onnx"],
                   help="Inference backend. 'pytorch' = Stage-2 fake-quant via best.pt; "
                        "'onnx' = Stage-3 deployment-check via exported ONNX files.")
    p.add_argument("--checkpoint", type=str,
                   help="PyTorch best.pt (required for --source pytorch)")
    p.add_argument("--fp32-onnx", type=str,
                   help="FP32 ONNX path (required for --source onnx)")
    p.add_argument("--int8-onnx", type=str,
                   help="INT8 ONNX path (required for --source onnx)")
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--target-image", type=str, default="0879.png",
                   help="Filename inside val-dir to render heatmap for")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=96, help="LR patch size for per-image eval")
    p.add_argument("--heatmap-crop-hr", type=int, default=1024,
                   help="HR-side square crop for the heatmap target")
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lpips-net", type=str, default="squeeze",
                   choices=["alex", "vgg", "squeeze"])
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.source == "pytorch" and not args.checkpoint:
        p.error("--checkpoint is required when --source=pytorch")
    if args.source == "onnx" and (not args.fp32_onnx or not args.int8_onnx):
        p.error("--fp32-onnx and --int8-onnx are required when --source=onnx")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LPIPS perceptual analysis")
    print("=" * 60)
    print(f"  source       : {args.source}")
    if args.source == "pytorch":
        print(f"  checkpoint   : {args.checkpoint}")
    else:
        print(f"  fp32 onnx    : {args.fp32_onnx}")
        print(f"  int8 onnx    : {args.int8_onnx}")
    print(f"  device       : {device}")
    print(f"  output       : {output_dir}")
    print(f"  target image : {args.target_image}")
    print(f"  lpips net    : {args.lpips_net}")
    print()

    # --- Data ---
    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation="realistic",
        is_train=False,
    )
    print(f"  val: {len(val_set)} HR images")

    fp32_model: nn.Module | None = None
    quant_model: nn.Module | None = None
    wrappers: dict[str, CalibratingConv2d] | None = None
    onnx_runner: OnnxSRRunner | None = None

    if args.source == "pytorch":
        calib_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=(device.type == "cuda"))

        def build() -> nn.Module:
            m = EDSR(scale_factor=args.scale, n_resblocks=args.n_resblocks,
                     n_feats=args.n_feats)
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            m.load_state_dict(ckpt["model"])
            return m

        fp32_model = build().to(device).eval()
        quant_model = build().to(device).eval()
        wrappers = wrap_convs(quant_model)

        print("Calibrating INT8 ...")
        calibrate_int8(quant_model, wrappers, calib_loader, device,
                       n_batches=args.calib_batches)
        print()
    else:
        onnx_runner = OnnxSRRunner(Path(args.fp32_onnx), Path(args.int8_onnx))
        print(f"  ONNX runner: {onnx_runner.describe()}")
        print()

    # --- LPIPS models (scalar + spatial) ---
    print(f"Loading LPIPS (net={args.lpips_net}) ...")
    lpips_scalar = _load_lpips(args.lpips_net, device, spatial=False)
    lpips_spatial = _load_lpips(args.lpips_net, device, spatial=True)
    print()

    # Output suffix so PyTorch (Stage 2) and ONNX (Stage 3) artefacts coexist.
    src_suffix = "_onnx" if args.source == "onnx" else ""

    # --- Per-image LPIPS sweep ---
    print("Computing per-image LPIPS across val set ...")
    rows = per_image_lpips(
        fp32_model, quant_model, wrappers, val_set, device, lpips_scalar,
        onnx_runner=onnx_runner,
    )
    write_per_image_csv(output_dir / f"per_image_lpips{src_suffix}.csv", rows)
    print(f"  wrote {output_dir / f'per_image_lpips{src_suffix}.csv'}")

    rises = np.array([r["lpips_rise"] for r in rows])
    print(f"  rise: mean={rises.mean():+.4f}  median={np.median(rises):+.4f}  "
          f"max={rises.max():+.4f}  min={rises.min():+.4f}")
    print()

    # --- Distribution histogram ---
    print("Saving distribution histogram ...")
    save_distribution_png(
        output_dir / f"distribution{src_suffix}.png", rows, args.target_image,
    )
    print(f"  wrote {output_dir / f'distribution{src_suffix}.png'}")
    print()

    # --- Heatmap on target image ---
    print(f"Rendering heatmap on {args.target_image} ...")
    target_path = Path(args.data_root) / args.val_dir / args.target_image
    target_idx = next(
        (i for i, p in enumerate(val_set.hr_paths) if p.name == args.target_image),
        None,
    )
    if target_idx is None:
        raise SystemExit(f"target-image {args.target_image} not found in val set")
    gt_hr, int8_sr, overlay, lp = make_heatmap(
        fp32_model, quant_model, wrappers,
        target_path=target_path, target_idx=target_idx,
        scale=args.scale, device=device, lpips_spatial=lpips_spatial,
        crop_hr=args.heatmap_crop_hr, onnx_runner=onnx_runner,
    )
    out_png = output_dir / f"heatmap_{Path(args.target_image).stem}{src_suffix}.png"
    save_heatmap_png(
        out_png, gt_hr, int8_sr, overlay,
        title=f"Quantization-induced perceptual gap on {args.target_image} [{args.source}]",
        lpips_scalar=lp,
    )
    print(f"  wrote {out_png}  (mean spatial LPIPS = {lp:.4f})")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
