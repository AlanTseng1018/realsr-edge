"""Structural distortion heatmap via gradient orientation difference.

Complements :mod:`src.quantization.lpips_heatmap`. LPIPS measures perceptual
similarity in deep-feature space and is known to be insensitive to geometric
distortion (warped window frames, bent rooflines, melted grid patterns).
This module visualizes structural distortion directly by comparing per-pixel
gradient orientation between two images.

Why this exists
---------------
Interview feedback raised the case where SR output of a building's windows
looks "obviously broken" to the human eye but LPIPS gives a passing score.
The mechanism: VGG features encode "this is a window" (texture / class) but
not "the window frame is straight". A pixel-level gradient orientation diff
catches exactly the failure mode LPIPS misses -- if a vertical edge becomes
slanted by 30 degrees, the local gradient direction rotates by 30 degrees,
regardless of whether the texture still looks window-like to VGG.

Output
------
``structural_heatmap_<image>.png``  2x3 layout
    Top row  : GT HR | FP32 SR | INT8 SR
    Bottom row: structural heatmap overlays for the three relevant pairs:
               GT vs FP32 (model's structural error vs ground truth)
               GT vs INT8 (deployed model's structural error vs ground truth)
               FP32 vs INT8 (quantization-induced structural delta only)

Colour reading:
    Red / yellow  = edge orientation rotated by ~90 deg (worst case)
    Blue / green  = edge well-aligned
    Black         = no significant edge in either image (flat region)

Usage::

    python -m src.quantization.structural_heatmap \\
        --checkpoint results/checkpoints/edsr_baseline/final.pt \\
        --output-dir results/quantization/200ep_with_report/structural_heatmaps \\
        --target-image 0879.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.data.degradation import RealisticDegradation
from src.models.edsr import EDSR
from src.quantization.analyze import calibrate_int8
from src.quantization._onnx_inference import OnnxSRRunner
from src.quantization.fake_quant import (
    CalibratingConv2d,
    set_all_modes,
    wrap_convs,
)
from src.quantization.lpips_heatmap import _make_realistic_lr, _to_chw


# ---------------------------------------------------------------------------
# Gradient orientation difference
# ---------------------------------------------------------------------------

COLORMAP_NAME = "YlOrRd"
COLORMAP_MAX_DEG = 30.0  # angular delta at which colormap saturates


def compute_gradient_orientation_heatmap(
    ref: np.ndarray,
    test: np.ndarray,
    edge_percentile: float = 70.0,
    blur_sigma: float = 1.0,
    colormap_max_deg: float = COLORMAP_MAX_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Per-pixel undirected gradient orientation difference between two images.

    Parameters
    ----------
    ref, test       : (H, W, 3) uint8 RGB
    edge_percentile : keep pixels whose max(ref_mag, test_mag) is above this
                      percentile -- masks out flat regions where orientation
                      is noise-dominated. Higher = stricter edge mask.
    blur_sigma      : Gaussian pre-blur sigma; suppresses one-pixel noise
                      jitter without affecting building-scale structure.

    Returns
    -------
    heatmap_rgb     : (H, W, 3) uint8, turbo-coloured, black on non-edges.
    edge_mask       : (H, W) bool, True where edge strength exceeds threshold.
                      Used by ``overlay_on`` to apply heatmap only on edges
                      while letting non-edge regions show the base image.
    delta_deg       : (H, W) float32, raw angular delta in degrees, NOT masked.
                      Used by ``find_worst_region`` to locate the area with
                      highest mean structural distortion for zoom-in.
    mean_delta_deg  : average angular delta in degrees, over edge pixels only
                      (more meaningful than over all pixels since flat areas
                      contribute pure noise).
    edge_pct_frac   : fraction of pixels classified as edge (sanity check).

    Strategy
    --------
    1. Convert to grayscale luminance + light Gaussian blur.
    2. Sobel gx, gy on both images.
    3. Pixel orientation theta = atan2(gy, gx).
    4. Undirected angular diff: min(|d|, pi - |d|). Edges are antipodal --
       an edge with gradient pointing +90 deg is the same as one pointing
       -90 deg; only the unsigned orientation matters.
    5. Mask to pixels where max(|grad_ref|, |grad_test|) is above the
       chosen percentile. We use max (not min) so that BOTH "edge moved"
       AND "edge appeared/disappeared" failures stay in the heatmap.
    """
    ref_g = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    test_g = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if blur_sigma > 0:
        ref_g = cv2.GaussianBlur(ref_g, (0, 0), blur_sigma)
        test_g = cv2.GaussianBlur(test_g, (0, 0), blur_sigma)

    gx_r = cv2.Sobel(ref_g, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(ref_g, cv2.CV_32F, 0, 1, ksize=3)
    gx_t = cv2.Sobel(test_g, cv2.CV_32F, 1, 0, ksize=3)
    gy_t = cv2.Sobel(test_g, cv2.CV_32F, 0, 1, ksize=3)

    theta_r = np.arctan2(gy_r, gx_r)
    theta_t = np.arctan2(gy_t, gx_t)
    mag_r = np.hypot(gx_r, gy_r)
    mag_t = np.hypot(gx_t, gy_t)

    # Undirected angular difference in [0, pi/2]
    delta = np.abs(theta_r - theta_t)
    delta = np.minimum(delta, np.pi - delta)
    delta = np.clip(delta, 0.0, np.pi / 2)

    mag_max = np.maximum(mag_r, mag_t)
    thresh = np.percentile(mag_max, edge_percentile)
    edge_mask = mag_max > thresh

    mean_delta_deg = (
        float(np.degrees(delta[edge_mask]).mean()) if edge_mask.any() else 0.0
    )
    edge_pct_frac = float(edge_mask.mean())

    # Normalize to [0, 1] where 1 = colormap_max_deg (default 30°).
    # Empirically, most per-pixel deltas fall in 0-30°; mapping the colormap
    # to [0°, 30°] uses the full color range for the meaningful regime
    # rather than wasting half of it on values that never occur.
    # Anything above colormap_max_deg saturates at the top colormap colour.
    delta_deg_for_cmap = np.degrees(delta)
    delta_norm = np.clip(delta_deg_for_cmap / colormap_max_deg, 0.0, 1.0)
    cmap = plt.get_cmap(COLORMAP_NAME)
    heatmap = (cmap(delta_norm)[..., :3] * 255).astype(np.uint8)
    heatmap[~edge_mask] = 255  # flat regions -> white (matches YlOrRd low end)

    delta_deg = np.degrees(delta).astype(np.float32)
    return heatmap, edge_mask, delta_deg, mean_delta_deg, edge_pct_frac


def find_worst_region(
    delta_deg: np.ndarray,
    edge_mask: np.ndarray,
    crop_size: int = 256,
    stride: int = 64,
    min_edge_frac: float = 0.20,
) -> tuple[int, int, float, float]:
    """Locate the (y, x) of a crop_size x crop_size window with the highest
    structurally-meaningful distortion concentration.

    Score = mean(delta over edges) * edge_fraction_in_region.

    Why both factors
    ----------------
    Earlier iteration scored on mean-delta-over-edges alone. That biased the
    zoom toward sparse-edge regions (sky with one spire silhouette, snow
    with one fur edge): a small number of outlier-orientation edges
    dominated the mean. The user-visible failure mode -- "windows look
    melted" -- happens in dense structured regions, where many edges are
    moderately mis-oriented. Multiplying by edge fraction forces the zoom
    onto dense content.

    Also gated by ``min_edge_frac`` so the search ignores low-content
    regions entirely.

    Returns ``(y, x, mean_delta_deg, edge_fraction)``. If no window
    qualifies, returns the global center crop.
    """
    H, W = delta_deg.shape
    if H < crop_size or W < crop_size:
        return 0, 0, 0.0, 0.0

    region_total = float(crop_size * crop_size)
    min_edge_pixels = int(region_total * min_edge_frac)

    cy = (H - crop_size) // 2
    cx = (W - crop_size) // 2
    best_y, best_x = cy, cx
    best_score = -1.0
    best_mean = 0.0
    best_frac = 0.0
    for y in range(0, H - crop_size + 1, stride):
        for x in range(0, W - crop_size + 1, stride):
            em = edge_mask[y:y + crop_size, x:x + crop_size]
            n_edge = int(em.sum())
            if n_edge < min_edge_pixels:
                continue
            region = delta_deg[y:y + crop_size, x:x + crop_size]
            mean_d = float(region[em].mean())
            frac = n_edge / region_total
            score = mean_d * frac
            if score > best_score:
                best_score = score
                best_mean = mean_d
                best_frac = frac
                best_y, best_x = y, x
    return best_y, best_x, best_mean, best_frac


def overlay_on(
    base: np.ndarray,
    heatmap: np.ndarray,
    edge_mask: np.ndarray,
    non_edge_dim: float = 0.70,
    edge_alpha: float = 0.75,
) -> np.ndarray:
    """Overlay heatmap on base image with edge-selective blending.

    - NON-EDGE pixels: base image at ``non_edge_dim`` brightness. Preserves
      spatial context (cathedral / scene structure visible) but visually
      recedes so coloured edges pop forward.
    - EDGE pixels: base * (1-edge_alpha) + heatmap * edge_alpha. Heatmap
      colour dominates (75%) while base structure still bleeds through
      (25%) so you can see WHICH architectural feature is mis-oriented.

    Works with both turbo (high values = bright) and YlOrRd (high values =
    deep red on white background) colormaps.
    """
    base_f = base.astype(np.float32)
    hm_f = heatmap.astype(np.float32)
    dimmed = base_f * non_edge_dim
    on_edge = base_f * (1.0 - edge_alpha) + hm_f * edge_alpha
    em = edge_mask[..., None]
    out = np.where(em, on_edge, dimmed)
    return out.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference on a single target image
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_structural_pack(
    fp32_model: nn.Module | None,
    quant_model: nn.Module | None,
    quant_wrappers: dict[str, CalibratingConv2d] | None,
    target_path: Path,
    target_idx: int,
    scale: int,
    device: torch.device,
    crop_hr: int,
    edge_percentile: float,
    blur_sigma: float,
    onnx_runner: OnnxSRRunner | None = None,
) -> dict:
    """FP32+INT8 inference on centre-crop of target, plus the three heatmaps.

    Inference backend is selected by ``onnx_runner``:
        * ``None``  -> Stage 2 mode: PyTorch fake-quant via fp32_model /
                       quant_model / quant_wrappers (must all be provided).
        * provided -> Stage 3 mode: real ONNX inference; the PyTorch model
                       args are unused and may be passed as ``None``.

    Returned pack dict shape is identical in both modes, so the same
    heatmap computation and figure layout apply downstream.
    """
    hr_full = cv2.imread(str(target_path))
    if hr_full is None:
        raise RuntimeError(f"cv2 failed to read {target_path}")
    hr_full = cv2.cvtColor(hr_full, cv2.COLOR_BGR2RGB)
    h, w = hr_full.shape[:2]
    if h < crop_hr or w < crop_hr:
        raise ValueError(f"image {target_path.name} {h}x{w} too small for crop {crop_hr}")

    cy, cx = h // 2, w // 2
    half = crop_hr // 2
    half -= half % scale
    hr = hr_full[cy - half:cy + half, cx - half:cx + half].copy()

    deg = RealisticDegradation()
    lr = _make_realistic_lr(hr, scale=scale, seed=target_idx, deg=deg)
    lr_t = _to_chw(lr).unsqueeze(0).to(device)

    if onnx_runner is not None:
        fp32_np = onnx_runner.run_fp32(lr_t)[0]   # (3, H, W)
        int8_np = onnx_runner.run_int8(lr_t)[0]
        fp32_rgb = (np.transpose(fp32_np, (1, 2, 0)) * 255).astype(np.uint8)
        int8_rgb = (np.transpose(int8_np, (1, 2, 0)) * 255).astype(np.uint8)
    else:
        sr_fp32 = fp32_model(lr_t).clamp(0.0, 1.0).float()

        set_all_modes(quant_wrappers, "quantize")
        sr_int8 = quant_model(lr_t).clamp(0.0, 1.0).float()
        set_all_modes(quant_wrappers, "fp32")

        fp32_rgb = (sr_fp32.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        int8_rgb = (sr_int8.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    hm_gt_fp32, em_gf, dd_gf, mean_gf, ef_gf = compute_gradient_orientation_heatmap(
        hr, fp32_rgb, edge_percentile=edge_percentile, blur_sigma=blur_sigma,
    )
    hm_gt_int8, em_gi, dd_gi, mean_gi, ef_gi = compute_gradient_orientation_heatmap(
        hr, int8_rgb, edge_percentile=edge_percentile, blur_sigma=blur_sigma,
    )
    hm_fp32_int8, em_fi, dd_fi, mean_fi, ef_fi = compute_gradient_orientation_heatmap(
        fp32_rgb, int8_rgb, edge_percentile=edge_percentile, blur_sigma=blur_sigma,
    )

    return {
        "gt": hr,
        "fp32_sr": fp32_rgb,
        "int8_sr": int8_rgb,
        "hm_gt_vs_fp32": hm_gt_fp32,
        "hm_gt_vs_int8": hm_gt_int8,
        "hm_fp32_vs_int8": hm_fp32_int8,
        "em_gt_vs_fp32": em_gf,
        "em_gt_vs_int8": em_gi,
        "em_fp32_vs_int8": em_fi,
        "dd_gt_vs_fp32": dd_gf,
        "dd_gt_vs_int8": dd_gi,
        "dd_fp32_vs_int8": dd_fi,
        "mean_gt_vs_fp32_deg": mean_gf,
        "mean_gt_vs_int8_deg": mean_gi,
        "mean_fp32_vs_int8_deg": mean_fi,
        "edge_frac_gt_vs_fp32": ef_gf,
        "edge_frac_gt_vs_int8": ef_gi,
        "edge_frac_fp32_vs_int8": ef_fi,
    }


def _draw_structural_panel(
    ax, base_rgb: np.ndarray, heatmap: np.ndarray, edge_mask: np.ndarray,
    mean_deg: float, title_prefix: str,
) -> None:
    """Render the heatmap overlay. Colormap saturates at COLORMAP_MAX_DEG,
    so panels with very low mean delta naturally show as mostly pale-yellow
    (= visually "calm") rather than the misleading dark purple that the
    earlier turbo-based version produced."""
    ax.imshow(overlay_on(base_rgb, heatmap, edge_mask))
    ax.set_title(f"{title_prefix}  mean={mean_deg:.2f}°")
    ax.axis("off")


def save_nine_panel(
    out_path: Path,
    pack: dict,
    target_name: str,
    zoom_crop: int = 256,
    zoom_stride: int = 64,
) -> None:
    """3x3 layout:
        Row 1: GT HR | FP32 SR | INT8 SR
        Row 2: structural Δ overlays (GT-vs-FP32, GT-vs-INT8, FP32-vs-INT8)
               -- panels with Δ < 1° collapse to text label.
        Row 3: zoom-in on the worst GT-vs-INT8 region
               -- GT crop | INT8 SR crop | raw heatmap crop, at the crop's
               native resolution within the subplot.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # ---- Row 1: source images ----
    axes[0, 0].imshow(pack["gt"]);      axes[0, 0].set_title("GT HR");   axes[0, 0].axis("off")
    axes[0, 1].imshow(pack["fp32_sr"]); axes[0, 1].set_title("FP32 SR"); axes[0, 1].axis("off")
    axes[0, 2].imshow(pack["int8_sr"]); axes[0, 2].set_title("INT8 SR"); axes[0, 2].axis("off")

    # ---- Row 2: structural overlays (with text fallback) ----
    _draw_structural_panel(
        axes[1, 0], pack["fp32_sr"], pack["hm_gt_vs_fp32"], pack["em_gt_vs_fp32"],
        pack["mean_gt_vs_fp32_deg"], "Structural Δ (GT vs FP32)",
    )
    _draw_structural_panel(
        axes[1, 1], pack["int8_sr"], pack["hm_gt_vs_int8"], pack["em_gt_vs_int8"],
        pack["mean_gt_vs_int8_deg"], "Structural Δ (GT vs INT8)",
    )
    _draw_structural_panel(
        axes[1, 2], pack["int8_sr"], pack["hm_fp32_vs_int8"], pack["em_fp32_vs_int8"],
        pack["mean_fp32_vs_int8_deg"], "Structural Δ (FP32 vs INT8)",
    )

    # ---- Row 3: zoom on worst GT-vs-INT8 region ----
    # Locate worst 256x256 region from GT-vs-INT8 (the deployed-model failure mode).
    y, x, region_mean_deg, region_edge_frac = find_worst_region(
        pack["dd_gt_vs_int8"], pack["em_gt_vs_int8"],
        crop_size=zoom_crop, stride=zoom_stride,
    )
    gt_crop = pack["gt"][y:y + zoom_crop, x:x + zoom_crop]
    sr_crop = pack["int8_sr"][y:y + zoom_crop, x:x + zoom_crop]
    hm_crop = pack["hm_gt_vs_int8"][y:y + zoom_crop, x:x + zoom_crop]

    axes[2, 0].imshow(gt_crop)
    axes[2, 0].set_title(f"GT (zoom @ y={y},x={x}, {zoom_crop}px)")
    axes[2, 0].axis("off")
    axes[2, 1].imshow(sr_crop)
    axes[2, 1].set_title("INT8 SR (zoom)")
    axes[2, 1].axis("off")
    axes[2, 2].imshow(hm_crop)
    axes[2, 2].set_title(
        f"Heatmap (zoom)  region mean Δ = {region_mean_deg:.2f}°  "
        f"edge density = {region_edge_frac*100:.1f}%"
    )
    axes[2, 2].axis("off")

    fig.suptitle(
        f"Gradient orientation structural distortion — {target_name}\n"
        f"Pale yellow = edge aligned (Δ ≈ 0°) • Deep red = edge orientation flipped "
        f"(Δ ≥ {COLORMAP_MAX_DEG:.0f}°) • Dim background = flat region\n"
        "Row 3 auto-zooms on the most structurally distorted 256px region of GT-vs-INT8 "
        "(scored by mean Δ × edge density).",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Val-set scan: rank images by structural distortion
# ---------------------------------------------------------------------------

@torch.no_grad()
def scan_val_set(
    quant_model: nn.Module | None,
    quant_wrappers: dict[str, CalibratingConv2d] | None,
    val_set: SRDataset,
    scale: int,
    device: torch.device,
    crop_hr: int,
    edge_percentile: float,
    blur_sigma: float,
    onnx_runner: OnnxSRRunner | None = None,
) -> list[dict]:
    """For every val image: compute GT-vs-INT8 mean angular delta over edges.

    Backend selected by ``onnx_runner`` (same convention as
    :func:`render_structural_pack`). Only INT8 path is run (FP32 is approximately
    equal as we already showed), halving the inference budget. Returns a list
    of dicts sorted by mean angular delta descending -- top entries are the
    most structurally distorted under deployed INT8 inference.
    """
    rows: list[dict] = []
    n = len(val_set.hr_paths)
    print(f"Scanning {n} val images for structural distortion ranking ...")
    deg = RealisticDegradation()
    skipped = 0
    for idx, hr_path in enumerate(val_set.hr_paths):
        hr_full = cv2.imread(str(hr_path))
        if hr_full is None:
            skipped += 1
            continue
        hr_full = cv2.cvtColor(hr_full, cv2.COLOR_BGR2RGB)
        h, w = hr_full.shape[:2]
        if h < crop_hr or w < crop_hr:
            skipped += 1
            continue

        cy, cx = h // 2, w // 2
        half = crop_hr // 2; half -= half % scale
        hr = hr_full[cy - half:cy + half, cx - half:cx + half].copy()

        lr = _make_realistic_lr(hr, scale=scale, seed=idx, deg=deg)
        lr_t = _to_chw(lr).unsqueeze(0).to(device)

        if onnx_runner is not None:
            int8_np = onnx_runner.run_int8(lr_t)[0]
            int8_rgb = (np.transpose(int8_np, (1, 2, 0)) * 255).astype(np.uint8)
        else:
            set_all_modes(quant_wrappers, "quantize")
            sr_int8 = quant_model(lr_t).clamp(0.0, 1.0).float()
            set_all_modes(quant_wrappers, "fp32")
            int8_rgb = (sr_int8.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        _, _, _, mean_deg, edge_frac = compute_gradient_orientation_heatmap(
            hr, int8_rgb, edge_percentile=edge_percentile, blur_sigma=blur_sigma,
        )
        rows.append({
            "name": hr_path.name,
            "mean_angular_delta_deg_gt_vs_int8": mean_deg,
            "edge_fraction": edge_frac,
        })
        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{n}]  latest: {hr_path.name} Δ={mean_deg:.2f}°")
    if skipped:
        print(f"  (skipped {skipped} images smaller than crop {crop_hr})")
    rows.sort(key=lambda r: r["mean_angular_delta_deg_gt_vs_int8"], reverse=True)
    return rows


def save_scan_csv(rows: list[dict], out_path: Path) -> None:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structural distortion heatmap (gradient orientation)")
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
    p.add_argument("--target-image", type=str, default="0879.png")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=96,
                   help="LR patch size for calibration loader (HR patch = patch_size * scale)")
    p.add_argument("--heatmap-crop-hr", type=int, default=1024,
                   help="HR-side square crop for the target image")
    p.add_argument("--edge-percentile", type=float, default=70.0,
                   help="Edge-strength percentile cutoff (higher = stricter)")
    p.add_argument("--blur-sigma", type=float, default=1.0,
                   help="Gaussian pre-blur sigma before Sobel")
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--scan", action="store_true",
                   help="Loop all val images, rank by structural distortion, save CSV.")
    p.add_argument("--scan-top-k", type=int, default=15,
                   help="Print top-K entries to stdout after scan.")
    args = p.parse_args()
    # Source-specific arg validation -- cleaner here than via mutex argparse groups.
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
    print("Structural distortion heatmap (gradient orientation)")
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
    print(f"  edge pct cut : {args.edge_percentile}")
    print(f"  blur sigma   : {args.blur_sigma}")
    print()

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
        # ONNX deployment-side: INT8 ONNX was calibrated at export time
        # (ORT quantize_static, MinMax) -- no runtime calibration here.
        onnx_runner = OnnxSRRunner(Path(args.fp32_onnx), Path(args.int8_onnx))
        print(f"  ONNX runner: {onnx_runner.describe()}")
        print()

    # Output suffix so PyTorch (Stage 2) and ONNX (Stage 3) artefacts coexist.
    src_suffix = "_onnx" if args.source == "onnx" else ""

    # --- Scan mode: rank all val images by structural distortion, then exit ---
    if args.scan:
        rows = scan_val_set(
            quant_model, wrappers, val_set, scale=args.scale, device=device,
            crop_hr=args.heatmap_crop_hr, edge_percentile=args.edge_percentile,
            blur_sigma=args.blur_sigma, onnx_runner=onnx_runner,
        )
        csv_path = output_dir / f"scan_ranking{src_suffix}.csv"
        save_scan_csv(rows, csv_path)
        print()
        print(f"Top {args.scan_top_k} most structurally distorted (GT-vs-INT8):")
        print(f"  {'rank':>4}  {'image':<14}  {'mean Δ (°)':>10}  {'edge %':>7}")
        for i, r in enumerate(rows[:args.scan_top_k], 1):
            print(f"  {i:>4}  {r['name']:<14}  "
                  f"{r['mean_angular_delta_deg_gt_vs_int8']:>10.3f}  "
                  f"{r['edge_fraction']*100:>6.2f}%")
        print(f"\n  full ranking -> {csv_path}")
        print("\nDone (scan).")
        return

    # --- Single-image mode: full 9-panel render ---
    target_path = Path(args.data_root) / args.val_dir / args.target_image
    target_idx = next(
        (i for i, p in enumerate(val_set.hr_paths) if p.name == args.target_image),
        None,
    )
    if target_idx is None:
        raise SystemExit(f"target-image {args.target_image} not found in val set")

    print(f"Rendering structural heatmap on {args.target_image} ...")
    pack = render_structural_pack(
        fp32_model, quant_model, wrappers,
        target_path=target_path, target_idx=target_idx,
        scale=args.scale, device=device, crop_hr=args.heatmap_crop_hr,
        edge_percentile=args.edge_percentile, blur_sigma=args.blur_sigma,
        onnx_runner=onnx_runner,
    )
    print(f"  edge fractions (sanity, ~{100 - args.edge_percentile:.0f}% expected):")
    print(f"    GT-vs-FP32 : {pack['edge_frac_gt_vs_fp32']*100:.1f}%")
    print(f"    GT-vs-INT8 : {pack['edge_frac_gt_vs_int8']*100:.1f}%")
    print(f"    FP32-vs-INT8: {pack['edge_frac_fp32_vs_int8']*100:.1f}%")
    print(f"  mean angular delta on edge pixels (degrees):")
    print(f"    GT-vs-FP32 : {pack['mean_gt_vs_fp32_deg']:.2f}°")
    print(f"    GT-vs-INT8 : {pack['mean_gt_vs_int8_deg']:.2f}°")
    print(f"    FP32-vs-INT8: {pack['mean_fp32_vs_int8_deg']:.2f}°")

    out_png = output_dir / f"structural_heatmap_{Path(args.target_image).stem}{src_suffix}.png"
    save_nine_panel(out_png, pack, f"{args.target_image} [{args.source}]")
    print(f"  wrote {out_png}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
