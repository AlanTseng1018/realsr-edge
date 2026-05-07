"""Evaluate a QAT-trained checkpoint and append its row to an existing shootout.

The QAT checkpoint has CalibratingConv2d-wrapped state (input_amax + calibrated
flags already set from training). To eval, we build EDSR, wrap_convs to install
the wrappers, load the QAT state dict, switch wrappers to "quantize" mode, and
run the same evaluate_metrics() as analyze.py's run_shootout. This guarantees
apples-to-apples comparison with the existing FP32 / FP16 / BF16 / INT8-PTQ
rows: same val loader, same metric implementations, same LPIPS net.

Run example::

    python -m src.quantization.eval_qat \\
        --qat-checkpoint results/runs/20260430_223739_ep0_b16_scale2_realistic_qat/checkpoints/best_qat.pt \\
        --fp32-checkpoint results/runs/20260427_143542_ep200_b16_scale2_realistic/checkpoints/best.pt \\
        --shootout-csv results/quantization/200ep_with_report/shootout.csv \\
        --shootout-md results/quantization/200ep_with_report/shootout.md
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.models.edsr import EDSR
from src.quantization.analyze import (
    evaluate_metrics,
    model_size_mb,
    write_shootout_md,
)
from src.quantization.fake_quant import (
    CalibratingConv2d,
    set_all_modes,
    wrap_convs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate QAT checkpoint and append to shootout")
    p.add_argument("--qat-checkpoint", type=str, required=True)
    p.add_argument("--fp32-checkpoint", type=str, required=True,
                   help="Reference FP32 checkpoint to compute drops against")
    p.add_argument("--shootout-csv", type=str, required=True)
    p.add_argument("--shootout-md", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lpips-net", type=str, default="squeeze",
                   choices=["alex", "vgg", "squeeze"])
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("QAT shootout-row evaluator")
    print("=" * 60)
    print(f"  qat ckpt   : {args.qat_checkpoint}")
    print(f"  fp32 ckpt  : {args.fp32_checkpoint}")
    print(f"  shootout   : {args.shootout_csv}")
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
    print(f"  val: {len(val_set)} HR images")

    # --- LPIPS ---
    print(f"Loading LPIPS (net={args.lpips_net}) ...")
    import lpips as lpips_pkg
    lpips_model = lpips_pkg.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    for p_ in lpips_model.parameters():
        p_.requires_grad_(False)

    # --- Build models ---
    def build() -> nn.Module:
        return EDSR(scale_factor=args.scale, n_resblocks=args.n_resblocks,
                    n_feats=args.n_feats)

    # FP32 baseline (for drop computation)
    fp32_model = build().to(device).eval()
    fp32_ckpt = torch.load(args.fp32_checkpoint, map_location=device, weights_only=False)
    fp32_model.load_state_dict(fp32_ckpt["model"])

    # QAT model: build + wrap, then load QAT state
    qat_model = build().to(device).eval()
    wrappers = wrap_convs(qat_model)  # installs CalibratingConv2d in place
    qat_ckpt = torch.load(args.qat_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = qat_model.load_state_dict(qat_ckpt["model"], strict=False)
    print(f"  QAT load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (first 5):", missing[:5])
    if unexpected:
        print("  unexpected keys (first 5):", unexpected[:5])
    print()

    # --- Eval FP32 baseline (for drops) ---
    print("Evaluating FP32 baseline (for drop reference) ...")
    fp32_psnr, fp32_ssim, fp32_lpips = evaluate_metrics(
        fp32_model, val_loader, device, lpips_model=lpips_model,
    )
    print(f"  FP32  PSNR={fp32_psnr:.3f} dB | SSIM={fp32_ssim:.4f} | LPIPS={fp32_lpips:.4f}")

    # --- Eval QAT-trained weights in FP32 mode (fake-quant OFF) ---
    # Tells us what QAT training did to the underlying weights, isolated from
    # the inference-time quantization. Comparing to FP32 baseline reveals
    # whether QAT regularized / degraded the FP32 weights themselves.
    print("Evaluating QAT-FP32 (QAT weights, fake-quant OFF) ...")
    set_all_modes(wrappers, "fp32")
    qfp_psnr, qfp_ssim, qfp_lpips = evaluate_metrics(
        qat_model, val_loader, device, lpips_model=lpips_model,
    )
    print(f"  QAT-FP32  PSNR={qfp_psnr:.3f} dB | SSIM={qfp_ssim:.4f} | LPIPS={qfp_lpips:.4f}")

    # --- Eval QAT (INT8 mode, fake-quant ON) ---
    print("Evaluating INT8 QAT (fake-quant ON) ...")
    set_all_modes(wrappers, "quantize")
    qat_psnr, qat_ssim, qat_lpips = evaluate_metrics(
        qat_model, val_loader, device, lpips_model=lpips_model,
    )
    set_all_modes(wrappers, "fp32")
    print(f"  QAT-INT8  PSNR={qat_psnr:.3f} dB | SSIM={qat_ssim:.4f} | LPIPS={qat_lpips:.4f}")
    print()

    # --- Build new rows ---
    qat_fp32_row = {
        "format": "FP32 (QAT weights)",
        "psnr_db": qfp_psnr,
        "psnr_drop_db": fp32_psnr - qfp_psnr,
        "ssim": qfp_ssim,
        "ssim_drop": fp32_ssim - qfp_ssim,
        "lpips": qfp_lpips,
        "lpips_rise": qfp_lpips - fp32_lpips,
        "size_mb": model_size_mb(qat_model, 4.0),
        "notes": "QAT-trained weights, fake-quant OFF (isolates training-time effect)",
    }
    qat_row = {
        "format": "INT8 QAT (fake-quant)",
        "psnr_db": qat_psnr,
        "psnr_drop_db": fp32_psnr - qat_psnr,
        "ssim": qat_ssim,
        "ssim_drop": fp32_ssim - qat_ssim,
        "lpips": qat_lpips,
        "lpips_rise": qat_lpips - fp32_lpips,
        "size_mb": model_size_mb(qat_model, 1.0),
        "notes": "QAT 20-epoch fine-tune from PTQ baseline, lr 1e-5",
    }

    # --- Read existing shootout, append QAT row ---
    shootout_csv = Path(args.shootout_csv)
    shootout_md = Path(args.shootout_md)

    with shootout_csv.open("r", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))

    # Drop any existing QAT rows (idempotent re-runs)
    qat_format_names = {qat_fp32_row["format"], qat_row["format"]}
    existing = [r for r in existing if r["format"] not in qat_format_names]

    # Coerce types from CSV strings back to floats
    numeric = ("psnr_db", "psnr_drop_db", "ssim", "ssim_drop",
               "lpips", "lpips_rise", "size_mb")
    for r in existing:
        for k in numeric:
            if k in r and r[k] != "":
                r[k] = float(r[k])

    rows = existing + [qat_fp32_row, qat_row]

    # --- Write back ---
    with shootout_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  appended QAT row -> {shootout_csv}")

    write_shootout_md(shootout_md, rows)
    print(f"  rewrote {shootout_md}")
    print()

    # --- Summary ---
    print("Final shootout:")
    print(f"  {'Format':<28}  {'PSNR':>8}  {'PSNR drop':>10}  "
          f"{'SSIM':>8}  {'LPIPS':>8}  {'Size MB':>8}")
    for r in rows:
        print(f"  {r['format']:<28}  {r['psnr_db']:>8.3f}  "
              f"{r['psnr_drop_db']:>+10.3f}  {r['ssim']:>8.4f}  "
              f"{r['lpips']:>8.4f}  {r['size_mb']:>8.2f}")


if __name__ == "__main__":
    main()
