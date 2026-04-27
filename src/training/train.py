"""Train EDSR-baseline on DIV2K, with optional torch.compile.

Run examples
------------
Smoke test (2 epochs, batch 2, no compile)::

    python -m src.training.train --quick

Full Track B (realistic degradation), default config, with torch.compile::

    python -m src.training.train --compile --compile-mode default

Track A (academic bicubic baseline), no compile::

    python -m src.training.train --degradation bicubic

Resume from a checkpoint::

    python -m src.training.train --resume results/runs/<run_id>/checkpoints/best.pt

Notes on torch.compile
----------------------
* `mode='default'` is fastest to compile (~10-30 s) and gives ~1.3-1.7x speedup.
* `mode='reduce-overhead'` uses CUDA graphs; ~1.5-1.8x but stricter shape requirements.
* `mode='max-autotune'` autotunes kernels (minutes to compile); ~1.8-2.5x.
* On Windows, torch.compile relies on Triton via the `triton-windows` package.
  If you see ``backend='inductor' raised`` errors, fall back to ``--compile``
  off and report the trace.

Notes on memory (RTX 3060 6GB)
------------------------------
* Default batch=16, patch=96 (LR) / 192 (HR), n_feats=64 fits in ~3-4 GB.
* If OOM: drop batch to 8 or patch to 64.
* AMP / mixed precision is intentionally NOT enabled in V1 — keeps the
  training loop minimal. Add later via ``torch.amp.autocast`` + ``GradScaler``.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.models.edsr import EDSR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_dict_for_save(model: nn.Module) -> dict[str, Any]:
    return getattr(model, "_orig_mod", model).state_dict()


def load_state_dict(model: nn.Module, state: dict[str, Any]) -> None:
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(state)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Return (mean PSNR dB, mean SSIM) over the validation loader."""
    model.eval()
    psnr_sum, ssim_sum, count = 0.0, 0.0, 0
    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        sr = model(lr).clamp(0.0, 1.0)

        mse = ((sr - hr) ** 2).mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
        psnr_sum += psnr.sum().item()

        sr_np = sr.cpu().numpy()
        hr_np = hr.cpu().numpy()
        for i in range(sr_np.shape[0]):
            s = ssim_fn(
                hr_np[i].transpose(1, 2, 0),
                sr_np[i].transpose(1, 2, 0),
                data_range=1.0,
                channel_axis=2,
            )
            ssim_sum += s

        count += psnr.numel()

    return psnr_sum / max(count, 1), ssim_sum / max(count, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_psnr: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state_dict_for_save(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "args": vars(args),
        },
        path,
    )


def make_run_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (
        f"{ts}_ep{args.epochs}_b{args.batch_size}"
        f"_scale{args.scale}_{args.degradation}"
    )
    run_dir = Path(args.runs_dir) / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "val_samples").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_val_samples(
    model: nn.Module,
    val_set: SRDataset,
    run_dir: Path,
    device: torch.device,
    n_samples: int = 5,
) -> None:
    """Save LR | Bicubic | SR | HR comparison PNGs for n_samples val images."""
    model.eval()
    indices = list(range(min(n_samples, len(val_set))))
    print(f"\n  Saving {len(indices)} validation sample images...")

    for idx in indices:
        lr, hr = val_set[idx]

        with torch.no_grad():
            sr = model(lr.unsqueeze(0).to(device)).clamp(0, 1).squeeze(0).cpu()

        lr_np = (lr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bic_np = cv2.resize(
            lr_np, (hr.shape[2], hr.shape[1]), interpolation=cv2.INTER_CUBIC
        )
        bic = torch.from_numpy(bic_np).permute(2, 0, 1).float() / 255.0

        def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
            mse = ((a - b) ** 2).mean().item()
            return 10 * math.log10(1.0 / max(mse, 1e-10))

        def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
            return ssim_fn(
                b.permute(1, 2, 0).numpy(),
                a.permute(1, 2, 0).numpy(),
                data_range=1.0,
                channel_axis=2,
            )

        psnr_bic = _psnr(bic, hr)
        psnr_sr  = _psnr(sr, hr)
        ssim_bic = _ssim(bic, hr)
        ssim_sr  = _ssim(sr, hr)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        axes[0].imshow(lr.permute(1, 2, 0).numpy())
        axes[0].set_title(f"LR  (degraded input)\n{tuple(lr.shape[1:])}", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(bic.permute(1, 2, 0).numpy().clip(0, 1))
        axes[1].set_title(
            f"Bicubic baseline\nPSNR {psnr_bic:.2f} dB | SSIM {ssim_bic:.4f}",
            fontsize=11,
        )
        axes[1].axis("off")

        axes[2].imshow(sr.permute(1, 2, 0).numpy().clip(0, 1))
        axes[2].set_title(
            f"EDSR SR  (restored)\nPSNR {psnr_sr:.2f} dB | SSIM {ssim_sr:.4f}",
            fontsize=11,
        )
        axes[2].axis("off")

        axes[3].imshow(hr.permute(1, 2, 0).numpy())
        axes[3].set_title(f"HR  (ground truth)\n{tuple(hr.shape[1:])}", fontsize=11)
        axes[3].axis("off")

        fig.suptitle(f"Val sample {idx + 1:04d}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = run_dir / "val_samples" / f"sample_{idx + 1:04d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    sample {idx + 1:04d} → PSNR {psnr_sr:.2f} dB | SSIM {ssim_sr:.4f}")


def save_curves(
    all_epochs: list[int],
    train_losses: list[float],
    val_epochs: list[int],
    val_psnrs: list[float],
    val_ssims: list[float],
    run_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(all_epochs, train_losses, "o-", color="tab:blue", linewidth=1.5, markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L1 Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_epochs, val_psnrs, "s-", color="tab:green", linewidth=1.5, markersize=5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR")
    axes[1].grid(True, alpha=0.3)
    if val_psnrs:
        best_ep = val_epochs[val_psnrs.index(max(val_psnrs))]
        axes[1].axvline(best_ep, color="tab:green", linestyle="--", alpha=0.5,
                        label=f"best ep {best_ep}")
        axes[1].legend(fontsize=9)

    axes[2].plot(val_epochs, val_ssims, "^-", color="tab:orange", linewidth=1.5, markersize=5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Validation SSIM")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = run_dir / "curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  curves saved → {out}")


def save_metrics_csv(
    val_epochs: list[int],
    train_losses: list[float],
    val_psnrs: list[float],
    val_ssims: list[float],
    run_dir: Path,
) -> None:
    out = run_dir / "metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_psnr_db", "val_ssim"])
        for ep, loss, psnr, ssim in zip(val_epochs, train_losses, val_psnrs, val_ssims):
            w.writerow([ep, f"{loss:.6f}", f"{psnr:.4f}", f"{ssim:.6f}"])
    print(f"  metrics saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDSR training (torch.compile-enabled)")

    # Data
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--train-dir", type=str, default="DIV2K_train_HR")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument("--runs-dir", type=str, default="results/runs",
                   help="base directory; each run gets a timestamped sub-folder")
    p.add_argument("--degradation", choices=("bicubic", "realistic"), default="realistic")

    # Model
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=96, help="LR-side patch; HR = patch * scale")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--scheduler-step", type=int, default=100, help="StepLR step_size in epochs")
    p.add_argument("--scheduler-gamma", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    # System
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true", help="Wrap model in torch.compile")
    p.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )

    # Logging / checkpoints
    p.add_argument("--val-every", type=int, default=5, help="run validation every N epochs")
    p.add_argument("--save-every", type=int, default=10, help="save periodic checkpoint every N epochs")
    p.add_argument("--log-every", type=int, default=50, help="print loss every N iterations")
    p.add_argument("--val-samples", type=int, default=5, help="number of val images to visualize at end")
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")

    # Quick smoke-test mode
    p.add_argument(
        "--quick",
        action="store_true",
        help="2 epochs, batch=2, num_workers=0, log every 5 iters — for end-to-end verification",
    )

    args = p.parse_args()

    if args.quick:
        args.epochs = 2
        args.batch_size = 2
        args.num_workers = 0
        args.val_every = 1
        args.save_every = 1
        args.log_every = 5
        args.val_samples = 2

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = make_run_dir(args)
    ckpt_dir = run_dir / "checkpoints"

    print("=" * 60)
    print("EDSR training")
    print("=" * 60)
    print(f"  device           : {device}")
    if device.type == "cuda":
        print(f"  cuda device      : {torch.cuda.get_device_name(0)}")
        print(f"  vram             : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  pytorch          : {torch.__version__}")
    print(f"  degradation      : {args.degradation}")
    print(f"  scale            : {args.scale}x")
    print(f"  batch / patch(LR): {args.batch_size} / {args.patch_size}")
    print(f"  epochs           : {args.epochs}")
    print(f"  compile          : {args.compile} (mode={args.compile_mode})")
    print(f"  run dir          : {run_dir}")
    print()

    # --- Datasets ---------------------------------------------------------
    train_set = SRDataset(
        hr_dir=Path(args.data_root) / args.train_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation=args.degradation,
        is_train=True,
    )
    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation=args.degradation,
        is_train=False,
    )
    print(f"  train images: {len(train_set)} | val: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # --- Model + compile --------------------------------------------------
    model = EDSR(
        scale_factor=args.scale,
        n_resblocks=args.n_resblocks,
        n_feats=args.n_feats,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    if args.compile:
        print(f"  compiling with mode='{args.compile_mode}' (first iter will be slow)...")
        model = torch.compile(model, mode=args.compile_mode)

    # --- Optim / loss / sched --------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma,
    )
    criterion = nn.L1Loss()

    # --- Resume -----------------------------------------------------------
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        print(f"  resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        load_state_dict(model, ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"  resumed at epoch {start_epoch} (best PSNR so far: {best_psnr:.2f} dB)")

    print()

    # --- Metric accumulators ---------------------------------------------
    all_epoch_nums: list[int] = []
    all_losses: list[float] = []
    val_epoch_nums: list[int] = []
    val_psnrs: list[float] = []
    val_ssims: list[float] = []

    # --- Training loop ----------------------------------------------------
    n_batches = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for it, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (it + 1) % args.log_every == 0:
                running = epoch_loss / (it + 1)
                print(f"  ep {epoch+1:3d} | iter {it+1:4d}/{n_batches} | loss {running:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"  ep {epoch+1:3d} done | avg loss {avg_loss:.4f} | "
            f"lr {cur_lr:.2e} | time {epoch_time:.1f}s"
        )

        all_epoch_nums.append(epoch + 1)
        all_losses.append(avg_loss)

        # Validation
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            psnr, ssim = evaluate_metrics(model, val_loader, device)
            is_best = psnr > best_psnr
            tag = "*** new best" if is_best else f"(best {best_psnr:.2f} dB)"
            print(
                f"  ep {epoch+1:3d} | val PSNR {psnr:.2f} dB | "
                f"SSIM {ssim:.4f} {tag}"
            )
            val_epoch_nums.append(epoch + 1)
            val_psnrs.append(psnr)
            val_ssims.append(ssim)

            if is_best:
                best_psnr = psnr
                save_checkpoint(
                    ckpt_dir / "best.pt", model, optimizer, scheduler,
                    epoch, best_psnr, args,
                )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch+1:03d}.pt",
                model, optimizer, scheduler, epoch, best_psnr, args,
            )

    # Final save
    save_checkpoint(
        ckpt_dir / "final.pt", model, optimizer, scheduler,
        args.epochs - 1, best_psnr, args,
    )

    # --- Post-training outputs -------------------------------------------
    print()
    print("=" * 60)
    print("Post-training analysis")
    print("=" * 60)

    save_metrics_csv(val_epoch_nums, all_losses, val_psnrs, val_ssims, run_dir)
    save_curves(all_epoch_nums, all_losses, val_epoch_nums, val_psnrs, val_ssims, run_dir)
    save_val_samples(model, val_set, run_dir, device, n_samples=args.val_samples)

    print()
    print(f"Training complete. Best val PSNR: {best_psnr:.2f} dB")
    print(f"Run artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
