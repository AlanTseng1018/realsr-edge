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

    python -m src.training.train --resume results/checkpoints/edsr_baseline/best.pt

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
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset
from src.models.edsr import EDSR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_dict_for_save(model: nn.Module) -> dict[str, Any]:
    """Return underlying ``state_dict`` even when ``model`` is ``torch.compile``-wrapped."""
    return getattr(model, "_orig_mod", model).state_dict()


def load_state_dict(model: nn.Module, state: dict[str, Any]) -> None:
    """Load ``state`` into ``model`` regardless of whether it's compiled."""
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(state)


@torch.no_grad()
def evaluate_psnr(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mean per-image PSNR (dB) over a validation loader. Inputs assumed in [0, 1]."""
    model.eval()
    psnr_sum, count = 0.0, 0
    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        sr = model(lr).clamp(0.0, 1.0)
        mse = ((sr - hr) ** 2).mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
        psnr_sum += psnr.sum().item()
        count += psnr.numel()
    return psnr_sum / max(count, 1)


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDSR training (torch.compile-enabled)")

    # Data
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--train-dir", type=str, default="DIV2K_train_HR")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument("--output-dir", type=str, default="results/checkpoints/edsr_baseline")
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

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  output           : {output_dir}")
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

        # Validation
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            psnr = evaluate_psnr(model, val_loader, device)
            is_best = psnr > best_psnr
            tag = "*** new best" if is_best else f"(best {best_psnr:.2f} dB)"
            print(f"  ep {epoch+1:3d} | val PSNR {psnr:.2f} dB {tag}")
            if is_best:
                best_psnr = psnr
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, scheduler, epoch, best_psnr, args,
                )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch+1:03d}.pt",
                model, optimizer, scheduler, epoch, best_psnr, args,
            )

    # Final save
    save_checkpoint(
        output_dir / "final.pt", model, optimizer, scheduler,
        args.epochs - 1, best_psnr, args,
    )
    print()
    print(f"Training complete. Best val PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints in: {output_dir}")


if __name__ == "__main__":
    main()
