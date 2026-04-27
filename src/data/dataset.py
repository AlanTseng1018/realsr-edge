"""SR training/validation dataset.

Reads HR images from a directory, samples a random patch each call, and
synthesizes a paired LR via either pure bicubic (Track A) or the realistic
degradation pipeline (Track B). Returns float32 ``(3, H, W)`` tensors in
``[0, 1]`` for both LR and HR.

Why a custom LR generator instead of calling
``RealisticDegradation.random_degradation_pipeline`` directly
---------------------------------------------------------------
The random pipeline applies the downsample step with probability 0.5 (so the
output may be HR-sized or LR-sized depending on the coin flip). That is fine
for visualization but breaks DataLoader batching — every sample in a batch
must have the same shape. Here we **always** downsample by ``scale``, while
the auxiliary augmentations (blur, banding, noise, JPEG) still fire
independently with probability 0.5 each. This keeps tensor shapes
deterministic while preserving the augmentation diversity.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.degradation import RealisticDegradation


def _to_chw_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    """``(H, W, 3) uint8`` -> ``(3, H, W) float32`` in ``[0, 1]``."""
    return torch.from_numpy(img_uint8).permute(2, 0, 1).contiguous().float() / 255.0


class SRDataset(Dataset):
    """Patch-based SR dataset.

    Args:
        hr_dir: Directory of HR images (``.png`` preferred, ``.jpg`` fallback).
        scale: Upscaling factor; LR side is ``hr_patch_size // scale``.
        hr_patch_size: HR-side patch edge length in pixels. Must divide by
            ``scale`` exactly.
        degradation: ``"realistic"`` (Track B) uses the
            :class:`RealisticDegradation` pipeline (always downsamples);
            ``"bicubic"`` (Track A) uses pure bicubic downsample only.
        is_train: If True, random crop + horizontal flip + 90-deg rotations.
            If False, deterministic center crop and no augmentation.
        repeat: Virtual length multiplier — ``len(self) == len(images) * repeat``.
            Useful so PyTorch sees a "full epoch" without re-reading the same
            image too many times in succession (also lets you run shorter
            epochs with the same scheduler logic).
    """

    def __init__(
        self,
        hr_dir: str | Path,
        scale: int = 2,
        hr_patch_size: int = 192,
        degradation: str = "realistic",
        is_train: bool = True,
        repeat: int = 1,
    ) -> None:
        super().__init__()
        self.hr_dir = Path(hr_dir)
        if not self.hr_dir.is_dir():
            raise FileNotFoundError(f"HR directory not found: {self.hr_dir}")

        self.hr_paths = sorted(self.hr_dir.glob("*.png"))
        if not self.hr_paths:
            self.hr_paths = sorted(self.hr_dir.glob("*.jpg"))
        if not self.hr_paths:
            raise FileNotFoundError(f"No .png or .jpg files in {self.hr_dir}")

        if hr_patch_size % scale != 0:
            raise ValueError(
                f"hr_patch_size ({hr_patch_size}) must be divisible by scale ({scale})"
            )
        if degradation not in ("bicubic", "realistic"):
            raise ValueError(
                f"degradation must be 'bicubic' or 'realistic', got {degradation!r}"
            )

        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.degradation = degradation
        self.is_train = is_train
        self.repeat = repeat
        self._deg = RealisticDegradation()

    def __len__(self) -> int:
        return len(self.hr_paths) * self.repeat

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.hr_paths[idx % len(self.hr_paths)]
        hr_full = cv2.imread(str(path))
        if hr_full is None:
            raise RuntimeError(f"cv2 failed to read {path}")
        hr_full = cv2.cvtColor(hr_full, cv2.COLOR_BGR2RGB)

        h, w = hr_full.shape[:2]
        ps = self.hr_patch_size
        if h < ps or w < ps:
            raise ValueError(
                f"Image {path.name} ({h}x{w}) smaller than patch {ps}x{ps}"
            )

        if self.is_train:
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            hr = hr_full[y:y + ps, x:x + ps]
            if random.random() < 0.5:
                hr = hr[:, ::-1]
            k = random.randint(0, 3)
            if k:
                hr = np.rot90(hr, k)
            hr = np.ascontiguousarray(hr)
        else:
            cy, cx = h // 2, w // 2
            hr = hr_full[cy - ps // 2:cy + ps // 2, cx - ps // 2:cx + ps // 2].copy()

        if self.degradation == "bicubic":
            lr = cv2.resize(
                hr, (self.lr_patch_size, self.lr_patch_size),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            # Realistic LR: blur (50%) -> always downsample -> banding/noise/JPEG (50% each)
            # For val we use a deterministic seed per index so PSNR is comparable
            # across epochs.
            seed = None if self.is_train else idx
            lr = self._make_realistic_lr(hr, seed=seed)

        return _to_chw_tensor(lr), _to_chw_tensor(hr)

    def _make_realistic_lr(self, hr: np.ndarray, seed: int | None) -> np.ndarray:
        """Apply (probabilistic) augmentations + always downsample.

        The augmentation steps mirror :meth:`RealisticDegradation.random_degradation_pipeline`
        but with downsample promoted from probabilistic to deterministic.
        """
        rng = np.random.default_rng(seed)
        out = hr
        if rng.random() < 0.5:
            out = self._deg.apply_blur(out, random_state=rng)
        out = self._deg.apply_downsample(out, scale=self.scale, random_state=rng)
        if rng.random() < 0.5:
            out = self._deg.apply_banding(out, random_state=rng)
        if rng.random() < 0.5:
            out = self._deg.apply_noise(out, random_state=rng)
        if rng.random() < 0.5:
            out = self._deg.apply_jpeg_compression(out, random_state=rng)
        return out


if __name__ == "__main__":
    # Quick smoke test against DIV2K
    train_set = SRDataset(
        hr_dir="data/DIV2K/DIV2K_train_HR",
        scale=2,
        hr_patch_size=192,
        degradation="realistic",
        is_train=True,
    )
    val_set = SRDataset(
        hr_dir="data/DIV2K/DIV2K_valid_HR",
        scale=2,
        hr_patch_size=192,
        degradation="realistic",
        is_train=False,
    )
    print(f"train: {len(train_set)} samples | val: {len(val_set)} samples")

    lr, hr = train_set[0]
    print(f"LR: {tuple(lr.shape)} {lr.dtype} range=[{lr.min():.3f}, {lr.max():.3f}]")
    print(f"HR: {tuple(hr.shape)} {hr.dtype} range=[{hr.min():.3f}, {hr.max():.3f}]")

    # Determinism: val[0] called twice should be identical
    a_lr, a_hr = val_set[0]
    b_lr, b_hr = val_set[0]
    assert torch.equal(a_lr, b_lr) and torch.equal(a_hr, b_hr), (
        "Validation samples should be deterministic (same idx -> same data)."
    )
    print("Validation determinism: OK")
