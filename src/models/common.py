"""Shared building blocks for SR models.

Design notes
------------
* **No BatchNorm.** Following EDSR (Lim et al., CVPR 2017), BN is removed because
  per-batch statistics distort feature scale in low-level vision tasks and hurt
  PSNR. Removing BN also widens the activation dynamic range, which is the root
  cause of SR INT8 quantization difficulty (this is what we want to study).
* **Static shapes only.** All operators here are pure conv / element-wise add /
  ReLU. Nothing depends on the input spatial size at trace time, so the block
  exports cleanly to ONNX with a fixed or dynamic spatial axis.
* **Residual scaling.** EDSR uses a small constant (~0.1) on the residual branch
  to stabilize training of deep stacks. Default 1.0 keeps the baseline behavior.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResBlock(nn.Module):
    """Residual block: Conv -> ReLU -> Conv, plus skip connection.

    Args:
        n_channels: Number of feature channels (kept constant in/out).
        kernel_size: Conv kernel size. Default 3 (same as EDSR).
        res_scale: Multiplier on the residual branch before the skip add.
            EDSR-baseline uses 1.0; deeper EDSR variants use 0.1.
    """

    def __init__(
        self,
        n_channels: int,
        kernel_size: int = 3,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=True)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(self.relu(self.conv1(x)))
        if self.res_scale != 1.0:
            residual = residual * self.res_scale
        return x + residual
