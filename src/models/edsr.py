"""EDSR-baseline implementation (Lim et al., CVPR 2017).

Architecture overview
---------------------
::

    Input (B, 3, H, W) in [0, 1]
        |
        head: Conv 3x3, 3 -> n_feats
        |
        body: [ResBlock x n_resblocks] -> Conv 3x3, n_feats -> n_feats
        |  (long skip from head output)
        |
        tail: PixelShuffle upsampler -> Conv 3x3, n_feats -> 3
        |
    Output (B, 3, scale*H, scale*W)

Design choices
--------------
* **No BatchNorm anywhere.** See ``common.py`` for the rationale; this is the
  defining choice of EDSR vs. SRResNet.
* **PixelShuffle upsampling.** A Conv that expands channels by ``scale**2``
  followed by ``nn.PixelShuffle(scale)`` is the standard sub-pixel upsampler.
  It maps cleanly to ``DepthToSpace`` in ONNX and is supported on every edge
  runtime we target (ONNX Runtime, TensorRT, SNPE, NeuroPilot).
* **No input normalization layer.** Inputs are expected in ``[0, 1]``. We do
  not subtract a DIV2K mean inside the model so that the exported ONNX graph
  matches what the C++ runtime feeds it. Mean subtraction, if used, is folded
  into preprocessing.
* **Static shapes throughout.** No ``view(-1, ...)`` with dynamic dims, no
  ``F.interpolate(size=...)`` from runtime tensors, no ``Tensor.shape`` math
  inside ``forward``. PixelShuffle works with symbolic H/W under ONNX export.
* **Scale factor at construction time.** ``scale_factor`` is a Python int set
  in ``__init__`` and baked into the upsampler. The exported graph is
  scale-specific by design — we want one ONNX file per deployment scale.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.common import ResBlock
from src.quantization.fake_quant import CalibratingAdd


class _Upsampler(nn.Sequential):
    """Sub-pixel upsampler: Conv expand-channels then PixelShuffle.

    For ``scale == 2`` this is a single (Conv, PixelShuffle) pair, which is
    exactly what we want for the 2x baseline. For ``scale == 4`` we stack two
    2x stages (the standard EDSR recipe). Other scales are rejected to keep
    the export graph predictable.
    """

    def __init__(self, scale: int, n_feats: int) -> None:
        layers: list[nn.Module] = []
        if scale == 2 or scale == 3:
            layers.append(nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1))
            layers.append(nn.PixelShuffle(scale))
        elif scale == 4:
            for _ in range(2):
                layers.append(nn.Conv2d(n_feats, n_feats * 4, 3, padding=1))
                layers.append(nn.PixelShuffle(2))
        else:
            raise ValueError(f"Unsupported scale_factor={scale}; expected 2, 3, or 4.")
        super().__init__(*layers)


class EDSR(nn.Module):
    """EDSR-baseline.

    Default config (16 ResBlocks, 64 feats, 2x) yields ~1.5M parameters and
    matches the public EDSR-baseline checkpoint topology.

    Args:
        scale_factor: Upscaling factor. 2 (default), 3, or 4.
        n_resblocks: Number of residual blocks in the body.
        n_feats: Feature channel width.
        n_colors: Input/output channel count (3 for RGB).
        res_scale: Residual scale inside each ResBlock. Baseline uses 1.0.

    Shape:
        Input:  (B, n_colors, H, W), float in [0, 1].
        Output: (B, n_colors, scale_factor*H, scale_factor*W).
    """

    def __init__(
        self,
        scale_factor: int = 2,
        n_resblocks: int = 16,
        n_feats: int = 64,
        n_colors: int = 3,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.n_feats = n_feats

        self.head = nn.Conv2d(n_colors, n_feats, 3, padding=1)

        body: list[nn.Module] = [
            ResBlock(n_feats, kernel_size=3, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body)

        self.upsampler = _Upsampler(scale_factor, n_feats)
        self.tail = nn.Conv2d(n_feats, n_colors, 3, padding=1)

        # Long skip Add as a module so it joins the per-layer sensitivity
        # sweep. In 'fp32' mode (default) this is a plain x + res, so the
        # exported ONNX graph is byte-identical to the un-wrapped model.
        self.long_skip_add = CalibratingAdd()

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(x)
        res = self.body(x)
        x = self.long_skip_add(x, res)  # long skip; module-ized for quant sweep
        x = self.upsampler(x)
        x = self.tail(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    model = EDSR(scale_factor=2, n_resblocks=16, n_feats=64).eval()

    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = model(dummy)

    expected_shape = (1, 3, 128, 128)
    assert tuple(out.shape) == expected_shape, (
        f"Output shape mismatch: got {tuple(out.shape)}, expected {expected_shape}"
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("EDSR-baseline sanity check: OK")
    print(f"  input shape    : {tuple(dummy.shape)}")
    print(f"  output shape   : {tuple(out.shape)}")
    print(f"  output range   : [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"  total params   : {n_params:,}")
    print(f"  trainable      : {n_trainable:,}")
