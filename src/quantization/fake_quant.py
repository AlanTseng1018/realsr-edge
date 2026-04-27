"""Minimal symmetric INT8 fake-quantization primitives + Conv2d wrapper.

"Fake-quant" = quantize-then-dequantize in float, simulating the precision
loss that would occur in a real INT8 deploy without actually running an
INT8 backend (no fbgemm / qnnpack / ONNX Runtime required). This is the
standard PTQ analysis path: it gives you the **accuracy** numbers without
the **latency** numbers, which is exactly what we need for §3.4 sensitivity
analysis.

Scheme choices (kept deliberately simple for V1 -- can be upgraded later):
  * Activations: **symmetric per-tensor, INT8 (range -128..127)**.
    Calibrated by tracking per-tensor max-abs over a small set of inputs.
    Symmetric is suboptimal for ReLU outputs (long tail biased to positive),
    but it matches what most edge runtimes use as the simplest path. We
    document this gap and may revisit with asymmetric / percentile-clipped
    schemes in a follow-up.
  * Weights: **symmetric per-output-channel, INT8**. Per-channel is the
    standard for convolution weights -- handles the common case where
    different output channels have very different dynamic range. This is
    what TensorRT, ONNX Runtime, SNPE, and NeuroPilot all do by default.

Calibration uses **plain max-abs** rather than percentile clipping or
KL-divergence -- max-abs is the worst case but the simplest. Percentile /
KL can be added on top of this scaffolding when we need them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Quant-dequant primitives
# ---------------------------------------------------------------------------

INT8_QMIN, INT8_QMAX = -128, 127


def quantize_dequantize_per_tensor(
    x: Tensor,
    scale: Tensor,
    qmin: int = INT8_QMIN,
    qmax: int = INT8_QMAX,
) -> Tensor:
    """Symmetric per-tensor q-dq. ``scale`` is a 0-d tensor."""
    q = (x / scale).round().clamp(qmin, qmax)
    return q * scale


def quantize_dequantize_per_channel(
    w: Tensor,
    scale: Tensor,
    ch_axis: int = 0,
    qmin: int = INT8_QMIN,
    qmax: int = INT8_QMAX,
) -> Tensor:
    """Symmetric per-channel q-dq. ``scale`` has shape ``(n_channels,)``.

    ``ch_axis`` is the axis of ``w`` along which channels run (0 for Conv2d
    weights with shape ``(out_ch, in_ch, kH, kW)``).
    """
    shape = [1] * w.ndim
    shape[ch_axis] = -1
    s = scale.view(shape)
    q = (w / s).round().clamp(qmin, qmax)
    return q * s


def per_tensor_scale(amax: Tensor, qmax: int = INT8_QMAX) -> Tensor:
    """Convert max-abs to symmetric quantization scale."""
    return amax.clamp(min=1e-8) / qmax


def per_channel_scale(w: Tensor, ch_axis: int = 0, qmax: int = INT8_QMAX) -> Tensor:
    """Per-channel symmetric scale of a weight tensor."""
    dims = [d for d in range(w.ndim) if d != ch_axis]
    amax = w.detach().abs().amax(dim=dims)
    return amax.clamp(min=1e-8) / qmax


# ---------------------------------------------------------------------------
# Conv2d wrapper
# ---------------------------------------------------------------------------

class CalibratingConv2d(nn.Module):
    """Drop-in wrapper around ``nn.Conv2d`` with three modes.

    * ``mode='fp32'``: pass-through, identical to the original conv. Default.
    * ``mode='calibrate'``: forward through the original conv, but observe
      the input tensor's max-abs so we can pick an INT8 scale later.
    * ``mode='quantize'``: fake-quantize input (per-tensor) and weight
      (per-output-channel), then run the conv. The activation scale comes
      from whatever was observed during ``'calibrate'`` mode -- if the
      module never saw calibration data, ``set_mode('quantize')`` will
      raise.

    The wrapper holds a reference to the original ``nn.Conv2d`` -- weights
    are not copied, so checkpoints don't drift between the FP32 and the
    wrapped versions.
    """

    _VALID_MODES = ("fp32", "calibrate", "quantize")

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv
        # Buffers so they move with .to(device) and survive state_dict round-trips
        self.register_buffer("input_amax", torch.tensor(0.0))
        self.register_buffer("calibrated", torch.tensor(False))
        self.mode: str = "fp32"

    def set_mode(self, mode: str) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be in {self._VALID_MODES}, got {mode!r}")
        if mode == "quantize" and not bool(self.calibrated):
            raise RuntimeError(
                "Cannot enter 'quantize' mode before calibration. "
                "Run a calibration pass with mode='calibrate' first."
            )
        self.mode = mode

    def reset_calibration(self) -> None:
        self.input_amax.zero_()
        self.calibrated.fill_(False)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "fp32":
            return self.conv(x)

        if self.mode == "calibrate":
            with torch.no_grad():
                cur = x.detach().abs().amax()
                self.input_amax.copy_(torch.maximum(self.input_amax, cur))
                self.calibrated.fill_(True)
            return self.conv(x)

        # mode == 'quantize'
        a_scale = per_tensor_scale(self.input_amax)
        x_q = quantize_dequantize_per_tensor(x, a_scale)
        w_scale = per_channel_scale(self.conv.weight, ch_axis=0)
        w_q = quantize_dequantize_per_channel(self.conv.weight, w_scale, ch_axis=0)
        return F.conv2d(
            x_q, w_q, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    def extra_repr(self) -> str:
        return f"mode={self.mode}, calibrated={bool(self.calibrated)}"


# ---------------------------------------------------------------------------
# Helpers to wrap / unwrap a model in-place
# ---------------------------------------------------------------------------

def wrap_convs(model: nn.Module) -> dict[str, CalibratingConv2d]:
    """Replace every ``nn.Conv2d`` in ``model`` with a :class:`CalibratingConv2d`.

    Mutates ``model`` in place. Returns ``{dotted_name -> wrapper}`` for
    convenient targeting (e.g. for per-layer sensitivity sweeps).

    Implementation note: we collect ALL ``(parent, child_name, child)`` tuples
    in a first pass, then mutate in a second pass. Mutating during traversal
    causes infinite recursion -- because each new wrapper contains the
    original conv, the traversal would descend into the new wrapper, find the
    original conv inside, wrap that too, and so on.
    """
    pairs: list[tuple[nn.Module, str, nn.Conv2d, str]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Conv2d) and not isinstance(child, CalibratingConv2d):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                pairs.append((parent, child_name, child, full))

    wrappers: dict[str, CalibratingConv2d] = {}
    for parent, child_name, child, full in pairs:
        wrapper = CalibratingConv2d(child)
        setattr(parent, child_name, wrapper)
        wrappers[full] = wrapper
    return wrappers


def set_all_modes(wrappers: dict[str, CalibratingConv2d], mode: str) -> None:
    for w in wrappers.values():
        w.set_mode(mode)


def reset_all_calibration(wrappers: dict[str, CalibratingConv2d]) -> None:
    for w in wrappers.values():
        w.reset_calibration()
