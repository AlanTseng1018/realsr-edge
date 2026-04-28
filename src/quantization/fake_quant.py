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
      input statistics. We track BOTH the running max-abs (cheap, online)
      AND an online histogram with dynamic-range rescaling (TensorRT-style).
      Either statistic can drive the final ``input_amax`` later via
      :meth:`apply_calibration_method`.
    * ``mode='quantize'``: fake-quantize input (per-tensor) and weight
      (per-output-channel), then run the conv. The activation scale comes
      from ``self.input_amax``, which is set during calibration (default to
      max-abs running) or overridden via :meth:`apply_calibration_method`.

    The wrapper holds a reference to the original ``nn.Conv2d`` -- weights
    are not copied, so checkpoints don't drift between the FP32 and the
    wrapped versions.

    Calibration buffers
    -------------------
    * ``_running_max_abs``: online max of ``|x|`` seen during calibration.
    * ``hist`` + ``hist_max``: online histogram of ``|x|`` with bins covering
      ``[0, hist_max]``. When a new batch's max exceeds the current
      ``hist_max``, the existing bins are rescaled into a wider range; the
      total mass is preserved.
    * ``input_amax``: the FINAL scale source consumed by the quantize path.
      During calibration it tracks max-abs running; after calibration the
      caller can switch it to a percentile-based value via
      :meth:`apply_calibration_method`.
    """

    _VALID_MODES = ("fp32", "calibrate", "quantize")

    def __init__(self, conv: nn.Conv2d, n_bins: int = 2048) -> None:
        super().__init__()
        self.conv = conv
        self.n_bins = n_bins

        # Match the wrapped conv's device so buffers don't end up on CPU
        # when wrap_convs is called after model.to(device). 0-d scalar
        # buffers happen to tolerate device mismatches via PyTorch's
        # scalar-broadcast leniency, but the histogram (1-d, 2048 elems)
        # does not -- ``scatter_add_`` strictly checks device.
        device = next(conv.parameters()).device

        # Final scale used in quantize mode (set during calibrate; can be
        # overridden later via apply_calibration_method).
        self.register_buffer("input_amax", torch.tensor(0.0, device=device))

        # Raw stats accumulated during calibrate mode.
        self.register_buffer("_running_max_abs", torch.tensor(0.0, device=device))
        self.register_buffer("hist", torch.zeros(n_bins, device=device))
        self.register_buffer("hist_max", torch.tensor(0.0, device=device))

        self.register_buffer("calibrated", torch.tensor(False, device=device))
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
        self._running_max_abs.zero_()
        self.hist.zero_()
        self.hist_max.zero_()
        self.calibrated.fill_(False)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "fp32":
            return self.conv(x)

        if self.mode == "calibrate":
            with torch.no_grad():
                abs_x = x.detach().abs()
                cur_max = abs_x.amax()
                self._running_max_abs.copy_(torch.maximum(self._running_max_abs, cur_max))
                self._update_histogram(abs_x)
                # Default scale = running max-abs; apply_calibration_method
                # can swap this to percentile-based later.
                self.input_amax.copy_(self._running_max_abs)
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

    # ------------------------------------------------------------------
    # Histogram bookkeeping (TensorRT-style with rescaling)
    # ------------------------------------------------------------------

    def _update_histogram(self, abs_x: Tensor) -> None:
        """Add ``abs_x`` into the running histogram, expanding range if needed.

        Strategy: when the incoming batch's max exceeds the histogram's
        current upper bound, re-bin the existing counts into a wider range
        before adding the new contribution. Adds 10% headroom on rescale
        to amortize away frequent re-binning across many small jumps.
        """
        cur_max = abs_x.amax().item()
        cur_hist_max = self.hist_max.item()

        if cur_hist_max == 0.0:
            self._rescale_histogram(max(cur_max, 1e-8))
        elif cur_max > cur_hist_max:
            self._rescale_histogram(max(cur_max * 1.1, cur_hist_max * 1.5))

        # Add new batch contribution. We avoid ``torch.histc`` because on
        # some CUDA configurations it returns a CPU tensor and then fails
        # the device-mismatched ``add_``. ``scatter_add_`` is guaranteed
        # device-preserving.
        flat = abs_x.flatten().float()
        hmax = self.hist_max.item()
        if hmax > 0.0:
            indices = (flat / hmax * self.n_bins).long().clamp_(0, self.n_bins - 1)
            ones = torch.ones_like(flat)
            self.hist.scatter_add_(0, indices, ones)

    def _rescale_histogram(self, new_max: float) -> None:
        """Re-bin the existing histogram from [0, hist_max] into [0, new_max]."""
        old_max = self.hist_max.item()
        if old_max == 0.0:
            self.hist_max.fill_(new_max)
            self.hist.zero_()
            return

        # Map each old bin's center to a new bin index, then sum-reduce.
        n = self.n_bins
        device = self.hist.device
        old_centers = (
            torch.arange(n, dtype=torch.float32, device=device) + 0.5
        ) * (old_max / n)
        new_indices = (old_centers / (new_max / n)).long().clamp(0, n - 1)
        rescaled = torch.zeros_like(self.hist)
        rescaled.index_add_(0, new_indices, self.hist)
        self.hist.copy_(rescaled)
        self.hist_max.fill_(new_max)

    def percentile_from_hist(self, percentile: float) -> float:
        """Return the activation magnitude at ``percentile`` of the cumulative
        histogram. ``percentile`` is a fraction in (0, 1], e.g. 0.999."""
        if not 0.0 < percentile <= 1.0:
            raise ValueError(f"percentile must be in (0, 1], got {percentile}")
        total = self.hist.sum().item()
        if total <= 0.0:
            return 0.0

        cdf = self.hist.cumsum(0) / total
        target = (cdf >= percentile).nonzero()
        bin_width = self.hist_max.item() / self.n_bins
        if len(target) == 0:
            return self.hist_max.item()
        idx = int(target[0].item())

        # Linear interpolation within the bin for sub-bin precision.
        prev_cdf = cdf[idx - 1].item() if idx > 0 else 0.0
        cur_cdf = cdf[idx].item()
        if cur_cdf <= prev_cdf:
            return (idx + 0.5) * bin_width
        bin_start = idx * bin_width
        frac = (percentile - prev_cdf) / (cur_cdf - prev_cdf)
        return bin_start + frac * bin_width

    def apply_calibration_method(
        self,
        method: str = "max-abs",
        percentile: float = 0.999,
    ) -> None:
        """Set ``self.input_amax`` based on choice of calibration method.

        ``method='max-abs'``  -> use the running max-abs from calibrate
        ``method='percentile'`` -> use ``self.percentile_from_hist(percentile)``

        Must be called after at least one calibration pass.
        """
        if not bool(self.calibrated):
            raise RuntimeError(
                "apply_calibration_method called before any calibration pass."
            )
        if method == "max-abs":
            self.input_amax.copy_(self._running_max_abs)
        elif method == "percentile":
            v = self.percentile_from_hist(percentile)
            self.input_amax.fill_(v)
        else:
            raise ValueError(
                f"Unknown calibration method: {method!r}. "
                f"Expected 'max-abs' or 'percentile'."
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


def apply_calibration_to_all(
    wrappers: dict[str, CalibratingConv2d],
    method: str = "max-abs",
    percentile: float = 0.999,
) -> None:
    """Update ``input_amax`` on every wrapper using the chosen calibration method.

    See :meth:`CalibratingConv2d.apply_calibration_method`. Common usage:
    after a single calibration pass that collected both max-abs and
    histogram statistics, call this once per scheme to flip the entire
    model between calibration variants for ablation.
    """
    for w in wrappers.values():
        w.apply_calibration_method(method=method, percentile=percentile)
