"""Shared ONNX FP32+INT8 inference runner for spatial heatmap scripts.

Used by both :mod:`src.quantization.lpips_heatmap` and
:mod:`src.quantization.structural_heatmap` to drive their Stage-3
(deployment-check) execution path -- producing heatmaps from real ONNX
inference output rather than PyTorch fake-quant simulation.

The aggregate LPIPS / PSNR / latency in §3.2 already comes from ONNX via
``benchmark_onnx.py``; this runner extends the same backend coverage to
the SPATIAL detectors that previously only ran on PyTorch fake-quant.

Provider order: CUDA -> CPU. TensorRT EP is intentionally skipped because
it triggers JIT engine builds on first call (slow, memory-heavy) and its
QDQ behaviour differs from CUDA / CPU EPs in ways §3.3 characterises
separately. For per-image heatmap inspection we want "what the deployed
ONNX actually outputs on this hardware", which CUDA EP delivers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxSRRunner:
    """FP32 + INT8 ONNX inference returning numpy SR tensors on host."""

    def __init__(
        self,
        fp32_onnx: Path,
        int8_onnx: Path,
        providers: list[str] | None = None,
    ) -> None:
        if providers is None:
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
        self.providers = providers
        self.fp32_sess = ort.InferenceSession(str(fp32_onnx), providers=providers)
        self.int8_sess = ort.InferenceSession(str(int8_onnx), providers=providers)
        self._fp32_in = self.fp32_sess.get_inputs()[0].name
        self._fp32_out = self.fp32_sess.get_outputs()[0].name
        self._int8_in = self.int8_sess.get_inputs()[0].name
        self._int8_out = self.int8_sess.get_outputs()[0].name

    @staticmethod
    def _to_numpy(lr) -> np.ndarray:
        """Accept torch tensor or numpy; return float32 numpy of shape (1,3,H,W)."""
        if hasattr(lr, "detach"):
            return lr.detach().cpu().numpy().astype(np.float32)
        return np.asarray(lr, dtype=np.float32)

    def run_fp32(self, lr) -> np.ndarray:
        """LR (1,3,H,W) float[0,1] -> SR (1,3,sH,sW) float, clipped to [0,1]."""
        lr_np = self._to_numpy(lr)
        sr = self.fp32_sess.run([self._fp32_out], {self._fp32_in: lr_np})[0]
        return np.clip(sr, 0.0, 1.0).astype(np.float32)

    def run_int8(self, lr) -> np.ndarray:
        lr_np = self._to_numpy(lr)
        sr = self.int8_sess.run([self._int8_out], {self._int8_in: lr_np})[0]
        return np.clip(sr, 0.0, 1.0).astype(np.float32)

    def describe(self) -> str:
        return f"OnnxSRRunner(providers={self.providers})"
