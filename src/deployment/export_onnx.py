"""Export a trained EDSR checkpoint to ONNX, then verify against PyTorch.

This is the **first deployment milestone**. Without a verified ONNX file,
nothing downstream works:
  * The C++ inference reference (cpp_inference/) needs an ONNX file
  * ONNX Runtime quantization (PTQ static / dynamic) starts from FP32 ONNX
  * Vendor NPU compilers (TensorRT / SNPE / NeuroPilot / ...) take ONNX as input

Verification is integrated into this script so the deploy chain has a
**byte-level numeric ground truth** at the very first export. If PyTorch
and ONNX RT disagree by more than the documented tolerance on the same
input, every downstream PSNR / latency number is suspect, so we fail
loudly here rather than silently propagate corruption.

Run examples
------------
::

    # Export FP32 ONNX from the smoke-trained checkpoint, verify, exit
    python -m src.deployment.export_onnx \
        --checkpoint results/checkpoints/edsr_baseline/final.pt \
        --output results/onnx_models/edsr_fp32.onnx

    # As above but skip the verification step (faster, NOT recommended)
    python -m src.deployment.export_onnx \
        --checkpoint results/checkpoints/edsr_baseline/final.pt \
        --output results/onnx_models/edsr_fp32.onnx \
        --skip-verify

    # Export with fixed dummy shape (1x3x96x96) but allow dynamic H/W at runtime
    # via --dynamic-axes (default: enabled)
    python -m src.deployment.export_onnx \
        --checkpoint results/checkpoints/edsr_baseline/final.pt \
        --output results/onnx_models/edsr_fp32.onnx \
        --dummy-h 96 --dummy-w 96
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.edsr import EDSR


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(
    checkpoint_path: Path,
    scale: int,
    n_resblocks: int,
    n_feats: int,
    device: torch.device,
) -> torch.nn.Module:
    model = EDSR(scale_factor=scale, n_resblocks=n_resblocks, n_feats=n_feats)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    dummy_input: torch.Tensor,
    opset: int,
    dynamic_axes: bool,
) -> None:
    """Export `model` to ONNX at `output_path`.

    ``dynamic_axes=True`` makes batch / H / W flexible at inference time
    (model can take any spatial size as long as it's a multiple of `scale`).
    Without this, the ONNX file is locked to whatever shape ``dummy_input``
    had at export time -- a common silent gotcha.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dyn_axes = None
    if dynamic_axes:
        dyn_axes = {
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height_out", 3: "width_out"},
        }

    print(f"  exporting (opset={opset}, dynamic_axes={dynamic_axes}) ...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dyn_axes,
    )

    # Re-load the saved file and run ONNX's own structural / shape-inference
    # checker. This catches corruption right after write and gives a clearer
    # error than waiting for ONNX RT to fail at session creation.
    print("  running onnx.checker.check_model ...")
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto, full_check=True)


# ---------------------------------------------------------------------------
# Verify (PyTorch vs ONNX Runtime)
# ---------------------------------------------------------------------------

def verify_onnx_matches_pytorch(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    sample_shapes: list[tuple[int, int, int, int]],
    device: torch.device,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    """Run identical inputs through PyTorch and ONNX Runtime; compare outputs.

    We test multiple input shapes to confirm dynamic axes survived the export.
    Tolerance ``atol=1e-4`` / ``rtol=1e-3`` is the ONNX RT documented level
    for FP32; if the model can't meet it, something is wrong (bad export, op
    mismatch, etc.).
    """
    # Pick the best provider available -- prefer CUDA if torch was on CUDA
    providers: list[str] = []
    if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    print(f"  ORT providers: {providers}")

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    used_provider = sess.get_providers()[0]
    print(f"  ORT chose: {used_provider}")

    pytorch_model.eval()
    rng = np.random.default_rng(0)
    failures: list[str] = []
    for i, shape in enumerate(sample_shapes):
        x_np = rng.random(size=shape, dtype=np.float32)
        x_pt = torch.from_numpy(x_np).to(device)

        with torch.no_grad():
            y_pt = pytorch_model(x_pt).cpu().numpy()

        y_ort = sess.run(["output"], {"input": x_np})[0]

        max_abs = float(np.max(np.abs(y_pt - y_ort)))
        max_rel = float(np.max(np.abs(y_pt - y_ort) / (np.abs(y_pt) + 1e-8)))
        ok = np.allclose(y_pt, y_ort, atol=atol, rtol=rtol)
        tag = "OK" if ok else "FAIL"
        print(
            f"  [shape {shape}] {tag}  "
            f"max|diff|={max_abs:.2e}  max rel={max_rel:.2e}  "
            f"out shape={y_ort.shape}"
        )
        if not ok:
            failures.append(f"shape={shape}: max_abs={max_abs:.2e}")

    if failures:
        raise RuntimeError(
            "PyTorch and ONNX RT outputs disagree beyond tolerance:\n  - "
            + "\n  - ".join(failures)
        )
    print(f"  numeric match: PASS ({len(sample_shapes)} shapes within atol={atol})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EDSR checkpoint to ONNX (FP32) and verify.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True, help="output .onnx path")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)
    p.add_argument("--dummy-h", type=int, default=96, help="dummy input height for tracing")
    p.add_argument("--dummy-w", type=int, default=96, help="dummy input width for tracing")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (PixelShuffle is fine from 11+)")
    p.add_argument(
        "--no-dynamic-axes",
        action="store_true",
        help="lock the ONNX to the dummy shape (NOT recommended -- breaks variable-size deploy)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device for the PyTorch reference run during verification",
    )
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_path = Path(args.output)

    print("=" * 60)
    print("EDSR ONNX export")
    print("=" * 60)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  output     : {output_path}")
    print(f"  device     : {device}  (for PyTorch reference)")
    print(f"  scale      : {args.scale}x")
    print(f"  dummy shape: (1, 3, {args.dummy_h}, {args.dummy_w})")
    print(f"  opset      : {args.opset}")
    print(f"  dynamic    : {not args.no_dynamic_axes}")
    print()

    print("Building PyTorch model from checkpoint ...")
    model = build_model_from_checkpoint(
        Path(args.checkpoint), args.scale, args.n_resblocks, args.n_feats, device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")
    print()

    print(f"Exporting to {output_path} ...")
    dummy_input = torch.randn(1, 3, args.dummy_h, args.dummy_w, device=device)
    export_onnx(
        model=model,
        output_path=output_path,
        dummy_input=dummy_input,
        opset=args.opset,
        dynamic_axes=not args.no_dynamic_axes,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {output_path} ({size_mb:.2f} MB)")
    print()

    if args.skip_verify:
        print("Skipping verification (--skip-verify).")
        return

    print("Verifying PyTorch vs ONNX Runtime numerics ...")
    # Test multiple shapes to exercise dynamic axes -- including a non-square one
    sample_shapes: list[tuple[int, int, int, int]] = [
        (1, 3, args.dummy_h, args.dummy_w),
        (1, 3, 64, 64),
        (1, 3, 128, 64),
        (2, 3, args.dummy_h, args.dummy_w),
    ] if not args.no_dynamic_axes else [(1, 3, args.dummy_h, args.dummy_w)]

    t0 = time.time()
    verify_onnx_matches_pytorch(
        pytorch_model=model,
        onnx_path=output_path,
        sample_shapes=sample_shapes,
        device=device,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(f"  verification time: {time.time() - t0:.1f}s")
    print()
    print("Done. Export + verify complete.")


if __name__ == "__main__":
    main()
