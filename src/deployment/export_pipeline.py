"""Multi-precision ONNX export pipeline.

Produces three ONNX artifacts from a single PyTorch checkpoint, all written
into one timestamped output folder so the run is identifiable as a single
"export node" in the deployment pipeline:

* ``edsr_fp32.onnx``         -- direct ``torch.onnx.export`` of FP32 weights
* ``edsr_fp16.onnx``         -- FP32 ONNX converted to FP16 via
                                ``onnxconverter-common``
* ``edsr_int8_static.onnx``  -- FP32 ONNX statically quantized via
                                ``onnxruntime.quantization.quantize_static``,
                                with calibration drawn from the val set

Each ONNX is verified against the PyTorch model on multiple input shapes,
with a precision-appropriate tolerance:

============= =====================
Precision     Tolerance (atol)
============= =====================
FP32          1e-4 (near bit-level)
FP16          5e-2
INT8          1e-1
============= =====================

Output folder structure::

    results/onnx_exports/<run_name>/
    ├── README.md             -- human-readable summary of what's here
    ├── metadata.json         -- machine-readable: paths, sizes, mtimes
    ├── verification.md       -- per-format PyTorch-vs-ORT diff results
    ├── edsr_fp32.onnx
    ├── edsr_fp16.onnx
    └── edsr_int8_static.onnx

Run example::

    python -m src.deployment.export_pipeline \\
        --checkpoint results/runs/.../checkpoints/best.pt \\
        --output-dir results/onnx_exports/edsr_200ep \\
        --calib-samples 64
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from src.data.dataset import SRDataset
from src.models.edsr import EDSR


# ---------------------------------------------------------------------------
# Calibration reader for ORT static quantization
# ---------------------------------------------------------------------------

class SRCalibrationReader(CalibrationDataReader):
    """Yields LR batches from an SRDataset for ORT's quantize_static.

    Pre-builds all batches at construction time so calibration is a simple
    iteration. The dataset is expected to be in ``is_train=False`` mode so
    every call yields a deterministic crop.
    """

    def __init__(
        self,
        val_set: SRDataset,
        n_samples: int,
        batch_size: int,
        input_name: str,
    ) -> None:
        self.input_name = input_name
        self._batches: list[np.ndarray] = []
        # Build batches from the first ``n_samples`` indices of the val set
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            tensors = [val_set[i][0].numpy() for i in range(start, end)]
            batch = np.stack(tensors).astype(np.float32)  # (B, 3, H, W)
            self._batches.append(batch)
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._idx >= len(self._batches):
            return None
        batch = self._batches[self._idx]
        self._idx += 1
        return {self.input_name: batch}

    def rewind(self) -> None:
        self._idx = 0


# ---------------------------------------------------------------------------
# Step 1: FP32 export from PyTorch
# ---------------------------------------------------------------------------

def export_fp32_onnx(
    pytorch_model: torch.nn.Module,
    output_path: Path,
    dummy_input: torch.Tensor,
    opset: int = 17,
    dynamic_axes: bool = True,
) -> None:
    """Direct PyTorch -> FP32 ONNX export."""
    dyn_axes = None
    if dynamic_axes:
        dyn_axes = {
            "input":  {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height_out", 3: "width_out"},
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dyn_axes,
    )
    onnx.checker.check_model(onnx.load(str(output_path)), full_check=True)


# ---------------------------------------------------------------------------
# Step 2: FP32 -> FP16 conversion
# ---------------------------------------------------------------------------

def convert_fp32_to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    """Convert an existing FP32 ONNX to FP16 using onnxconverter-common.

    Default behavior converts every eligible op; for EDSR (Conv + ReLU +
    Add + DepthToSpace) all are FP16-safe. If new ops are added that aren't
    FP16-friendly (e.g. some reduction ops can overflow), pass
    ``op_block_list`` to keep them in FP32.
    """
    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_model = onnx.load(str(fp32_path))
    fp16_model = float16.convert_float_to_float16(
        fp32_model,
        keep_io_types=False,  # input/output also become FP16 -- simpler
    )
    onnx.save(fp16_model, str(fp16_path))


# ---------------------------------------------------------------------------
# Step 3: FP32 -> INT8 static (PTQ) via ORT
# ---------------------------------------------------------------------------

def quantize_static_int8(
    fp32_path: Path,
    int8_path: Path,
    calibration_reader: SRCalibrationReader,
) -> None:
    """Run ORT's static PTQ to produce a QDQ-format INT8 ONNX.

    Settings:
      * QDQ format -- the ONNX standard quantization layout, supported by
        most edge runtimes (ORT, TensorRT, OpenVINO, vendor SDKs).
      * Symmetric INT8 for both activations and weights, per-channel weights.
        This matches our PyTorch fake-quant scheme so the accuracy comparison
        is apples-to-apples.
    """
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )


# ---------------------------------------------------------------------------
# Verification: PyTorch vs ORT numeric match
# ---------------------------------------------------------------------------

def verify_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    sample_shapes: list[tuple[int, int, int, int]],
    device: torch.device,
    atol: float,
    onnx_input_dtype: np.dtype = np.float32,
) -> dict[str, Any]:
    """Compare PyTorch and ORT outputs across multiple input shapes.

    Returns a dict that records per-shape diff stats, with an overall
    ``passed: bool`` field. We tolerate higher diff for FP16 / INT8.
    """
    providers: list[str] = []
    if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    pytorch_model.eval()
    rng = np.random.default_rng(0)
    per_shape: list[dict] = []
    all_passed = True

    for shape in sample_shapes:
        x_np = rng.random(size=shape, dtype=np.float32)
        x_pt = torch.from_numpy(x_np).to(device)

        with torch.no_grad():
            y_pt = pytorch_model(x_pt).cpu().numpy().astype(np.float32)

        x_ort = x_np.astype(onnx_input_dtype)
        y_ort_raw = sess.run(["output"], {"input": x_ort})[0]
        y_ort = y_ort_raw.astype(np.float32)

        max_abs = float(np.max(np.abs(y_pt - y_ort)))
        max_rel = float(np.max(np.abs(y_pt - y_ort) / (np.abs(y_pt) + 1e-8)))
        ok = max_abs <= atol
        all_passed = all_passed and ok
        per_shape.append({
            "shape": list(shape),
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "passed": ok,
        })

    return {
        "ort_provider": sess.get_providers()[0],
        "atol": atol,
        "passed": all_passed,
        "per_shape": per_shape,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_metadata_json(
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def write_readme(output_dir: Path, metadata: dict[str, Any]) -> None:
    artifacts = metadata["artifacts"]
    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# ONNX Export Pipeline Output\n\n")
        f.write("Single execution node of the multi-precision ONNX export pipeline. "
                "All three ONNX artifacts in this folder come from the same source "
                "checkpoint and the same calibration set; they are directly comparable.\n\n")

        f.write("## Source\n\n")
        src = metadata["source"]
        f.write(f"- **Checkpoint**: `{src['checkpoint_path']}`\n")
        f.write(f"  - mtime: {src['checkpoint_mtime']}, size: {src['checkpoint_size_mb']:.2f} MB\n")
        f.write(f"- **Model**: {src['model_arch']} -- {src['model_params']:,} params\n")
        f.write(f"- **Generated**: {metadata['datetime']}\n")
        f.write(f"- **Device used for verification**: {src['device']} ({src['device_name']})\n")
        f.write(f"- **PyTorch**: {src['torch_version']}, "
                f"**ONNX**: {src['onnx_version']}, "
                f"**ORT**: {src['ort_version']}\n\n")

        f.write("## Calibration set\n\n")
        c = metadata["calibration"]
        f.write(f"- **Source**: `{c['val_set_dir']}` ({c['val_set_size']} images, "
                f"realistic degradation, deterministic seed)\n")
        f.write(f"- **Samples used**: {c['n_samples']} LR images, "
                f"batch {c['batch_size']} -> "
                f"{(c['n_samples'] + c['batch_size'] - 1) // c['batch_size']} batches\n")
        f.write(f"- **LR patch size**: {c['patch_size']}x{c['patch_size']}\n\n")

        f.write("## Artifacts\n\n")
        f.write("| File | Size (MB) | Verified vs PyTorch |\n")
        f.write("|---|---:|:---:|\n")
        for a in artifacts:
            tag = "PASS" if a["verification"]["passed"] else "FAIL"
            f.write(f"| `{a['path_relative']}` | {a['size_mb']:.2f} | "
                    f"{tag} (atol={a['verification']['atol']:.0e}) |\n")
        f.write("\n")
        f.write("Per-shape numeric diffs are in `verification.md`. The raw shape and "
                "size info is in `metadata.json` for programmatic consumption.\n\n")

        f.write("## Quantization scheme (INT8)\n\n")
        q = metadata["int8_quantization"]
        f.write(f"- **Tool**: ONNX Runtime `quantize_static`\n")
        f.write(f"- **Format**: {q['quant_format']}\n")
        f.write(f"- **Activations**: {q['activation_type']}\n")
        f.write(f"- **Weights**: {q['weight_type']}, "
                f"per-channel = {q['per_channel']}\n")
        f.write(f"- **Calibration method**: ORT default (MinMax)\n\n")

        f.write("## Next steps\n\n")
        f.write("These three ONNX files become the input for:\n\n")
        f.write("1. `benchmark_onnx.py` (planned) -- runs each on the val set, "
                "outputs per-format PSNR + latency + memory across providers.\n")
        f.write("2. `cpp_inference/sr_cli` -- C++ deploy reference; can load any of "
                "the three.\n")
        f.write("3. Vendor toolchains (TensorRT / SNPE / NeuroPilot) -- consume "
                "`edsr_fp32.onnx` and produce backend-native engines.\n")


def write_verification_md(output_dir: Path, metadata: dict[str, Any]) -> None:
    with (output_dir / "verification.md").open("w", encoding="utf-8") as f:
        f.write("# ONNX Verification\n\n")
        f.write("Each ONNX file is compared against the PyTorch reference on the same "
                "random input across multiple shapes. Tolerance is precision-appropriate: "
                "FP32 expects near-bit-level match; FP16 and INT8 expect larger gaps.\n\n")
        for a in metadata["artifacts"]:
            f.write(f"## `{a['path_relative']}`\n\n")
            v = a["verification"]
            f.write(f"- **ORT provider**: `{v['ort_provider']}`\n")
            f.write(f"- **Tolerance (atol)**: {v['atol']:.1e}\n")
            f.write(f"- **Overall**: {'**PASS**' if v['passed'] else '**FAIL**'}\n\n")
            f.write("| Shape | max abs diff | max rel diff | Passed |\n")
            f.write("|---|---:|---:|:---:|\n")
            for r in v["per_shape"]:
                tag = "PASS" if r["passed"] else "FAIL"
                f.write(f"| {tuple(r['shape'])} | {r['max_abs_diff']:.2e} | "
                        f"{r['max_rel_diff']:.2e} | {tag} |\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-precision ONNX export pipeline")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True,
                   help="Folder for all three ONNX + metadata + README")
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--n-resblocks", type=int, default=16)
    p.add_argument("--n-feats", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=96,
                   help="LR patch edge for calibration; HR is patch * scale")
    p.add_argument("--calib-samples", type=int, default=64,
                   help="Number of LR images used for INT8 calibration")
    p.add_argument("--calib-batch", type=int, default=8)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--atol-fp32", type=float, default=1e-4)
    p.add_argument("--atol-fp16", type=float, default=5e-2)
    p.add_argument("--atol-int8", type=float, default=1e-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ONNX Export Pipeline (FP32 + FP16 + INT8)")
    print("=" * 60)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  output     : {output_dir}")
    print(f"  device     : {device}")
    print()

    # --- Build PyTorch model ---
    print("[1/4] Building PyTorch model from checkpoint ...")
    pytorch_model = EDSR(
        scale_factor=args.scale,
        n_resblocks=args.n_resblocks,
        n_feats=args.n_feats,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pytorch_model.load_state_dict(ckpt["model"])
    pytorch_model.to(device).eval()
    n_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"  {n_params:,} params loaded")

    # Verification shapes -- exercise dynamic axes
    sample_shapes: list[tuple[int, int, int, int]] = [
        (1, 3, args.patch_size, args.patch_size),
        (1, 3, 64, 64),
        (1, 3, 128, 64),
    ]

    artifacts: list[dict[str, Any]] = []

    # --- FP32 export ---
    print()
    print("[2/4] FP32 export ...")
    fp32_path = output_dir / "edsr_fp32.onnx"
    dummy = torch.randn(1, 3, args.patch_size, args.patch_size, device=device)
    export_fp32_onnx(pytorch_model, fp32_path, dummy, opset=args.opset)
    fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024)
    fp32_verify = verify_onnx(
        pytorch_model, fp32_path, sample_shapes, device, atol=args.atol_fp32,
    )
    artifacts.append({
        "path": str(fp32_path),
        "path_relative": fp32_path.name,
        "precision": "FP32",
        "size_mb": fp32_size_mb,
        "verification": fp32_verify,
    })
    print(f"  wrote {fp32_path.name} ({fp32_size_mb:.2f} MB) "
          f"-- verification: {'PASS' if fp32_verify['passed'] else 'FAIL'}")

    # --- FP16 conversion ---
    print()
    print("[3/4] FP32 -> FP16 conversion ...")
    fp16_path = output_dir / "edsr_fp16.onnx"
    convert_fp32_to_fp16(fp32_path, fp16_path)
    fp16_size_mb = fp16_path.stat().st_size / (1024 * 1024)
    fp16_verify = verify_onnx(
        pytorch_model, fp16_path, sample_shapes, device,
        atol=args.atol_fp16,
        onnx_input_dtype=np.float16,
    )
    artifacts.append({
        "path": str(fp16_path),
        "path_relative": fp16_path.name,
        "precision": "FP16",
        "size_mb": fp16_size_mb,
        "verification": fp16_verify,
    })
    print(f"  wrote {fp16_path.name} ({fp16_size_mb:.2f} MB) "
          f"-- verification: {'PASS' if fp16_verify['passed'] else 'FAIL'}")

    # --- INT8 static quantization ---
    print()
    print("[4/4] FP32 -> INT8 static (PTQ) ...")
    int8_path = output_dir / "edsr_int8_static.onnx"

    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation="realistic",
        is_train=False,
    )
    print(f"  building calibration reader from val set ({len(val_set)} images, "
          f"using first {args.calib_samples}) ...")
    calib_reader = SRCalibrationReader(
        val_set,
        n_samples=args.calib_samples,
        batch_size=args.calib_batch,
        input_name="input",
    )
    print(f"  running ORT quantize_static (this may take a moment) ...")
    quantize_static_int8(fp32_path, int8_path, calib_reader)
    int8_size_mb = int8_path.stat().st_size / (1024 * 1024)
    int8_verify = verify_onnx(
        pytorch_model, int8_path, sample_shapes, device,
        atol=args.atol_int8,
    )
    artifacts.append({
        "path": str(int8_path),
        "path_relative": int8_path.name,
        "precision": "INT8 (static PTQ)",
        "size_mb": int8_size_mb,
        "verification": int8_verify,
    })
    print(f"  wrote {int8_path.name} ({int8_size_mb:.2f} MB) "
          f"-- verification: {'PASS' if int8_verify['passed'] else 'FAIL'}")

    # --- Metadata + README ---
    print()
    print("Writing metadata and README ...")

    ckpt_path = Path(args.checkpoint)
    metadata: dict[str, Any] = {
        "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_mtime": datetime.datetime.fromtimestamp(
                ckpt_path.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "checkpoint_size_mb": ckpt_path.stat().st_size / (1024 * 1024),
            "model_arch": (
                f"EDSR(scale_factor={args.scale}, "
                f"n_resblocks={args.n_resblocks}, n_feats={args.n_feats})"
            ),
            "model_params": n_params,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda"
                else (platform.processor() or "CPU")
            ),
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "ort_version": ort.__version__,
        },
        "calibration": {
            "val_set_dir": str(Path(args.data_root) / args.val_dir),
            "val_set_size": len(val_set),
            "n_samples": args.calib_samples,
            "batch_size": args.calib_batch,
            "patch_size": args.patch_size,
        },
        "int8_quantization": {
            "tool": "onnxruntime.quantization.quantize_static",
            "quant_format": "QDQ",
            "activation_type": "QInt8 (symmetric per-tensor)",
            "weight_type": "QInt8 (symmetric per-channel)",
            "per_channel": True,
            "reduce_range": False,
        },
        "verification_shapes": [list(s) for s in sample_shapes],
        "artifacts": artifacts,
    }

    write_metadata_json(output_dir, metadata)
    write_verification_md(output_dir, metadata)
    write_readme(output_dir, metadata)

    print()
    print(f"  metadata.json   -> {output_dir / 'metadata.json'}")
    print(f"  verification.md -> {output_dir / 'verification.md'}")
    print(f"  README.md       -> {output_dir / 'README.md'}")

    # --- Final summary ---
    print()
    print("=" * 60)
    print("Pipeline complete. Artifacts in:", output_dir)
    print("=" * 60)
    for a in artifacts:
        tag = "PASS" if a["verification"]["passed"] else "FAIL"
        print(f"  {a['path_relative']:30s}  {a['size_mb']:>6.2f} MB  [{tag}]")


if __name__ == "__main__":
    main()
