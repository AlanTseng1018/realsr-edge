"""ONNX deployment benchmark.

Loads one or more ``.onnx`` files (e.g. the FP32/FP16/INT8 trio produced by
``export_pipeline.py``) and measures, for each (ONNX, ORT execution provider)
combination:

* **PSNR on the val set** -- the deploy-side accuracy ground truth.
* **Forward latency** at a fixed input shape, mean +/- std with proper
  warmup and per-iter ``cuda.synchronize`` (when CUDA is in use).
* **File size** -- model footprint on disk.

A single run of this script writes a folder containing a markdown shootout
report, a CSV with the same data, and a JSON metadata block. One folder =
one benchmark execution node.

Run example::

    python -m src.deployment.benchmark_onnx --onnx-dir results/onnx_exports/edsr_200ep --output-dir results/onnx_benchmark/edsr_200ep --providers cuda cpu --bench-shape 1x3x96x96
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


def _setup_dll_paths() -> None:
    """Make TensorRT libs discoverable on Windows.

    ORT loads ``onnxruntime_providers_tensorrt.dll`` lazily at session
    creation. That DLL depends on ``nvinfer_*.dll`` which ships with the
    ``tensorrt-cu12`` pip package but lives in the package directory, NOT
    on the system PATH. Without an explicit ``add_dll_directory`` call,
    Windows can't find it and TRT EP silently falls back to CUDA EP.
    """
    try:
        import tensorrt_libs
        os.add_dll_directory(os.path.dirname(tensorrt_libs.__file__))
    except ImportError:
        pass  # tensorrt-cu12 not installed; TRT EP unavailable, that's OK


_setup_dll_paths()


import numpy as np
import onnxruntime as ort
import torch
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

PROVIDER_CHOICES = {
    "cuda":     ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "cpu":      ["CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}


def make_session(
    onnx_path: Path,
    provider_name: str,
    trt_cache_dir: Path | None = None,
    bench_shape: tuple[int, int, int, int] | None = None,
    max_batch: int = 1,
    input_name: str = "input",
) -> ort.InferenceSession | None:
    """Build an ORT session, falling back gracefully if the provider isn't available.

    For ``provider_name == "tensorrt"`` we configure the EP with:
      * ``trt_engine_cache_enable`` + ``trt_engine_cache_path`` so the
        compiled engine is reused on subsequent runs (first build takes
        1-3 minutes on this model; cached run is instant).
      * ``trt_profile_*_shapes`` locked to ``bench_shape`` so TensorRT
        can specialize for that shape. Without these, dynamic-axes ONNX
        either fails to build or picks an arbitrary default.

    INT8 ONNX (QDQ format) is automatically run in INT8 by TensorRT --
    it honors the QDQ scales embedded in the graph.
    """
    available = set(ort.get_available_providers())
    chain = PROVIDER_CHOICES.get(provider_name)
    if chain is None:
        raise ValueError(f"Unknown provider {provider_name!r}")

    chosen: list = []
    for p in chain:
        if p not in available:
            continue
        if p == "TensorrtExecutionProvider" and trt_cache_dir is not None:
            opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_cache_dir),
            }
            if bench_shape is not None:
                # TRT needs an explicit shape range. We use the bench_shape for
                # min/opt (latency benchmark uses this exact shape, hot path),
                # and bump max to ``max_batch`` so PSNR eval batches still fit.
                # Without this, a session built for batch=1 fails when val
                # eval feeds batch=8.
                _, c, h, w = bench_shape
                bench_str = f"{input_name}:{'x'.join(str(d) for d in bench_shape)}"
                max_str = f"{input_name}:{max_batch}x{c}x{h}x{w}"
                opts["trt_profile_min_shapes"] = bench_str
                opts["trt_profile_opt_shapes"] = bench_str
                opts["trt_profile_max_shapes"] = max_str
            chosen.append((p, opts))
        else:
            chosen.append(p)

    if not chosen:
        return None
    try:
        return ort.InferenceSession(str(onnx_path), providers=chosen)
    except Exception:
        return None


def session_input_dtype(session: ort.InferenceSession) -> np.dtype:
    """Match numpy dtype to ORT input type (FP32 ONNX: float32; FP16 ONNX: float16)."""
    type_str = session.get_inputs()[0].type
    if "float16" in type_str:
        return np.float16
    if "double" in type_str:
        return np.float64
    return np.float32


# ---------------------------------------------------------------------------
# Eval / latency
# ---------------------------------------------------------------------------

def evaluate_metrics_onnx(
    session: ort.InferenceSession,
    val_loader: DataLoader,
) -> tuple[float, float]:
    """Mean per-image (PSNR dB, SSIM) over the val loader via the ORT session.

    Both metrics are computed on the same tensor pair so the deploy-side
    numbers stay apples-to-apples with the train / fake-quant pipelines.
    """
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    in_dtype = session_input_dtype(session)

    psnr_sum, ssim_sum, count = 0.0, 0.0, 0
    for lr, hr in val_loader:
        lr_np = lr.numpy().astype(in_dtype)
        sr_np = session.run([out_name], {in_name: lr_np})[0]
        sr = torch.from_numpy(sr_np.astype(np.float32)).clamp(0.0, 1.0)
        # hr is FP32 [0, 1]
        mse = ((sr - hr) ** 2).mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
        psnr_sum += psnr.sum().item()

        sr_arr = sr.numpy()
        hr_arr = hr.numpy()
        for i in range(sr_arr.shape[0]):
            # Cast to Python float so the running sum stays JSON-serialisable
            # (skimage SSIM returns numpy scalar types).
            ssim_sum += float(ssim_fn(
                hr_arr[i].transpose(1, 2, 0),
                sr_arr[i].transpose(1, 2, 0),
                data_range=1.0,
                channel_axis=2,
            ))

        count += psnr.numel()
    n = max(count, 1)
    return psnr_sum / n, ssim_sum / n


def benchmark_latency_onnx(
    session: ort.InferenceSession,
    bench_shape: tuple[int, int, int, int],
    n_warmup: int = 10,
    n_iter: int = 50,
    is_cuda: bool = False,
) -> tuple[float, float]:
    """Forward latency in ms (mean, std) at ``bench_shape``."""
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    in_dtype = session_input_dtype(session)
    sample = np.random.rand(*bench_shape).astype(in_dtype)

    for _ in range(n_warmup):
        session.run([out_name], {in_name: sample})
    if is_cuda:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        session.run([out_name], {in_name: sample})
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_benchmark_md(
    path: Path,
    rows: list[dict],
    metadata: dict[str, Any],
) -> None:
    """Write the human-readable benchmark report.

    Adds derived columns (drop vs FP32, speedup vs FP32 same-provider) for
    deploy-decision readability.
    """
    # Pick FP32 baseline rows (one per provider) for derived comparisons
    fp32_psnr = next(
        (r["psnr_db"] for r in rows
         if r["precision"] == "FP32" and r["psnr_db"] is not None),
        None,
    )
    fp32_ssim = next(
        (r["ssim"] for r in rows
         if r["precision"] == "FP32" and r.get("ssim") is not None),
        None,
    )
    fp32_lat_per_provider: dict[str, float] = {
        r["provider"]: r["latency_ms_mean"]
        for r in rows
        if r["precision"] == "FP32"
        and r["latency_ms_mean"] is not None
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# ONNX Deployment Benchmark\n\n")
        f.write("Single execution node of the ONNX runtime benchmark. Each row "
                "is one (ONNX file, ORT execution provider) pair, evaluated on "
                "the same val set with the same input shape for latency.\n\n")

        # What was tested
        f.write("## What was tested\n\n")
        f.write(f"- **Generated**: {metadata['datetime']}\n")
        f.write(f"- **ONNX folder**: `{metadata['onnx_dir']}`\n")
        f.write(f"- **Validation set**: `{metadata['val_set_dir']}` "
                f"({metadata['val_set_size']} images, realistic degradation, "
                f"LR patch {metadata['val_lr_patch']}x{metadata['val_lr_patch']})\n")
        f.write(f"- **Latency input shape**: `{metadata['bench_shape']}` "
                f"({metadata['n_warmup']} warmup + {metadata['n_iter']} timed iters)\n")
        f.write(f"- **Providers tested**: "
                f"{', '.join('`' + p + '`' for p in metadata['providers'])}\n")
        f.write(f"- **Hardware**: {metadata['device_name']}\n")
        f.write(f"- **ORT version**: {metadata['ort_version']}\n\n")

        # Main shootout table
        f.write("## Shootout table\n\n")
        f.write("| ONNX | Provider | PSNR (dB) | PSNR drop | SSIM | SSIM drop | "
                "Latency (ms) | Speedup vs FP32 same-provider | Size (MB) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            psnr = (
                f"{r['psnr_db']:.3f}"
                if r["psnr_db"] is not None else "n/a"
            )
            if r["psnr_db"] is None or fp32_psnr is None:
                drop = "n/a"
            else:
                drop = f"{fp32_psnr - r['psnr_db']:+.3f}"

            ssim_val = (
                f"{r['ssim']:.4f}"
                if r.get("ssim") is not None else "n/a"
            )
            if r.get("ssim") is None or fp32_ssim is None:
                ssim_drop = "n/a"
            else:
                ssim_drop = f"{fp32_ssim - r['ssim']:+.4f}"

            if r["latency_ms_mean"] is not None:
                lat = f"{r['latency_ms_mean']:.2f} +/- {r['latency_ms_std']:.2f}"
            else:
                lat = r.get("error", "n/a")

            base = fp32_lat_per_provider.get(r["provider"])
            if base is None or r["latency_ms_mean"] is None:
                speedup = "n/a"
            else:
                ratio = base / r["latency_ms_mean"]
                if ratio >= 1.0:
                    speedup = f"{ratio:.2f}x faster"
                else:
                    speedup = f"{1.0 / ratio:.2f}x slower"

            size_mb = f"{r['size_mb']:.2f}" if r["size_mb"] is not None else "n/a"

            f.write(f"| `{r['onnx']}` | `{r['provider']}` | "
                    f"{psnr} | {drop} | {ssim_val} | {ssim_drop} | "
                    f"{lat} | {speedup} | {size_mb} |\n")
        f.write("\n")

        # How to read
        f.write("## How to read\n\n")
        f.write("- **PSNR** is the deploy-side accuracy: ONNX session output "
                "evaluated on the val set against HR ground truth. "
                "**Provider-invariant within rounding** -- if it differs much "
                "between CUDA and CPU for the same ONNX, that's a debug "
                "signal.\n")
        f.write("- **SSIM** is the perceptual cross-check on the same tensor "
                "pair. PSNR can drop 0.2 dB without visible artefacts; "
                "SSIM going down by ~0.001 is usually safe, ~0.005+ is the "
                "point at which side-by-side viewing starts to reveal the "
                "loss. Disagreement between PSNR and SSIM rankings is a "
                "signal that the precision is hurting structure (edges) more "
                "than bulk fidelity, or vice versa.\n")
        f.write("- **Drop vs FP32** uses the FP32 PSNR / SSIM as baseline. "
                "Should match (within ~0.1 dB / ~0.001 SSIM) the fake-quant "
                "prediction in "
                "`results/quantization/200ep_with_report/report.md`.\n")
        f.write("- **Latency** is forward-pass only. **Provider-specific**: "
                "INT8 ONNX often runs slower on CUDA EP than FP32 due to "
                "QDQ insertion + memcpy nodes; on CPU EP, INT8 typically "
                "wins because of VNNI / native INT8 instructions. "
                "**TensorRT EP is the right path for true GPU INT8 deploy** "
                "(not benchmarked here).\n")
        f.write("- **Speedup** is per provider: it answers \"if I'm "
                "deploying on this hardware, what does each precision give "
                "me?\" -- not \"is X precision globally fastest\".\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_bench_shape(s: str) -> tuple[int, int, int, int]:
    """Parse a NxCxHxW string like '1x3x96x96' into a tuple of ints."""
    parts = s.lower().split("x")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--bench-shape must be NxCxHxW, got {s!r}"
        )
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark a folder of ONNX files across providers."
    )
    p.add_argument("--onnx-dir", type=str, required=True,
                   help="Folder containing .onnx files (e.g. export_pipeline output).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Folder for benchmark.md / benchmark.csv / metadata.json")
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--providers", type=str, nargs="+",
                   choices=list(PROVIDER_CHOICES.keys()),
                   default=["cuda", "cpu"])
    p.add_argument("--bench-shape", type=parse_bench_shape, default=(1, 3, 96, 96),
                   help="Input shape for latency benchmark, NxCxHxW (default 1x3x96x96)")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-iter", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"No .onnx files found in {onnx_dir}")

    # Map filename -> precision label (best-effort)
    def precision_of(name: str) -> str:
        n = name.lower()
        if "fp16" in n:
            return "FP16"
        if "int8" in n:
            return "INT8"
        return "FP32"

    print("=" * 60)
    print("ONNX Deployment Benchmark")
    print("=" * 60)
    print(f"  onnx-dir   : {onnx_dir}")
    print(f"  output     : {output_dir}")
    print(f"  providers  : {args.providers}")
    print(f"  bench shape: {args.bench_shape}")
    print(f"  ORT        : {ort.__version__}")
    print(f"  ORT EPs    : {ort.get_available_providers()}")
    print()

    # Build val loader once
    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation="realistic",
        is_train=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    print(f"  val set: {len(val_set)} images")
    print(f"  found {len(onnx_files)} ONNX file(s):")
    for p in onnx_files:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"    - {p.name} ({size_mb:.2f} MB, {precision_of(p.name)})")
    print()

    # TensorRT engine cache lives next to the report so it's "owned" by this
    # execution node. First run builds, subsequent runs reuse instantly.
    trt_cache_dir = output_dir / "trt_engine_cache"
    if "tensorrt" in args.providers:
        trt_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  TRT engine cache -> {trt_cache_dir}")
        print(f"  (first run builds engines; expect 1-3 min per ONNX. "
              f"cached runs are near-instant.)")
    print()

    rows: list[dict[str, Any]] = []
    for onnx_path in onnx_files:
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        for provider in args.providers:
            print(f"[{onnx_path.name} @ {provider}] ", end="", flush=True)
            sess_t0 = time.perf_counter()
            session = make_session(
                onnx_path, provider,
                trt_cache_dir=trt_cache_dir if provider == "tensorrt" else None,
                bench_shape=args.bench_shape,
                max_batch=args.batch_size,
            )
            sess_build_ms = (time.perf_counter() - sess_t0) * 1000.0

            row: dict[str, Any] = {
                "onnx": onnx_path.name,
                "precision": precision_of(onnx_path.name),
                "provider": provider,
                "size_mb": size_mb,
                "psnr_db": None,
                "ssim": None,
                "latency_ms_mean": None,
                "latency_ms_std": None,
                "session_build_ms": sess_build_ms,
                "active_provider": None,
                "error": None,
            }
            if session is None:
                row["error"] = "provider unavailable"
                rows.append(row)
                print("provider unavailable")
                continue

            row["active_provider"] = session.get_providers()[0]
            if provider == "tensorrt":
                # On TRT EP, the active EP is "TensorrtExecutionProvider" if
                # the engine actually built. Print build time prominently --
                # this is the cost users care about.
                print(f"(session build: {sess_build_ms:.0f} ms) ", end="", flush=True)

            try:
                psnr, ssim = evaluate_metrics_onnx(session, val_loader)
                row["psnr_db"] = psnr
                row["ssim"] = ssim
            except Exception as e:
                row["error"] = f"metrics failed: {e!s}"
                rows.append(row)
                print(f"metrics FAIL ({e!s})")
                continue

            is_cuda = "CUDA" in row["active_provider"] or "Tensorrt" in row["active_provider"]
            try:
                lat_mean, lat_std = benchmark_latency_onnx(
                    session, args.bench_shape,
                    n_warmup=args.n_warmup, n_iter=args.n_iter,
                    is_cuda=is_cuda,
                )
                row["latency_ms_mean"] = lat_mean
                row["latency_ms_std"] = lat_std
            except Exception as e:
                row["error"] = f"latency failed: {e!s}"
                rows.append(row)
                print(f"PSNR={psnr:.3f}, SSIM={ssim:.4f}, latency FAIL ({e!s})")
                continue

            rows.append(row)
            print(f"PSNR={psnr:.3f} dB | SSIM={ssim:.4f} | "
                  f"latency={lat_mean:.2f} +/- {lat_std:.2f} ms "
                  f"(active_ep={row['active_provider']})")

    # Write outputs
    print()
    print("Writing outputs ...")

    write_csv(output_dir / "benchmark.csv", rows)

    # Metadata
    metadata: dict[str, Any] = {
        "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "onnx_dir": str(onnx_dir),
        "val_set_dir": str(Path(args.data_root) / args.val_dir),
        "val_set_size": len(val_set),
        "val_lr_patch": args.patch_size,
        "bench_shape": list(args.bench_shape),
        "n_warmup": args.n_warmup,
        "n_iter": args.n_iter,
        "providers": args.providers,
        "ort_version": ort.__version__,
        "ort_available_providers": ort.get_available_providers(),
        "device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else (platform.processor() or "CPU")
        ),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({**metadata, "rows": rows}, f, indent=2, default=str)

    write_benchmark_md(output_dir / "benchmark.md", rows, metadata)

    print(f"  benchmark.md   -> {output_dir / 'benchmark.md'}")
    print(f"  benchmark.csv  -> {output_dir / 'benchmark.csv'}")
    print(f"  metadata.json  -> {output_dir / 'metadata.json'}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
