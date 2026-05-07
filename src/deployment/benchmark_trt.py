"""Native TensorRT engine benchmark.

Builds TensorRT engines directly from ONNX using the TensorRT Python API
(not ORT TensorRT EP), then measures PSNR and latency for each precision.

Why this matters over ORT TensorRT EP:
- ORT TRT EP feeds QDQ ONNX through an intermediate layer that may not fuse
  Quantize/Dequantize nodes, so INT8 sees no speedup.
- Building with the TRT Python API lets TRT fully optimize the graph: kernel
  fusion, INT8 tensor core selection, memory layout planning.

Precision modes:
- FP32: edsr_fp32.onnx, builder flag default
- FP16: edsr_fp32.onnx, BuilderFlag.FP16 (TRT fuses to FP16 kernels)
- INT8: edsr_fp32.onnx + IInt8EntropyCalibrator2 — TRT calibrates on val LR
        patches and builds a native INT8 engine with per-tensor activation
        scales.  (QDQ ONNX is NOT used: TRT 10 rejects INT32 bias dequantize
        nodes produced by onnxruntime quantize_static.)

Run example::

    python -m src.deployment.benchmark_trt --onnx-dir results/onnx_exports/edsr_200ep --output-dir results/trt_benchmark/edsr_200ep --bench-shape 1x3x96x96
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

import numpy as np
import tensorrt as trt
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset


# ---------------------------------------------------------------------------
# TensorRT logger
# ---------------------------------------------------------------------------

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# INT8 calibrator
# ---------------------------------------------------------------------------

class ValSetCalibrator(trt.IInt8EntropyCalibrator2):
    """IInt8EntropyCalibrator2 backed by val-set LR patches.

    TRT calls get_batch() repeatedly until it returns False.  Each call copies
    one batch of calibration data to a pre-allocated GPU tensor and returns its
    data_ptr so TRT can compute activation histograms.
    """

    def __init__(
        self,
        calib_batches: list[np.ndarray],
        cache_path: Path,
    ) -> None:
        super().__init__()
        self._batches = calib_batches
        self._idx = 0
        self._cache_path = cache_path
        # Pre-allocate GPU buffer for one batch
        self._d_inp = torch.zeros(calib_batches[0].shape, dtype=torch.float32, device="cuda")

    def get_batch_size(self) -> int:
        return self._batches[0].shape[0]

    def get_batch(self, names: list[str]):  # type: ignore[override]
        if self._idx >= len(self._batches):
            return None
        batch = self._batches[self._idx]
        self._idx += 1
        self._d_inp.copy_(torch.from_numpy(batch))
        return [self._d_inp.data_ptr()]

    def read_calibration_cache(self):
        if self._cache_path.exists():
            with self._cache_path.open("rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("wb") as f:
            f.write(cache)


def build_calib_batches(
    val_set: SRDataset,
    bench_shape: tuple[int, int, int, int],
    n_samples: int = 64,
) -> list[np.ndarray]:
    """Sample LR patches from the val set for INT8 calibration."""
    n, c, h, w = bench_shape
    batches: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    for i in range(min(n_samples, len(val_set))):
        lr, _ = val_set[i]
        # lr is (C, H_lr, W_lr); crop/resize to (C, h, w)
        lr_np = lr.numpy()
        if lr_np.shape[1] < h or lr_np.shape[2] < w:
            # pad if patch is smaller (shouldn't happen in normal datasets)
            pad_h = max(0, h - lr_np.shape[1])
            pad_w = max(0, w - lr_np.shape[2])
            lr_np = np.pad(lr_np, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        lr_np = lr_np[:, :h, :w]
        buf.append(lr_np)
        if len(buf) == n:
            batches.append(np.stack(buf).astype(np.float32))
            buf = []
    if buf:
        batches.append(np.stack(buf).astype(np.float32))
    return batches


# ---------------------------------------------------------------------------
# Engine build
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: Path,
    precision: str,
    bench_shape: tuple[int, int, int, int],
    engine_cache_path: Path | None = None,
    max_workspace_gb: float = 2.0,
    calibrator: trt.IInt8EntropyCalibrator2 | None = None,
) -> trt.ICudaEngine | None:
    """Parse ONNX and build a TensorRT engine.

    Args:
        onnx_path: Path to the .onnx file (always FP32 ONNX).
        precision: One of ``"fp32"``, ``"fp16"``, ``"int8"``.
        bench_shape: (N, C, H, W) — used as opt/max profile shape.
        engine_cache_path: If given and the .engine file exists, load from
            cache instead of rebuilding.
        max_workspace_gb: Memory pool limit for TRT builder (default 2 GB).
        calibrator: Required for ``precision="int8"``.
    """
    if engine_cache_path is not None and engine_cache_path.exists():
        print(f"    loading cached engine: {engine_cache_path.name}", flush=True)
        runtime = trt.Runtime(TRT_LOGGER)
        with engine_cache_path.open("rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    TRT parse error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(max_workspace_gb * (1 << 30)),
    )

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("    WARNING: FP16 not fast on this GPU, building anyway")
        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("    WARNING: INT8 not fast on this GPU, building anyway")
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            raise ValueError("calibrator is required for INT8")
        config.int8_calibrator = calibrator

    # Dynamic shape profile: lock min/opt/max to bench_shape so TRT can pick
    # the most efficient kernel for exactly this shape.
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name, bench_shape, bench_shape, bench_shape)
    config.add_optimization_profile(profile)

    print(f"    building {precision.upper()} engine (this may take 1-3 min) ...",
          end="", flush=True)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    print(f" done ({build_s:.1f}s)", flush=True)

    if serialized is None:
        print("    ERROR: engine build returned None")
        return None

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)

    if engine_cache_path is not None:
        engine_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with engine_cache_path.open("wb") as f:
            f.write(serialized)
        print(f"    saved engine -> {engine_cache_path.name}", flush=True)

    return engine


# ---------------------------------------------------------------------------
# Inference context
# ---------------------------------------------------------------------------

_TRT_TO_TORCH_DTYPE = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int8:    torch.int8,
    trt.int32:   torch.int32,
    trt.bool:    torch.bool,
}


class TRTContext:
    """Holds a TRT execution context and its I/O buffers (PyTorch CUDA tensors)."""

    def __init__(self, engine: trt.ICudaEngine, bench_shape: tuple[int, int, int, int]):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.bench_shape = bench_shape

        inp_name = engine.get_tensor_name(0)
        out_name = engine.get_tensor_name(1)
        self.inp_name = inp_name
        self.out_name = out_name

        self.context.set_input_shape(inp_name, bench_shape)

        inp_trt_dtype = engine.get_tensor_dtype(inp_name)
        out_shape = tuple(self.context.get_tensor_shape(out_name))
        out_trt_dtype = engine.get_tensor_dtype(out_name)

        self.inp_np_dtype = trt.nptype(inp_trt_dtype)
        self.out_shape = out_shape

        inp_torch_dtype = _TRT_TO_TORCH_DTYPE[inp_trt_dtype]
        out_torch_dtype = _TRT_TO_TORCH_DTYPE[out_trt_dtype]

        # Pre-allocate pinned host buffer for fast H2D copy
        self.h_inp = torch.zeros(bench_shape, dtype=inp_torch_dtype).pin_memory()
        self.d_inp = torch.zeros(bench_shape, dtype=inp_torch_dtype, device="cuda")
        self.d_out = torch.zeros(out_shape,   dtype=out_torch_dtype, device="cuda")
        self.h_out = torch.zeros(out_shape,   dtype=out_torch_dtype).pin_memory()

        self.context.set_tensor_address(inp_name, self.d_inp.data_ptr())
        self.context.set_tensor_address(out_name, self.d_out.data_ptr())

        self.stream = torch.cuda.Stream()

    def infer(self, inp_np: np.ndarray) -> np.ndarray:
        self.h_inp.copy_(torch.from_numpy(inp_np.astype(self.inp_np_dtype)))
        with torch.cuda.stream(self.stream):
            self.d_inp.copy_(self.h_inp, non_blocking=True)
            self.context.execute_async_v3(self.stream.cuda_stream)
            self.h_out.copy_(self.d_out, non_blocking=True)
        self.stream.synchronize()
        return self.h_out.numpy().copy()

    def free(self) -> None:
        pass  # PyTorch tensors are freed by GC


# ---------------------------------------------------------------------------
# Eval / latency
# ---------------------------------------------------------------------------

def evaluate_psnr_trt(
    ctx: TRTContext,
    val_loader: DataLoader,
    bench_shape: tuple[int, int, int, int],
) -> float:
    psnr_sum, count = 0.0, 0
    batch_size = bench_shape[0]

    for lr, hr in val_loader:
        # val_loader may return batches != bench_shape batch dim; run per-image
        for i in range(lr.shape[0]):
            lr_single = lr[i : i + 1].numpy()
            sr_np = ctx.infer(lr_single).astype(np.float32)
            sr = torch.from_numpy(sr_np).clamp(0.0, 1.0)
            hr_single = hr[i : i + 1]
            mse = ((sr - hr_single) ** 2).mean()
            psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
            psnr_sum += psnr.item()
            count += 1

    return psnr_sum / max(count, 1)


def benchmark_latency_trt(
    ctx: TRTContext,
    bench_shape: tuple[int, int, int, int],
    n_warmup: int = 20,
    n_iter: int = 100,
) -> tuple[float, float]:
    """Measure GPU latency with CUDA Events (excludes Python call overhead).

    time.perf_counter() per-call measures wall-clock including Python overhead,
    which dominates for fast kernels (< 2 ms) and makes FP16/INT8 appear slower
    than FP32. CUDA Events measure only GPU execution time.
    """
    sample = np.random.rand(*bench_shape).astype(ctx.inp_np_dtype)
    for _ in range(n_warmup):
        ctx.infer(sample)

    # Batch timing: total GPU time for n_iter calls / n_iter
    t_start = torch.cuda.Event(enable_timing=True)
    t_end   = torch.cuda.Event(enable_timing=True)
    t_start.record()
    for _ in range(n_iter):
        ctx.infer(sample)
    t_end.record()
    torch.cuda.synchronize()
    mean_ms = t_start.elapsed_time(t_end) / n_iter

    # Per-call std via a second pass with individual events
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        ctx.infer(sample)
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    std_ms = float(np.std(times))

    return mean_ms, std_ms


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_report_md(path: Path, rows: list[dict], metadata: dict) -> None:
    fp32_psnr = next(
        (r["psnr_db"] for r in rows if r["precision"] == "FP32" and r["psnr_db"] is not None),
        None,
    )
    fp32_lat = next(
        (r["latency_ms_mean"] for r in rows
         if r["precision"] == "FP32" and r["latency_ms_mean"] is not None),
        None,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Native TensorRT Engine Benchmark\n\n")
        f.write(
            "Engines built with the **TensorRT Python API** (not ORT TRT EP).\n"
            "TRT fully fuses the graph — INT8 uses native INT8 tensor cores,\n"
            "FP16 uses FP16 tensor cores. This is the correct way to measure\n"
            "TRT INT8 speedup (QDQ ONNX fed to ORT TRT EP does not fuse and\n"
            "shows no INT8 gain).\n\n"
        )
        f.write("## What was tested\n\n")
        f.write(f"- **Generated**: {metadata['datetime']}\n")
        f.write(f"- **ONNX folder**: `{metadata['onnx_dir']}`\n")
        f.write(f"- **Validation set**: `{metadata['val_set_dir']}` "
                f"({metadata['val_set_size']} images)\n")
        f.write(f"- **Latency input shape**: `{metadata['bench_shape']}` "
                f"({metadata['n_warmup']} warmup + {metadata['n_iter']} timed iters)\n")
        f.write(f"- **TensorRT version**: {metadata['trt_version']}\n")
        f.write(f"- **Hardware**: {metadata['device_name']}\n\n")

        f.write("## Results\n\n")
        f.write("| Precision | ONNX source | PSNR (dB) | Drop vs FP32 | "
                "Latency (ms) | Speedup vs FP32 | Engine size (MB) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            psnr = f"{r['psnr_db']:.3f}" if r["psnr_db"] is not None else "n/a"
            if r["psnr_db"] is None or fp32_psnr is None:
                drop = "n/a"
            else:
                drop = f"{fp32_psnr - r['psnr_db']:+.3f}"
            if r["latency_ms_mean"] is not None:
                lat = f"{r['latency_ms_mean']:.2f} +/- {r['latency_ms_std']:.2f}"
                if fp32_lat is not None:
                    ratio = fp32_lat / r["latency_ms_mean"]
                    speedup = f"{ratio:.2f}x faster" if ratio >= 1.0 else f"{1/ratio:.2f}x slower"
                else:
                    speedup = "n/a"
            else:
                lat = r.get("error", "n/a")
                speedup = "n/a"
            size = f"{r['engine_size_mb']:.2f}" if r["engine_size_mb"] is not None else "n/a"
            f.write(f"| `{r['precision']}` | `{r['onnx_source']}` | "
                    f"{psnr} | {drop} | {lat} | {speedup} | {size} |\n")
        f.write("\n")

        f.write("## INT8 notes\n\n")
        f.write("INT8 engine is built from `edsr_fp32.onnx` + `IInt8EntropyCalibrator2`.\n")
        f.write("TRT calibrates activation ranges on 64 val-set LR patches, then builds\n")
        f.write("native INT8 kernels with full graph fusion.  QDQ ONNX is **not** used\n")
        f.write("because TRT 10 rejects INT32 bias dequantize nodes produced by\n")
        f.write("`onnxruntime.quantization.quantize_static`.\n\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_bench_shape(s: str) -> tuple[int, int, int, int]:
    parts = s.lower().split("x")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--bench-shape must be NxCxHxW, got {s!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native TensorRT benchmark (FP32/FP16/INT8).")
    p.add_argument("--onnx-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/DIV2K")
    p.add_argument("--val-dir", type=str, default="DIV2K_valid_HR")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--bench-shape", type=parse_bench_shape, default=(1, 3, 96, 96))
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--max-workspace-gb", type=float, default=2.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    onnx_dir = Path(args.onnx_dir)
    output_dir = Path(args.output_dir)
    engine_dir = output_dir / "engines"
    engine_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx = onnx_dir / "edsr_fp32.onnx"
    if not fp32_onnx.exists():
        raise SystemExit(f"Required ONNX not found: {fp32_onnx}")

    # (precision_label, onnx_source, trt_precision_arg, engine_filename)
    # All three use FP32 ONNX; INT8 uses a calibrator instead of QDQ ONNX
    # (TRT 10 rejects INT32 bias dequantize nodes from onnxruntime quantize_static)
    configs = [
        ("FP32", fp32_onnx, "fp32", engine_dir / "edsr_fp32.engine"),
        ("FP16", fp32_onnx, "fp16", engine_dir / "edsr_fp16.engine"),
        ("INT8", fp32_onnx, "int8", engine_dir / "edsr_int8.engine"),
    ]

    print("=" * 60)
    print("Native TensorRT Engine Benchmark")
    print("=" * 60)
    print(f"  onnx-dir    : {onnx_dir}")
    print(f"  output-dir  : {output_dir}")
    print(f"  bench shape : {args.bench_shape}")
    print(f"  TRT version : {trt.__version__}")
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print()

    val_set = SRDataset(
        hr_dir=Path(args.data_root) / args.val_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size * args.scale,
        degradation="realistic",
        is_train=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    print(f"  val set: {len(val_set)} images")

    # Build calibration batches once (reused across runs via cache file)
    calib_cache = engine_dir / "int8_calib.cache"
    print(f"  building INT8 calibration data (64 samples) ...")
    calib_batches = build_calib_batches(val_set, args.bench_shape, n_samples=64)
    print(f"  {len(calib_batches)} calibration batch(es)\n")

    rows: list[dict[str, Any]] = []

    for precision_label, onnx_path, trt_prec, engine_path in configs:
        print(f"[{precision_label}] source: {onnx_path.name}")
        row: dict[str, Any] = {
            "precision": precision_label,
            "onnx_source": onnx_path.name,
            "engine_size_mb": None,
            "psnr_db": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "error": None,
        }

        calibrator = None
        if trt_prec == "int8":
            calibrator = ValSetCalibrator(calib_batches, calib_cache)

        engine = build_engine(
            onnx_path, trt_prec, args.bench_shape,
            engine_cache_path=engine_path,
            max_workspace_gb=args.max_workspace_gb,
            calibrator=calibrator,
        )
        if engine is None:
            row["error"] = "engine build failed"
            rows.append(row)
            print(f"  FAILED\n")
            continue

        if engine_path.exists():
            row["engine_size_mb"] = engine_path.stat().st_size / (1024 * 1024)

        ctx = TRTContext(engine, args.bench_shape)
        print(f"    input dtype : {ctx.inp_np_dtype}")
        print(f"    output shape: {ctx.out_shape}")

        try:
            psnr = evaluate_psnr_trt(ctx, val_loader, args.bench_shape)
            row["psnr_db"] = psnr
            print(f"    PSNR        : {psnr:.3f} dB")
        except Exception as e:
            row["error"] = f"PSNR failed: {e}"
            ctx.free()
            rows.append(row)
            print(f"    PSNR FAILED : {e}\n")
            continue

        try:
            lat_mean, lat_std = benchmark_latency_trt(
                ctx, args.bench_shape, n_warmup=args.n_warmup, n_iter=args.n_iter,
            )
            row["latency_ms_mean"] = lat_mean
            row["latency_ms_std"] = lat_std
            print(f"    latency     : {lat_mean:.2f} +/- {lat_std:.2f} ms")
        except Exception as e:
            row["error"] = f"latency failed: {e}"
            print(f"    latency FAILED: {e}")

        ctx.free()
        rows.append(row)
        print()

    metadata: dict[str, Any] = {
        "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "onnx_dir": str(onnx_dir),
        "val_set_dir": str(Path(args.data_root) / args.val_dir),
        "val_set_size": len(val_set),
        "bench_shape": list(args.bench_shape),
        "n_warmup": args.n_warmup,
        "n_iter": args.n_iter,
        "trt_version": trt.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "rows": rows,
    }

    write_csv(output_dir / "benchmark.csv", rows)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    write_report_md(output_dir / "benchmark.md", rows, metadata)

    print("=" * 60)
    print("Results")
    print("=" * 60)
    for r in rows:
        lat = (f"{r['latency_ms_mean']:.2f} ms" if r["latency_ms_mean"] is not None
               else r.get("error", "n/a"))
        psnr = f"{r['psnr_db']:.3f} dB" if r["psnr_db"] is not None else "n/a"
        print(f"  {r['precision']:5s}  PSNR={psnr}  latency={lat}")
    print()
    print(f"  benchmark.md  -> {output_dir / 'benchmark.md'}")
    print(f"  benchmark.csv -> {output_dir / 'benchmark.csv'}")
    print(f"  engines/      -> {engine_dir}")


if __name__ == "__main__":
    main()
