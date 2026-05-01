"""TensorRT engine profiler with roofline analysis.

For each precision (FP32, FP16, INT8) this script:

1. Profiles the TRT engine with ``torch.profiler`` (CUDA activity) to capture
   which CUDA kernels run and how long they take.
2. Computes **arithmetic intensity** (FLOP / byte) from the PyTorch model's
   FLOPs (via torchinfo) and estimated memory traffic.
3. Plots each precision as a point on the **roofline model** against RTX 3090's
   peak compute and memory-bandwidth ceilings.
4. Reports the kernel-level breakdown: what fraction of time is compute vs
   memory ops vs overhead.

This turns the black-box observation "INT8 is slower" into a verifiable
statement: "INT8 is memory-bound / compute-bound / overhead-dominated
because the profiler shows X".

Run example::

    python -m src.deployment.profile_trt \\
        --onnx-dir  results/onnx_exports/edsr_200ep \\
        --engine-dir results/trt_benchmark/edsr_200ep/engines \\
        --checkpoint results/runs/20260427_143542_ep200_b16_scale2_realistic/checkpoints/best.pt \\
        --output-dir results/trt_profile/edsr_200ep \\
        --bench-shape 1x3x96x96
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorrt as trt
import torch
import torchinfo
from torch.profiler import ProfilerActivity, profile, record_function

from src.models.edsr import EDSR
from src.deployment.benchmark_trt import TRTContext, build_engine, TRT_LOGGER


# ---------------------------------------------------------------------------
# RTX 3090 hardware specs for roofline
# ---------------------------------------------------------------------------

GPU_SPECS = {
    "name": "NVIDIA GeForce RTX 3090",
    # Peak FLOPS (TFLOPS) — from NVIDIA product page
    "peak_fp32_tflops":  35.58,
    "peak_fp16_tflops": 142.3,   # FP16 Tensor Core
    "peak_int8_tops":   284.6,   # INT8 Tensor Core (TOPS, same unit scale)
    # Memory bandwidth GB/s
    "mem_bw_gbs": 936.2,
}

PRECISION_COLOR = {"FP32": "#4c8cbf", "FP16": "#e8872a", "INT8": "#2ca02c"}


# ---------------------------------------------------------------------------
# FLOPs + memory traffic estimation
# ---------------------------------------------------------------------------

def get_model_flops(
    checkpoint_path: Path,
    input_shape: tuple[int, int, int, int],
) -> dict[str, float]:
    """Return FLOPs and parameter count for the EDSR model."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    train_args = ckpt.get("args", {})

    # Prefer saved training args; fall back to inferring from state dict
    n_feats = train_args.get("n_feats") or state["head.weight"].shape[0]
    n_resblocks = train_args.get("n_resblocks") or sum(
        1 for k in state if k.startswith("body.") and k.endswith(".weight") and "conv1" in k
    )
    scale = train_args.get("scale") or int(round((state["tail.0.weight"].shape[0] / 3) ** 0.5))

    model = EDSR(scale_factor=scale, n_resblocks=n_resblocks, n_feats=n_feats)
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.zeros(input_shape)
    info = torchinfo.summary(model, input_data=dummy, verbose=0)
    return {
        "total_flops": info.total_mult_adds * 2,  # MACs → FLOPs
        "total_params": info.total_params,
        "n_feats": n_feats,
        "n_resblocks": n_resblocks,
        "scale": scale,
    }


def estimate_memory_traffic(
    flops_info: dict[str, float],
    input_shape: tuple[int, int, int, int],
    precision: str,
) -> dict[str, float]:
    """Estimate bytes read + written for one forward pass.

    Uses a simplified analytical model:
    - Activations: each layer reads its input and writes its output.
    - Weights: each conv reads its weights from global memory.
    - Precision multiplier: FP32 = 4B, FP16 = 2B, INT8 = 1B per element.

    This is a lower-bound estimate (ignores cache reuse) but is sufficient
    to place the model on the roofline.
    """
    bytes_per_elem = {"FP32": 4, "FP16": 2, "INT8": 1}[precision]

    n, c, h, w = input_shape
    n_feats = flops_info["n_feats"]
    n_res = flops_info["n_resblocks"]
    scale = flops_info["scale"]

    # Input / output tensors
    input_bytes  = n * c * h * w * 4          # input always FP32
    output_bytes = n * c * (h * scale) * (w * scale) * 4  # output always FP32

    # Intermediate activations (rough: each residual block: 2× (N,F,H,W))
    act_per_block = 2 * n * n_feats * h * w * bytes_per_elem
    act_total = act_per_block * n_res

    # Weights
    weight_bytes = flops_info["total_params"] * bytes_per_elem

    total_bytes = input_bytes + output_bytes + act_total + weight_bytes
    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "activation_bytes": act_total,
        "weight_bytes": weight_bytes,
        "total_bytes": total_bytes,
        "arithmetic_intensity": flops_info["total_flops"] / total_bytes,
    }


# ---------------------------------------------------------------------------
# Kernel profiling with torch.profiler
# ---------------------------------------------------------------------------

def profile_engine(
    ctx: TRTContext,
    bench_shape: tuple[int, int, int, int],
    precision_label: str,
    n_warmup: int = 10,
    n_profile: int = 20,
) -> dict[str, Any]:
    """Profile a TRT engine with torch.profiler and return kernel statistics."""
    sample = np.random.rand(*bench_shape).astype(ctx.inp_np_dtype)

    for _ in range(n_warmup):
        ctx.infer(sample)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(n_profile):
            with record_function(f"trt_infer_{precision_label}"):
                ctx.infer(sample)

    events = prof.key_averages()

    cuda_events = [e for e in events if e.device_time_total > 0]
    cuda_events.sort(key=lambda e: e.device_time_total, reverse=True)

    total_cuda_us = sum(e.device_time_total for e in cuda_events)

    kernel_rows = []
    for e in cuda_events[:15]:
        kernel_rows.append({
            "name": e.key[:80],
            "calls": e.count,
            "cuda_us_total": e.device_time_total,
            "cuda_us_avg": e.device_time_total / max(e.count, 1),
            "pct": 100.0 * e.device_time_total / max(total_cuda_us, 1),
        })

    # Classify kernels: memory ops vs compute vs overhead
    def classify(name: str) -> str:
        n = name.lower()
        if any(x in n for x in ["memcpy", "memset", "copy", "h2d", "d2h", "d2d"]):
            return "memory_transfer"
        if any(x in n for x in ["gemm", "conv", "wmma", "cutlass", "cudnn", "volta",
                                  "ampere", "turing", "tensorop", "imma", "hgemm"]):
            return "compute"
        return "other"

    by_class: dict[str, float] = {"memory_transfer": 0.0, "compute": 0.0, "other": 0.0}
    for e in cuda_events:
        by_class[classify(e.key)] += e.device_time_total

    return {
        "total_cuda_us": total_cuda_us,
        "n_profile_iters": n_profile,
        "avg_iter_cuda_us": total_cuda_us / n_profile,
        "kernel_rows": kernel_rows,
        "by_class_us": by_class,
        "by_class_pct": {k: 100.0 * v / max(total_cuda_us, 1) for k, v in by_class.items()},
    }


# ---------------------------------------------------------------------------
# Roofline plot
# ---------------------------------------------------------------------------

def plot_roofline(
    points: list[dict],   # [{label, ai, achieved_gflops, precision}, ...]
    output_path: Path,
    gpu_specs: dict = GPU_SPECS,
) -> None:
    """Draw roofline model with FP32 / FP16 / INT8 ceilings and benchmark points."""
    fig, ax = plt.subplots(figsize=(10, 6))

    mem_bw = gpu_specs["mem_bw_gbs"]
    peak_fp32 = gpu_specs["peak_fp32_tflops"] * 1e3   # → GFLOPS
    peak_fp16 = gpu_specs["peak_fp16_tflops"] * 1e3
    peak_int8 = gpu_specs["peak_int8_tops"]  * 1e3

    ai_range = np.logspace(-2, 4, 500)

    def roofline(ai, peak):
        return np.minimum(mem_bw * ai, peak)

    ax.loglog(ai_range, roofline(ai_range, peak_fp32),
              color="#4c8cbf", lw=2, ls="-",  label=f"FP32 ceiling ({gpu_specs['peak_fp32_tflops']:.1f} TFLOPS)")
    ax.loglog(ai_range, roofline(ai_range, peak_fp16),
              color="#e8872a", lw=2, ls="--", label=f"FP16 Tensor Core ({gpu_specs['peak_fp16_tflops']:.1f} TFLOPS)")
    ax.loglog(ai_range, roofline(ai_range, peak_int8),
              color="#2ca02c", lw=2, ls=":",  label=f"INT8 Tensor Core ({gpu_specs['peak_int8_tops']:.1f} TOPS)")

    # Memory-bandwidth slope annotation
    ax.axvline(peak_fp32 / mem_bw, color="#4c8cbf", lw=0.8, ls="--", alpha=0.4)
    ax.axvline(peak_fp16 / mem_bw, color="#e8872a", lw=0.8, ls="--", alpha=0.4)
    ax.axvline(peak_int8 / mem_bw, color="#2ca02c", lw=0.8, ls="--", alpha=0.4)

    for pt in points:
        color = PRECISION_COLOR.get(pt["precision"], "black")
        ax.scatter(pt["ai"], pt["achieved_gflops"],
                   color=color, s=120, zorder=5, marker="D")
        ax.annotate(
            f"  {pt['label']}\n  {pt['achieved_gflops']:.1f} GFLOPS\n  AI={pt['ai']:.2f}",
            (pt["ai"], pt["achieved_gflops"]),
            fontsize=8, color=color,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=11)
    ax.set_title(f"Roofline Model — {gpu_specs['name']}\n"
                 f"Memory BW: {mem_bw} GB/s", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1, peak_int8 * 2)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  roofline saved -> {output_path.name}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    path: Path,
    results: list[dict],
    gpu_specs: dict,
    bench_shape: tuple,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# TensorRT Profiling Report\n\n")
        f.write(f"- **Generated**: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- **Hardware**: {gpu_specs['name']}\n")
        f.write(f"- **Bench shape**: {list(bench_shape)}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Precision | Latency (ms) | Achieved GFLOPS | Arith. Intensity | "
                "Ridge Point | Region |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for r in results:
            ridge = gpu_specs["mem_bw_gbs"] * r["ai"] / 1e3
            peak  = {"FP32": gpu_specs["peak_fp32_tflops"],
                     "FP16": gpu_specs["peak_fp16_tflops"],
                     "INT8": gpu_specs["peak_int8_tops"]}[r["precision"]] * 1e3
            region = "memory-bound" if r["achieved_gflops"] < ridge else "compute-bound"
            f.write(f"| `{r['precision']}` | {r['latency_ms']:.2f} | "
                    f"{r['achieved_gflops']:.1f} | {r['ai']:.2f} | "
                    f"{ridge:.1f} | **{region}** |\n")
        f.write("\n")

        f.write("> **How to read**: if Achieved GFLOPS << Ridge Point, the model is\n")
        f.write("> memory-bound (bottleneck = DRAM bandwidth). If Achieved GFLOPS ~= Peak,\n")
        f.write("> the model is compute-bound. Both are ceiling-limited; everything else\n")
        f.write("> (e.g. kernel launch overhead) shows as Achieved GFLOPS << both ceilings.\n\n")

        # Per-precision kernel breakdown
        f.write("## Kernel breakdown\n\n")
        for r in results:
            f.write(f"### {r['precision']}\n\n")
            bp = r["profiler"]["by_class_pct"]
            f.write(f"- Compute kernels : {bp['compute']:.1f}%\n")
            f.write(f"- Memory transfer  : {bp['memory_transfer']:.1f}%\n")
            f.write(f"- Other / overhead : {bp['other']:.1f}%\n\n")
            f.write("Top CUDA kernels (by total device time):\n\n")
            f.write("| Kernel | Calls | Avg (us) | Share |\n")
            f.write("|---|---:|---:|---:|\n")
            for row in r["profiler"]["kernel_rows"]:
                f.write(f"| `{row['name']}` | {row['calls']} | "
                        f"{row['cuda_us_avg']:.1f} | {row['pct']:.1f}% |\n")
            f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        for r in results:
            ridge = gpu_specs["mem_bw_gbs"] * r["ai"] / 1e3
            region = "memory-bound" if r["achieved_gflops"] < ridge else "compute-bound"
            gap = ridge / max(r["achieved_gflops"], 1e-9)
            f.write(f"**{r['precision']}**: Achieved {r['achieved_gflops']:.1f} GFLOPS, "
                    f"ridge point {ridge:.1f} GFLOPS — **{region}**")
            if region == "memory-bound":
                f.write(f". Running at {100/gap:.0f}% of memory-bandwidth ceiling.\n")
            else:
                f.write(".\n")
        f.write("\n")
        f.write("*See `roofline.png` for the visual summary.*\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_bench_shape(s: str) -> tuple[int, int, int, int]:
    parts = s.lower().split("x")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"must be NxCxHxW, got {s!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile TRT engines + roofline analysis.")
    p.add_argument("--engine-dir",   type=str, required=True,
                   help="Folder with edsr_fp32.engine, edsr_fp16.engine, edsr_int8.engine")
    p.add_argument("--onnx-dir",     type=str, required=True,
                   help="Folder with edsr_fp32.onnx (used if engine missing)")
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="PyTorch checkpoint (.pt) for FLOPs calculation")
    p.add_argument("--output-dir",   type=str, required=True)
    p.add_argument("--bench-shape",  type=parse_bench_shape, default=(1, 3, 96, 96))
    p.add_argument("--n-warmup",     type=int, default=10)
    p.add_argument("--n-profile",    type=int, default=20)
    p.add_argument("--max-workspace-gb", type=float, default=2.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    engine_dir = Path(args.engine_dir)
    onnx_dir   = Path(args.onnx_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TensorRT Profiler + Roofline Analysis")
    print("=" * 60)
    print(f"  engine-dir  : {engine_dir}")
    print(f"  bench shape : {args.bench_shape}")
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print()

    # FLOPs + memory traffic
    print("[Step 1] Computing FLOPs from checkpoint ...")
    flops_info = get_model_flops(Path(args.checkpoint), args.bench_shape)
    print(f"  FLOPs      : {flops_info['total_flops'] / 1e9:.2f} GFLOPs")
    print(f"  Params     : {flops_info['total_params'] / 1e6:.2f} M")
    print()

    mem_traffic = {
        prec: estimate_memory_traffic(flops_info, args.bench_shape, prec)
        for prec in ["FP32", "FP16", "INT8"]
    }
    for prec, mt in mem_traffic.items():
        print(f"  {prec} arith. intensity: {mt['arithmetic_intensity']:.2f} FLOP/byte "
              f"(ridge @ {GPU_SPECS['mem_bw_gbs'] * mt['arithmetic_intensity'] / 1e3:.1f} GFLOPS)")
    print()

    # Engines to profile
    fp32_engine = engine_dir / "edsr_fp32.engine"
    fp16_engine = engine_dir / "edsr_fp16.engine"
    int8_engine = engine_dir / "edsr_int8.engine"
    fp32_onnx   = onnx_dir / "edsr_fp32.onnx"

    configs = [
        ("FP32", fp32_engine, fp32_onnx, "fp32"),
        ("FP16", fp16_engine, fp32_onnx, "fp16"),
        ("INT8", int8_engine, fp32_onnx, "int8"),
    ]

    results: list[dict[str, Any]] = []
    roofline_points: list[dict] = []

    for precision, engine_path, onnx_path, trt_prec in configs:
        print(f"[{precision}] profiling ...")

        engine = build_engine(
            onnx_path, trt_prec, args.bench_shape,
            engine_cache_path=engine_path,
            max_workspace_gb=args.max_workspace_gb,
        )
        if engine is None:
            print(f"  SKIP (engine build failed)\n")
            continue

        ctx = TRTContext(engine, args.bench_shape)

        # Precise wall-clock latency via CUDA events
        sample = np.random.rand(*args.bench_shape).astype(ctx.inp_np_dtype)
        for _ in range(args.n_warmup):
            ctx.infer(sample)
        t_start = torch.cuda.Event(enable_timing=True)
        t_end   = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for _ in range(50):
            ctx.infer(sample)
        t_end.record()
        torch.cuda.synchronize()
        latency_ms = t_start.elapsed_time(t_end) / 50.0

        # torch.profiler kernel breakdown
        prof_data = profile_engine(ctx, args.bench_shape, precision,
                                   n_warmup=args.n_warmup, n_profile=args.n_profile)

        # Achieved performance
        flops = flops_info["total_flops"]
        achieved_gflops = flops / (latency_ms * 1e-3) / 1e9
        ai = mem_traffic[precision]["arithmetic_intensity"]

        print(f"  latency        : {latency_ms:.2f} ms")
        print(f"  achieved GFLOPS: {achieved_gflops:.1f}")
        print(f"  arith. intensity: {ai:.2f} FLOP/byte")
        bpc = prof_data["by_class_pct"]
        print(f"  compute kernels: {bpc['compute']:.1f}%  "
              f"memory transfer: {bpc['memory_transfer']:.1f}%  "
              f"other: {bpc['other']:.1f}%")
        print()

        ctx.free()

        r = {
            "precision": precision,
            "latency_ms": latency_ms,
            "achieved_gflops": achieved_gflops,
            "ai": ai,
            "flops": flops,
            "mem_traffic": mem_traffic[precision],
            "profiler": prof_data,
        }
        results.append(r)
        roofline_points.append({
            "label": precision,
            "precision": precision,
            "ai": ai,
            "achieved_gflops": achieved_gflops,
        })

    if not results:
        raise SystemExit("No engines profiled successfully.")

    print("[Step 3] Writing outputs ...")
    plot_roofline(roofline_points, output_dir / "roofline.png")
    write_report(output_dir / "profile_report.md", results, GPU_SPECS, args.bench_shape)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
            "gpu": GPU_SPECS,
            "bench_shape": list(args.bench_shape),
            "flops_info": flops_info,
            "results": [
                {k: v for k, v in r.items() if k != "profiler"}
                for r in results
            ],
        }, f, indent=2, default=str)

    print(f"  profile_report.md -> {output_dir / 'profile_report.md'}")
    print(f"  roofline.png      -> {output_dir / 'roofline.png'}")
    print(f"  metadata.json     -> {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
