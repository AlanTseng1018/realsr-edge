"""Post-training quantization & precision analysis pipeline runner.

Executes the complete validation pipeline in order.
Each stage must pass before the next runs.

Usage::

    # Full pipeline (first run, builds everything)
    python -m src.deployment.run_pipeline

    # Skip ONNX export if already done
    python -m src.deployment.run_pipeline --skip-export

    # Skip to visualization only
    python -m src.deployment.run_pipeline --only-viz
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


CHECKPOINT   = "results/runs/20260427_143542_ep200_b16_scale2_realistic/checkpoints/best.pt"
ONNX_DIR     = "results/onnx_exports/edsr_200ep"
ORT_OUT      = "results/onnx_benchmark/edsr_200ep_trt_v2"
TRT_OUT      = "results/trt_benchmark/edsr_200ep_v2"
PROFILE_OUT  = "results/trt_profile/edsr_200ep"
REPORT_OUT   = "results/deploy_report/edsr_200ep"
LAYER_OUT    = "results/layer_analysis/edsr_200ep"
DATA_ROOT    = "data/DIV2K"
BENCH_SHAPE  = "1x3x96x96"


def run(cmd: list[str], stage: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {stage}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m"] + cmd,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] {stage} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] {stage} ({elapsed:.1f}s)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-layer",   action="store_true", help="Skip per-layer analysis")
    p.add_argument("--skip-export",  action="store_true", help="Skip ONNX export (already done)")
    p.add_argument("--skip-ort",     action="store_true", help="Skip ORT benchmark")
    p.add_argument("--skip-trt",     action="store_true", help="Skip TRT build + benchmark")
    p.add_argument("--skip-profile", action="store_true", help="Skip profiling")
    p.add_argument("--only-viz",     action="store_true", help="Only run visualization")
    p.add_argument("--only-layer",   action="store_true", help="Only run per-layer analysis")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.only_viz:
        args.skip_layer = args.skip_export = args.skip_ort = args.skip_trt = args.skip_profile = True
    if args.only_layer:
        args.skip_export = args.skip_ort = args.skip_trt = args.skip_profile = True

    # ------------------------------------------------------------------ #
    # Stage 0: Per-layer precision analysis (from best.pt, no ONNX)      #
    # ------------------------------------------------------------------ #
    if not args.skip_layer:
        run([
            "src.deployment.analyze_layers",
            "--checkpoint", CHECKPOINT,
            "--output-dir", LAYER_OUT,
            "--data-root",  DATA_ROOT,
            "--val-dir",    "DIV2K_valid_HR",
        ], "Stage 0 / 5 — Per-Layer Precision Analysis  (from best.pt)")

    # ------------------------------------------------------------------ #
    # Stage 1: Export — PyTorch → ONNX (FP32 / FP16 / INT8 static PTQ)  #
    # ------------------------------------------------------------------ #
    if not args.skip_export:
        run([
            "src.deployment.export_pipeline",
            "--checkpoint", CHECKPOINT,
            "--output-dir", ONNX_DIR,
            "--data-root",  DATA_ROOT,
            "--val-dir",    "DIV2K_valid_HR",
        ], "Stage 1 / 5 — ONNX Export  (FP32 + FP16 + INT8 static PTQ)")

    # ------------------------------------------------------------------ #
    # Stage 2: ORT Benchmark — CUDA EP / CPU EP / TRT EP                 #
    # Purpose : baseline PSNR + latency; exposes silent TRT fallback      #
    # ------------------------------------------------------------------ #
    if not args.skip_ort:
        run([
            "src.deployment.benchmark_onnx",
            "--onnx-dir",   ONNX_DIR,
            "--output-dir", ORT_OUT,
            "--data-root",  DATA_ROOT,
            "--val-dir",    "DIV2K_valid_HR",
            "--providers",  "tensorrt", "cuda", "cpu",
            "--bench-shape", BENCH_SHAPE,
        ], "Stage 2 / 5 — ORT Benchmark  (CUDA EP / TRT EP / CPU EP)")

    # ------------------------------------------------------------------ #
    # Stage 3: Native TRT Engine Build + Benchmark                        #
    # Purpose : true TRT performance; INT8 via calibrator (not QDQ)       #
    # ------------------------------------------------------------------ #
    if not args.skip_trt:
        run([
            "src.deployment.benchmark_trt",
            "--onnx-dir",   ONNX_DIR,
            "--output-dir", TRT_OUT,
            "--data-root",  DATA_ROOT,
            "--val-dir",    "DIV2K_valid_HR",
            "--bench-shape", BENCH_SHAPE,
        ], "Stage 3 / 5 — Native TRT Engine  (FP32 / FP16 / INT8 calibrator)")

    # ------------------------------------------------------------------ #
    # Stage 4: Profiling + Roofline                                       #
    # Purpose : WHY each precision is fast or slow                        #
    # ------------------------------------------------------------------ #
    if not args.skip_profile:
        run([
            "src.deployment.profile_trt",
            "--engine-dir",  str(Path(TRT_OUT) / "engines"),
            "--onnx-dir",    ONNX_DIR,
            "--checkpoint",  CHECKPOINT,
            "--output-dir",  PROFILE_OUT,
            "--bench-shape", BENCH_SHAPE,
        ], "Stage 4 / 5 — Profiling + Roofline")

    # ------------------------------------------------------------------ #
    # Stage 5: Visualize                                                  #
    # Purpose : single-PNG summary of all results                         #
    # ------------------------------------------------------------------ #
    run([
        "src.deployment.visualize_results",
        "--ort-benchmark", ORT_OUT,
        "--trt-benchmark", TRT_OUT,
        "--trt-profile",   PROFILE_OUT,
        "--output-dir",    REPORT_OUT,
    ], "Stage 5 / 5 — Visualization")

    print(f"\n{'='*60}")
    print("  Pipeline complete")
    print(f"{'='*60}")
    print(f"  Summary PNG : {REPORT_OUT}/deploy_summary.png")
    print(f"  Mind map    : results/deploy_report/validation_mindmap.png")


if __name__ == "__main__":
    main()
