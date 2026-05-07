"""Aggregate ONNX benchmark results into a single systematic deployment report.

Reads the raw shootout from ``benchmark_onnx.py`` and produces a richer
``deploy_summary.md`` with multiple views of the same data:

1. **Headline latency matrix** -- precision × provider, single table.
2. **Headline accuracy table** -- PSNR drop per precision.
3. **Per-provider deep dives** -- one section per provider, comparing
   precisions on that hardware path.
4. **Per-precision deep dives** -- one section per precision, comparing
   providers for that data type.
5. **Speedup matrix** -- normalized to FP32 same-provider baseline.
6. **Deploy recommendation matrix** -- "if your target is X, pick Y".

The aggregator is deliberately READ-ONLY -- it never re-runs benchmarks
or models. It just rearranges existing CSV data into a deploy-team
friendly markdown report. Re-runs are cheap: change anything in the
underlying benchmark and re-run this script in seconds.

Run example::

    python -m src.deployment.deploy_summary --benchmark-csv results/onnx_benchmark/edsr_200ep_full/benchmark.csv --output results/onnx_benchmark/edsr_200ep_full/deploy_summary.md
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def precision_of(onnx_name: str) -> str:
    n = onnx_name.lower()
    if "fp16" in n:
        return "FP16"
    if "int8" in n:
        return "INT8"
    return "FP32"


def load_benchmark_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Convert numeric fields. ``ssim`` is optional for backwards
            # compat with old benchmark.csv files that pre-date the SSIM
            # extension; rows missing it just stay None and the writer
            # renders "n/a".
            for k in ("psnr_db", "ssim", "latency_ms_mean", "latency_ms_std",
                      "size_mb", "session_build_ms"):
                v = r.get(k, "")
                r[k] = float(v) if v not in ("", None) else None
            rows.append(r)
    return rows


def load_metadata_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pivot helpers
# ---------------------------------------------------------------------------

def fmt_latency(row: dict[str, Any]) -> str:
    if row.get("latency_ms_mean") is None:
        err = row.get("error") or "n/a"
        return f"_{err[:30]}_"
    return f"{row['latency_ms_mean']:.2f} +/- {row['latency_ms_std']:.2f}"


def pivot_latency(rows: list[dict[str, Any]],
                  precisions: list[str],
                  providers: list[str]) -> list[list[str]]:
    """Build a 2D table cells[precision_idx][provider_idx] of latency strings."""
    by_pp: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        prec = precision_of(r["onnx"])
        prov = r["provider"]
        by_pp[(prec, prov)] = r
    return [[fmt_latency(by_pp[(p, pv)]) if (p, pv) in by_pp else "—"
             for pv in providers] for p in precisions]


def speedup_vs_fp32(rows: list[dict[str, Any]],
                    precisions: list[str],
                    providers: list[str]) -> list[list[str]]:
    """Per-cell speedup vs FP32 on the same provider."""
    by_pp: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        by_pp[(precision_of(r["onnx"]), r["provider"])] = r

    cells: list[list[str]] = []
    for prec in precisions:
        row: list[str] = []
        for prov in providers:
            cur = by_pp.get((prec, prov))
            base = by_pp.get(("FP32", prov))
            if (cur is None or base is None
                    or cur.get("latency_ms_mean") is None
                    or base.get("latency_ms_mean") is None):
                row.append("n/a")
                continue
            ratio = base["latency_ms_mean"] / cur["latency_ms_mean"]
            if abs(ratio - 1.0) < 0.02:
                row.append("baseline")
            elif ratio >= 1.0:
                row.append(f"**{ratio:.2f}x faster**")
            else:
                row.append(f"{1.0 / ratio:.2f}x slower")
        cells.append(row)
    return cells


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------

def write_md(
    out_path: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    benchmark_dir: Path,
) -> None:
    precisions = ["FP32", "FP16", "INT8"]
    providers_all = sorted({r["provider"] for r in rows},
                           key=lambda p: ["tensorrt", "cuda", "cpu"].index(p)
                           if p in ("tensorrt", "cuda", "cpu") else 99)
    by_pp = {(precision_of(r["onnx"]), r["provider"]): r for r in rows}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        # ---------- Header ----------
        f.write("# Deployment Performance Summary\n\n")
        f.write("Aggregated view of the ONNX runtime benchmark, organized so a "
                "deploy-team reader can answer the three questions:\n\n")
        f.write("1. **What latency / accuracy do I get at each precision?**\n")
        f.write("2. **How does my chosen runtime affect the answer?**\n")
        f.write("3. **What should I deploy on my target hardware?**\n\n")

        # ---------- Test configuration ----------
        f.write("## 1. Test configuration\n\n")
        f.write(f"- **Generated**: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        if metadata:
            f.write(f"- **Source benchmark**: `{metadata.get('onnx_dir', 'n/a')}`\n")
            f.write(f"- **Validation set**: `{metadata.get('val_set_dir', 'n/a')}` "
                    f"({metadata.get('val_set_size', 'n/a')} images, realistic degradation)\n")
            shape = metadata.get("bench_shape")
            if shape:
                f.write(f"- **Latency input shape**: `{tuple(shape)}` "
                        f"({metadata.get('n_warmup', 'n/a')} warmup + "
                        f"{metadata.get('n_iter', 'n/a')} timed iters)\n")
            f.write(f"- **Hardware**: {metadata.get('device_name', 'n/a')}\n")
            f.write(f"- **ORT version**: {metadata.get('ort_version', 'n/a')}\n")
            f.write(f"- **Available EPs**: "
                    f"{', '.join('`' + p + '`' for p in metadata.get('ort_available_providers', []))}\n")
        f.write("\n")

        # ---------- Headline latency matrix ----------
        f.write("## 2. Headline latency matrix (ms, lower is better)\n\n")
        f.write("| Precision \\ Provider | "
                + " | ".join(f"`{p}`" for p in providers_all) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(providers_all)) + "|\n")
        cells = pivot_latency(rows, precisions, providers_all)
        for prec, row in zip(precisions, cells):
            f.write(f"| **{prec}** | " + " | ".join(row) + " |\n")
        f.write("\n")

        # ---------- Speedup matrix ----------
        f.write("## 3. Speedup vs FP32 (same provider)\n\n")
        f.write("Per cell: `latency(FP32 same-EP) / latency(this cell)`. "
                "**Bold** = faster than FP32 same EP.\n\n")
        f.write("| Precision \\ Provider | "
                + " | ".join(f"`{p}`" for p in providers_all) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(providers_all)) + "|\n")
        sp = speedup_vs_fp32(rows, precisions, providers_all)
        for prec, row in zip(precisions, sp):
            f.write(f"| **{prec}** | " + " | ".join(row) + " |\n")
        f.write("\n")

        # ---------- Accuracy ----------
        f.write("## 4. Accuracy per precision (PSNR + SSIM on val set)\n\n")
        f.write("Both metrics are provider-invariant within float-rounding "
                "noise; we report the mean across providers per precision and "
                "the spread (max - min) so any unexpected provider divergence "
                "shows up here. PSNR is the headline number; SSIM is the "
                "perceptual cross-check.\n\n")
        f.write("| Precision | mean PSNR (dB) | PSNR spread | PSNR drop vs FP32 | "
                "mean SSIM | SSIM spread | SSIM drop vs FP32 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")

        fp32_psnrs = [r["psnr_db"] for r in rows
                      if precision_of(r["onnx"]) == "FP32"
                      and r.get("psnr_db") is not None]
        fp32_ssims = [r["ssim"] for r in rows
                      if precision_of(r["onnx"]) == "FP32"
                      and r.get("ssim") is not None]
        fp32_psnr_mean = sum(fp32_psnrs) / len(fp32_psnrs) if fp32_psnrs else None
        fp32_ssim_mean = sum(fp32_ssims) / len(fp32_ssims) if fp32_ssims else None

        for prec in precisions:
            psnrs = [r["psnr_db"] for r in rows
                     if precision_of(r["onnx"]) == prec
                     and r.get("psnr_db") is not None]
            ssims = [r["ssim"] for r in rows
                     if precision_of(r["onnx"]) == prec
                     and r.get("ssim") is not None]
            if not psnrs:
                f.write(f"| {prec} | n/a | n/a | n/a | n/a | n/a | n/a |\n")
                continue

            psnr_mean = sum(psnrs) / len(psnrs)
            psnr_spread = max(psnrs) - min(psnrs)
            psnr_drop = ((fp32_psnr_mean - psnr_mean)
                         if fp32_psnr_mean is not None else 0.0)

            if ssims:
                ssim_mean = sum(ssims) / len(ssims)
                ssim_spread = max(ssims) - min(ssims)
                ssim_drop = ((fp32_ssim_mean - ssim_mean)
                             if fp32_ssim_mean is not None else 0.0)
                ssim_mean_s = f"{ssim_mean:.4f}"
                ssim_spread_s = f"{ssim_spread:.4f}"
                ssim_drop_s = f"{ssim_drop:+.4f}"
            else:
                ssim_mean_s = ssim_spread_s = ssim_drop_s = "n/a"

            f.write(f"| **{prec}** | {psnr_mean:.3f} | {psnr_spread:.3f} | "
                    f"{psnr_drop:+.3f} | "
                    f"{ssim_mean_s} | {ssim_spread_s} | {ssim_drop_s} |\n")
        f.write("\n")

        # Keep the legacy variable name for downstream sections
        fp32_mean = fp32_psnr_mean

        # ---------- Per-provider deep dive ----------
        f.write("## 5. Per-provider deep dive\n\n")
        for prov in providers_all:
            f.write(f"### `{prov}`\n\n")
            f.write("| Precision | PSNR (dB) | PSNR drop | SSIM | SSIM drop | "
                    "Latency (ms) | Speedup vs FP32 | Build (ms) | Notes |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
            base_lat = (by_pp.get(("FP32", prov), {}) or {}).get("latency_ms_mean")
            base_ssim = (by_pp.get(("FP32", prov), {}) or {}).get("ssim")
            for prec in precisions:
                r = by_pp.get((prec, prov))
                if r is None:
                    f.write(f"| {prec} | n/a | n/a | n/a | n/a | "
                            f"n/a | n/a | n/a | (no row) |\n")
                    continue
                psnr = (f"{r['psnr_db']:.3f}"
                        if r.get("psnr_db") is not None else "n/a")
                drop = ("n/a" if r.get("psnr_db") is None or fp32_mean is None
                        else f"{fp32_mean - r['psnr_db']:+.3f}")
                ssim_v = (f"{r['ssim']:.4f}"
                          if r.get("ssim") is not None else "n/a")
                ssim_d = ("n/a" if r.get("ssim") is None or base_ssim is None
                          else f"{base_ssim - r['ssim']:+.4f}")
                lat = fmt_latency(r)
                if (r.get("latency_ms_mean") is None
                        or base_lat is None or base_lat <= 0):
                    sp_str = "n/a"
                else:
                    ratio = base_lat / r["latency_ms_mean"]
                    sp_str = ("baseline" if abs(ratio - 1.0) < 0.02
                              else (f"**{ratio:.2f}x faster**" if ratio >= 1.0
                                    else f"{1.0 / ratio:.2f}x slower"))
                build = (f"{r['session_build_ms']:.0f}"
                         if r.get("session_build_ms") is not None else "n/a")
                note = r.get("error") or ""
                f.write(f"| {prec} | {psnr} | {drop} | {ssim_v} | {ssim_d} | "
                        f"{lat} | {sp_str} | {build} | {note[:40]} |\n")
            f.write("\n")

        # ---------- Per-precision deep dive ----------
        f.write("## 6. Per-precision deep dive\n\n")
        for prec in precisions:
            f.write(f"### {prec}\n\n")
            f.write("| Provider | PSNR (dB) | SSIM | Latency (ms) | "
                    "Size (MB) | Active EP | Notes |\n")
            f.write("|---|---:|---:|---:|---:|---|---|\n")
            for prov in providers_all:
                r = by_pp.get((prec, prov))
                if r is None:
                    f.write(f"| `{prov}` | n/a | n/a | n/a | n/a | n/a "
                            f"| (no row) |\n")
                    continue
                psnr = (f"{r['psnr_db']:.3f}"
                        if r.get("psnr_db") is not None else "n/a")
                ssim_v = (f"{r['ssim']:.4f}"
                          if r.get("ssim") is not None else "n/a")
                lat = fmt_latency(r)
                size = (f"{r['size_mb']:.2f}"
                        if r.get("size_mb") is not None else "n/a")
                active = r.get("active_provider", "n/a")
                note = r.get("error") or ""
                f.write(f"| `{prov}` | {psnr} | {ssim_v} | {lat} | {size} | "
                        f"`{active}` | {note[:40]} |\n")
            f.write("\n")

        # ---------- Deploy recommendation matrix ----------
        f.write("## 7. Deploy recommendation matrix\n\n")
        # Compute the best (precision, provider) per latency
        best_lat: tuple[str, str, float] | None = None
        for prec in precisions:
            for prov in providers_all:
                r = by_pp.get((prec, prov))
                if r is None or r.get("latency_ms_mean") is None:
                    continue
                if best_lat is None or r["latency_ms_mean"] < best_lat[2]:
                    best_lat = (prec, prov, r["latency_ms_mean"])
        f.write(f"**Lowest latency on this hardware**: "
                f"`{best_lat[0]}` on `{best_lat[1]}` -> "
                f"{best_lat[2]:.2f} ms\n\n" if best_lat else "")
        f.write("| Target | Best precision | Provider | Reason |\n")
        f.write("|---|---|---|---|\n")
        f.write("| **NVIDIA Jetson / Orin / Drive** | "
                "FP16 (or INT8 on larger models) | TensorRT | "
                "Tensor Core FP16 saturates on small SR models; INT8 "
                "wins only for larger / batched workloads |\n")
        f.write("| **NVIDIA desktop edge** | FP16 | TensorRT | "
                "Same as Jetson reasoning |\n")
        f.write("| **x86 CPU server / edge** | FP32 (or INT8 on larger models) | "
                "ORT CPU | All precisions roughly equivalent for small "
                "models on CPU; VNNI INT8 wins on big models |\n")
        f.write("| **Mobile / TV SoC NPU** | INT8 | Vendor SDK | "
                "NPU silicon is INT8-native, memory-bound; this benchmark "
                "is reference only -- vendor SDK gives final numbers |\n")
        f.write("\n")

        # ---------- Notes / caveats ----------
        f.write("## 8. Notes and caveats\n\n")
        f.write("### Why INT8 isn't always faster on GPU\n\n")
        f.write("For this 1.37M-param SR model on a consumer Tensor Core GPU, "
                "FP16 outperforms INT8 because:\n\n")
        f.write("- Tensor Core FP16 saturates at small batch / small model sizes "
                "(no compute headroom for INT8 to fill).\n")
        f.write("- INT8 adds Q/DQ ops + scale arithmetic; the overhead "
                "dominates for small graphs.\n")
        f.write("- INT8's main lever -- 4× weight compression / memory "
                "bandwidth -- is only decisive on memory-bound hardware "
                "(NPUs, mobile DSPs). On Tensor Core, compute is rarely the "
                "bottleneck for small SR models.\n\n")
        f.write("**INT8 expected to win** on: larger models (5M+ params), "
                "higher batch sizes, NPU silicon, or 4K input resolution "
                "(where memory bandwidth matters).\n\n")
        f.write("### TensorRT INT8 calibration must be symmetric\n\n")
        f.write("ORT's `quantize_static` defaults to **asymmetric** (non-zero "
                "zero point). TensorRT EP rejects that with "
                "\"Non-zero zero point is not supported\". The export "
                "pipeline forces `ActivationSymmetric=True` + "
                "`WeightSymmetric=True` + ``quant_pre_process`` to make the "
                "INT8 ONNX TRT-compatible. The trade-off: ~0.05 dB more "
                "PSNR drop than asymmetric.\n\n")
        f.write("### ORT CUDA EP + INT8 anti-pattern\n\n")
        f.write("ORT's CUDA EP doesn't have native INT8 conv kernels for "
                "QDQ format. It runs Q/DQ ops on CPU, conv on GPU FP32, "
                "and inserts Memcpy nodes between. Result is slower than "
                "FP32 CUDA. The fix is using TensorRT EP (this benchmark "
                "shows it works) or vendor NPU SDKs.\n\n")

        # ---------- Cross-references ----------
        f.write("## 9. Cross-references\n\n")
        f.write(f"- Raw benchmark: `{benchmark_dir}/benchmark.md` and "
                f"`benchmark.csv`\n")
        f.write("- Accuracy analysis (PyTorch fake-quant): "
                "`results/quantization/200ep_with_report/report.md`\n")
        f.write("- Calibration scheme ablation: "
                "`results/quantization/calibration_ablation/calibration_ablation.md`\n")
        f.write("- ONNX export: "
                "`results/onnx_exports/edsr_200ep/README.md`\n")
        f.write("- Deploy methodology framework: "
                "`learning/deployment_methodology.md`\n")
        f.write("- Lessons learned: "
                "`learning/deployment_lessons_learned.md`\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate benchmark.csv into deploy_summary.md"
    )
    p.add_argument("--benchmark-csv", type=str, required=True)
    p.add_argument("--metadata-json", type=str, default=None,
                   help="Path to metadata.json (default: benchmark.csv's sibling)")
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench_csv = Path(args.benchmark_csv)
    meta_json = Path(args.metadata_json) if args.metadata_json else bench_csv.parent / "metadata.json"
    out_path = Path(args.output)

    rows = load_benchmark_csv(bench_csv)
    metadata = load_metadata_json(meta_json)
    write_md(out_path, rows, metadata, benchmark_dir=bench_csv.parent)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
