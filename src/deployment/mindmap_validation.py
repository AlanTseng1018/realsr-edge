"""Generate deployment validation mind map as PNG.

Outputs a single PNG showing the complete pre-TRT-deployment validation
methodology, organized as a sequential pipeline with cross-cutting concerns.

Run::

    python -m src.deployment.mindmap_validation --output-dir results/deploy_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
import numpy as np

# Use a CJK-capable font available on Windows
for _fname in ["Microsoft JhengHei", "Microsoft YaHei", "DFKai-SB", "MS Gothic"]:
    if any(f.name == _fname for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _fname
        break


# ---------------------------------------------------------------------------
# Mind map data
# ---------------------------------------------------------------------------

PIPELINE = [
    {
        "stage": "① PyTorch\nFP32 基線",
        "color": "#3a7ebf",
        "checks": [
            "Val set per-image PSNR",
            "Output 確定性驗證",
            "作為全程比較基準",
            "記錄模型架構與參數量",
        ],
        "risk": None,
    },
    {
        "stage": "② ONNX\nExport",
        "color": "#e07b2a",
        "checks": [
            "dynamo=False (torch 2.6+坑)",
            "File size sanity check",
            "CUDA EP 數值等價 (atol)",
            "Multi-shape testing",
            "onnx.checker 驗證",
        ],
        "risk": "FAIL：檔案異常小\n→ dynamo exporter\n  沒有 export weights",
    },
    {
        "stage": "③ 量化\n精度驗證",
        "color": "#2a9e4e",
        "checks": [
            "Calibration data 代表性",
            "Deploy-side PSNR ≠ fake-quant",
            "FP16 / INT8 drop 對比",
            "QDQ vs TRT calibrator 選擇",
            "Format 相容性確認",
        ],
        "risk": "FAIL：QDQ INT32 bias\n→ TRT 10 拒絕解析\n  改用 calibrator",
    },
    {
        "stage": "④ TRT Engine\n建置",
        "color": "#9055a8",
        "checks": [
            "FP32 / FP16: from FP32 ONNX",
            "INT8: calibrator-based",
            "Optimization profile 鎖形狀",
            "Parser error 明確捕捉",
            "Engine size sanity check",
        ],
        "risk": "FAIL：engine = None\n→ parser error 沒處理\n  需逐條檢查 log",
    },
    {
        "stage": "⑤ Runtime\n效能驗證",
        "color": "#c0392b",
        "checks": [
            "Active provider 確認",
            "CUDA Events (非 perf_counter)",
            "充分 warmup iterations",
            "Batch timing / n iters",
            "Deploy-side PSNR 再驗",
        ],
        "risk": "FAIL：provider fallback\n→ TRT EP 靜默改回\n  CUDA EP，結果失真",
    },
    {
        "stage": "⑥ 硬體特性\n分析",
        "color": "#16a0b5",
        "checks": [
            "Roofline model",
            "Arithmetic intensity",
            "Memory vs compute-bound",
            "Kernel breakdown (profiler)",
            "Achieved vs peak GFLOPS",
        ],
        "risk": None,
    },
]

CROSS_CUTTING = [
    {
        "name": "Silent Failure\n全程防護",
        "color": "#c0392b",
        "items": [
            "每步驗證 active provider",
            "Memcpy node 數量監控",
            "回傳值不為 None ≠ 成功",
            "Build log 完整閱讀",
        ],
    },
    {
        "name": "量測方法論",
        "color": "#7f5ba0",
        "items": [
            "CUDA Events 而非 wall-clock",
            "perf_counter 對快 engine 失準",
            "Batch timing 攤薄 overhead",
            "結論需有數據而非推測",
        ],
    },
    {
        "name": "自動化\n與文件",
        "color": "#5a8a5a",
        "items": [
            "可重複執行的 benchmark",
            "Engine cache 策略",
            "Lessons learned 記錄",
            "視覺化報告生成",
        ],
    },
]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def rounded_box(ax, x, y, w, h, color, text, fontsize=9,
                text_color="white", alpha=1.0, style="round,pad=0.1"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=style,
                          linewidth=0, facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", zorder=4,
            multialignment="center")


def small_box(ax, x, y, w, h, color, text, fontsize=8):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=0.5, edgecolor=color,
                          facecolor=color + "22", alpha=1.0, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color="#222",
            zorder=4, multialignment="center")


def risk_box(ax, x, y, w, h, text, fontsize=7.5):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=1.0, edgecolor="#e74c3c",
                          facecolor="#fdf0ee", zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color="#c0392b",
            zorder=4, multialignment="center")


def arrow(ax, x1, y1, x2, y2, color="#888", lw=1.5):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12),
                zorder=2)


def line(ax, x1, y1, x2, y2, color="#aaa", lw=1.0, ls="-"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=1)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_mindmap(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(22, 16))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # ---------- Title ----------
    ax.text(11, 15.3, "TRT 部署前完整驗證流程",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="#1a2a4a")
    ax.text(11, 14.8,
            "每個階段均需通過驗證才能進入下一階段  ·  Silent Failure 是最危險的失效模式",
            ha="center", va="center", fontsize=10, color="#555")

    # ---------- Pipeline stages (top half) ----------
    n = len(PIPELINE)
    stage_xs = np.linspace(1.8, 20.2, n)
    stage_y  = 11.8
    stage_w  = 2.6
    stage_h  = 1.1

    for i, (sx, stage) in enumerate(zip(stage_xs, PIPELINE)):
        # Stage header box
        rounded_box(ax, sx, stage_y, stage_w, stage_h,
                    stage["color"], stage["stage"], fontsize=10)

        # Arrow to next
        if i < n - 1:
            arrow(ax, sx + stage_w/2 + 0.05, stage_y,
                  stage_xs[i+1] - stage_w/2 - 0.05, stage_y,
                  color=stage["color"], lw=2.0)

        # Check items below
        checks = stage["checks"]
        check_top = stage_y - stage_h/2 - 0.15
        check_h   = 0.52
        check_w   = stage_w - 0.1
        for j, chk in enumerate(checks):
            cy = check_top - j * (check_h + 0.08) - check_h/2
            small_box(ax, sx, cy, check_w, check_h, stage["color"], chk, fontsize=8)
            if j == 0:
                line(ax, sx, check_top, sx, cy + check_h/2,
                     color=stage["color"], lw=1.0)
            else:
                line(ax, sx, cy + check_h/2 + 0.08, sx, cy + check_h/2,
                     color=stage["color"], lw=1.0)

        # Risk / pitfall box (below checks)
        if stage["risk"]:
            n_checks = len(checks)
            risk_y = check_top - n_checks * (check_h + 0.08) - 0.55
            risk_box(ax, sx, risk_y, check_w, 0.85, stage["risk"], fontsize=7.5)
            line(ax, sx,
                 check_top - n_checks * (check_h + 0.08) + check_h/2 - 0.08,
                 sx, risk_y + 0.43,
                 color="#e74c3c", lw=0.8, ls="--")
            ax.text(sx, risk_y + 0.43 + 0.1, "⚠", ha="center", fontsize=8,
                    color="#e74c3c", zorder=5)

    # ---------- Cross-cutting concerns (bottom strip) ----------
    cc_y_top = 1.9
    cc_header_h = 0.7
    cc_item_h   = 0.42
    cc_w        = 5.8
    cc_xs       = [2.5 + i * 6.5 for i in range(3)]

    ax.text(11, cc_y_top + cc_header_h + 0.35,
            "── 貫穿全流程的橫向關注點 ──",
            ha="center", va="center", fontsize=10,
            color="#555", style="italic")

    for cc, cx in zip(CROSS_CUTTING, cc_xs):
        rounded_box(ax, cx, cc_y_top, cc_w, cc_header_h,
                    cc["color"], cc["name"], fontsize=9.5)
        for k, item in enumerate(cc["items"]):
            iy = cc_y_top - cc_header_h/2 - 0.1 - k * (cc_item_h + 0.06) - cc_item_h/2
            small_box(ax, cx, iy, cc_w - 0.2, cc_item_h, cc["color"], item, fontsize=8)

    # ---------- Horizontal separator ----------
    ax.plot([0.5, 21.5], [2.65, 2.65],
            color="#cccccc", lw=1.0, ls="--", zorder=1)

    # ---------- Legend for risk box ----------
    risk_patch = mpatches.Patch(facecolor="#fdf0ee", edgecolor="#e74c3c",
                                 linewidth=1, label="⚠ 常見失效模式 / 坑")
    ax.legend(handles=[risk_patch], loc="lower right",
              fontsize=8.5, framealpha=0.8, bbox_to_anchor=(0.99, 0.01))

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="results/deploy_report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_mindmap(out / "validation_mindmap.png")


if __name__ == "__main__":
    main()
