from __future__ import annotations

import argparse
import csv
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from ptxbench.config import REPO_ROOT


BG = "#FAF7F2"
GRID = "#D7D1C7"
TEXT = "#1F2430"
SUBTLE = "#6B7280"
PTX = "#1D5F8C"
CUDA = "#1F8A70"
PTX_HYBRID = "#7A4EAB"
ACCENT = "#D97706"
BASE = "#7C8799"
HYBRID = "#7A4EAB"
FAILURE_COLORS = {
    "success": CUDA,
    "compile": "#C84C3A",
    "assemble": "#E89B2F",
    "runtime": "#A8556E",
    "correctness": "#7251B5",
    "oom": "#4B5563",
    "evaluator_crash": "#94A3B8",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finalize_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.8, alpha=0.35, color=GRID)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def _save(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=240, bbox_inches="tight")


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", dy: float = 0.01) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )


def render_coverage_summary(input_dir: Path, output_dir: Path) -> Path:
    rows = _read_csv(input_dir / "coverage_summary.csv")
    return _render_coverage_summary_rows(
        rows,
        output_dir / "coverage_summary.png",
        title="Interactive Coverage Summary",
        footer_note="Color = backend lane; dark = correctness, light = fast@1.0",
    )


def _render_coverage_summary_rows(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    title: str,
    footer_note: str | None = None,
    show_geomean_labels: bool = True,
) -> Path:
    labels = [row["label"] for row in rows]
    correctness = np.array([float(row["correctness_rate"]) for row in rows], dtype=float)
    fast_p_1 = np.array([float(row["fast_p_1"]) for row in rows], dtype=float)
    geomean = np.array([float(row["geomean_correct_only"]) for row in rows], dtype=float)
    backend_colors = []
    for row in rows:
        if row["backend"] == "ptx":
            backend_colors.append(PTX)
        elif row["backend"] == "cuda":
            backend_colors.append(CUDA)
        else:
            backend_colors.append(PTX_HYBRID)

    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    bars1 = ax.bar(x - width / 2, correctness, width, label="Correctness Rate", color=backend_colors)
    bars2 = ax.bar(x + width / 2, fast_p_1, width, label="fast@1.0", color=backend_colors, alpha=0.45)

    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)

    if show_geomean_labels:
        for xpos, value in zip(x, geomean, strict=True):
            ax.text(
                xpos,
                max(correctness.max(), fast_p_1.max()) + 0.08,
                f"geomean speedup {value:.3f}x",
                ha="center",
                va="bottom",
                fontsize=9,
                color=SUBTLE,
            )

    ax.set_title(title)
    ax.set_ylabel("Fraction of Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(correctness.max(), fast_p_1.max()) + 0.18)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    if footer_note:
        fig.text(
            0.08,
            0.02,
            footer_note,
            fontsize=9,
            color=SUBTLE,
        )
    _finalize_axes(ax, "y")
    fig.tight_layout(rect=(0, 0.05 if footer_note else 0, 1, 1))

    _save(fig, output_path)
    plt.close(fig)
    return output_path


def render_coverage_summary_ptx_only(input_dir: Path, output_dir: Path) -> Path:
    rows = [
        row
        for row in _read_csv(input_dir / "coverage_summary.csv")
        if row["backend"] == "ptx"
    ]
    return _render_coverage_summary_rows(
        rows,
        output_dir / "coverage_summary_ptx_only.png",
        title="PTX Coverage Summary",
        footer_note="fast@1.0 and geomean speedup are measured versus PyTorch eager",
        show_geomean_labels=False,
    )


def render_overlap_comparison(input_dir: Path, output_dir: Path) -> Path:
    rows = _read_csv(input_dir / "overlap_comparison.csv")
    labels = [_wrap(f"{row['base_run']} {row['backend'].upper()} {row['level'].replace('level', 'L')}", 28) for row in rows]
    interactive_correct = np.array([float(row["interactive_correct"]) for row in rows], dtype=float)
    base_correct = np.array([float(row["base_correct"]) for row in rows], dtype=float)
    shared = np.array([float(row["shared_tasks"]) for row in rows], dtype=float)
    geomean = np.array([float(row["interactive_vs_base_geomean_speedup"]) for row in rows], dtype=float)

    y = np.arange(len(labels), dtype=float)
    height = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.barh(y - height / 2, interactive_correct, height, label="Interactive Correct", color=CUDA)
    ax.barh(y + height / 2, base_correct, height, label="Base Correct", color="#C84C3A")

    for idx, (shared_count, geo) in enumerate(zip(shared, geomean, strict=True)):
        ax.text(
            max(interactive_correct[idx], base_correct[idx]) + 0.12,
            idx,
            f"shared={int(shared_count)}  geo={geo:.3f}",
            va="center",
            fontsize=9,
            color=SUBTLE,
        )

    ax.set_title("Interactive vs Existing Base Runs")
    ax.text(
        0.0,
        1.03,
        "Only shared task overlaps are compared directly",
        transform=ax.transAxes,
        fontsize=10,
        color=SUBTLE,
    )
    ax.set_xlabel("Correct Tasks on Shared Overlap")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend(frameon=False, loc="lower right")
    _finalize_axes(ax, "x")
    ax.set_xlim(0, max(shared) + 1.25)
    fig.tight_layout()

    output_path = output_dir / "overlap_comparison.png"
    _save(fig, output_path)
    plt.close(fig)
    return output_path


def render_failure_breakdown(input_dir: Path, output_dir: Path) -> Path:
    rows = [
        row
        for row in _read_csv(input_dir / "failure_breakdown.csv")
        if not (row["backend"] == "cuda" and row["level"] == "level3")
    ]
    labels = []
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["backend"], row["level"])
        grouped.setdefault(key, {})[row["failure_stage"]] = float(row["count"])
    keys = sorted(grouped)
    labels = [f"{backend.upper()} {level.replace('level', 'L')}" for backend, level in keys]

    all_stages = ["success", "compile", "assemble", "runtime", "correctness", "oom", "evaluator_crash"]
    stages = [
        stage
        for stage in all_stages
        if any(grouped[key].get(stage, 0.0) > 0.0 for key in keys)
    ]

    fig_height = 6.4 if len(stages) <= 5 else 7.0
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(keys), dtype=float)
    totals = np.array(
        [sum(grouped[key].get(stage, 0.0) for stage in all_stages) for key in keys],
        dtype=float,
    )
    left = np.zeros(len(keys), dtype=float)
    for stage in stages:
        raw_values = np.array([grouped[key].get(stage, 0.0) for key in keys], dtype=float)
        values = np.divide(raw_values, totals, out=np.zeros_like(raw_values), where=totals > 0)
        ax.barh(y, values, left=left, label=stage, color=FAILURE_COLORS[stage])
        left += values

    ax.set_title("Failure Breakdown by Backend and Level")
    ax.set_xlabel("Share of Tasks")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    xticks = np.linspace(0.0, 1.0, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(value * 100)}%" for value in xticks])
    _finalize_axes(ax, "x")
    ax.legend(
        frameon=False,
        ncol=min(4, max(2, len(stages))),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_path = output_dir / "failure_breakdown.png"
    _save(fig, output_path)
    plt.close(fig)
    return output_path


def render_family_summary(input_dir: Path, output_dir: Path) -> Path:
    rows = _read_csv(input_dir / "family_summary.csv")
    keep = [row for row in rows if row["level"] in {"level1", "level2", "level3"}]
    keep.sort(key=lambda row: (row["level"], row["family"], row["backend"]))

    labels = [_wrap(f"{row['level'].replace('level', 'L')} {row['family'].replace('_', ' ')}", 14) for row in keep]
    correctness = np.array([float(row["correctness_rate"]) for row in keep], dtype=float)
    fast_p_1 = np.array([float(row["fast_p_1"]) for row in keep], dtype=float)
    colors = [PTX if row["backend"] == "ptx" else CUDA for row in keep]

    x = np.arange(len(labels), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.bar(x - width / 2, correctness, width, color=colors, alpha=0.90)
    ax.bar(x + width / 2, fast_p_1, width, color=colors, alpha=0.45)

    ax.set_title("Family-Level Interactive Performance")
    ax.set_ylabel("Fraction of Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.10)
    _finalize_axes(ax, "y")
    legend_handles = [
        Patch(facecolor=PTX, alpha=0.90, label="PTX correctness"),
        Patch(facecolor=PTX, alpha=0.45, label="PTX fast@1.0"),
        Patch(facecolor=CUDA, alpha=0.90, label="CUDA correctness"),
        Patch(facecolor=CUDA, alpha=0.45, label="CUDA fast@1.0"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=2,
        loc="upper right",
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.tight_layout()

    output_path = output_dir / "family_summary.png"
    _save(fig, output_path)
    plt.close(fig)
    return output_path


def render_family_summary_ptx_l1_selected(input_dir: Path, output_dir: Path) -> Path:
    rows = [
        row
        for row in _read_csv(input_dir / "family_summary.csv")
        if row["backend"] == "ptx"
        and row["level"] == "level1"
        and row["family"] in {"norm", "pooling", "reduction"}
    ]
    rows.sort(key=lambda row: ("norm", "pooling", "reduction").index(row["family"]))

    labels = [f"L1 {row['family']}" for row in rows]
    correctness = np.array([float(row["correctness_rate"]) for row in rows], dtype=float)
    fast_p_1 = np.array([float(row["fast_p_1"]) for row in rows], dtype=float)
    geomean = np.array([float(row["geomean_correct_only"]) for row in rows], dtype=float)

    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    bars1 = ax.bar(x - width / 2, correctness, width, color=PTX, alpha=0.90, label="Correctness Rate")
    bars2 = ax.bar(x + width / 2, fast_p_1, width, color=PTX, alpha=0.45, label="fast@1.0")

    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)

    ax.set_title("PTX Level 1 Family Snapshot")
    ax.set_ylabel("Fraction of Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(correctness.max(), fast_p_1.max()) + 0.18)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.text(
        0.08,
        0.02,
        "Focused view of PTX Level 1 families most affected by structural/runtime failures",
        fontsize=9,
        color=SUBTLE,
    )
    _finalize_axes(ax, "y")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    output_path = output_dir / "family_summary_ptx_l1_selected.png"
    _save(fig, output_path)
    plt.close(fig)
    return output_path


def render_top_wins(input_dir: Path, output_dir: Path) -> Path:
    rows = _read_csv(input_dir / "top_wins.csv")
    rows.sort(key=lambda row: float(row["speedup_vs_torch"]), reverse=True)
    top = rows[:12]

    labels = [f"{row['backend'].upper()} {row['level'].replace('level', 'L')} #{row['problem_id']}" for row in top]
    values = np.array([float(row["speedup_vs_torch"]) for row in top], dtype=float)
    colors = [PTX if row["backend"] == "ptx" else CUDA for row in top]

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    y = np.arange(len(labels), dtype=float)
    bars = ax.barh(y, values, color=colors)
    for bar, row in zip(bars, top, strict=True):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            _wrap(row["problem_name"].replace(".py", ""), 40),
            va="center",
            fontsize=8.5,
        )

    ax.set_title("Top Interactive Wins vs PyTorch Eager")
    ax.set_xlabel("Speedup vs torch")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    _finalize_axes(ax, "x")
    ax.text(0.01, 0.98, "Blue = PTX, green = CUDA", transform=ax.transAxes, fontsize=9, color=SUBTLE)
    fig.tight_layout()

    output_path = output_dir / "top_wins.png"
    _save(fig, output_path)
    plt.close(fig)
    return output_path


def render_hybrid_level3(input_dir: Path, output_dir: Path) -> Path:
    rows = _read_csv(input_dir / "hybrid_level3.csv")
    labels = [f"{row['problem_id']}: {row['problem_name'].replace('.py', '')}" for row in rows]
    runtime = np.array([float(row["runtime_ms"]) for row in rows], dtype=float)
    ref_runtime = np.array([float(row["ref_runtime_ms"]) for row in rows], dtype=float)
    speedup = np.array([float(row["speedup_vs_torch"]) for row in rows], dtype=float)

    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.bar(x - width / 2, ref_runtime, width, label="Torch Eager", color=BASE)
    ax.bar(x + width / 2, runtime, width, label="PTX-Hybrid", color=HYBRID)

    for xpos, value in zip(x, speedup, strict=True):
        ax.text(xpos, max(ref_runtime.max(), runtime.max()) + 0.05, f"{value:.3f}x", ha="center", fontsize=10, color=TEXT)

    ax.set_title("Hybrid PTX Level 3 Runtime")
    ax.text(
        0.0,
        1.03,
        "Torch handles GEMMs; PTX handles hidden-layer ReLU",
        transform=ax.transAxes,
        fontsize=10,
        color=SUBTLE,
    )
    ax.set_ylabel("Runtime (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    _finalize_axes(ax, "y")
    fig.tight_layout()

    output_path = output_dir / "hybrid_level3.png"
    _save(fig, output_path)
    plt.close(fig)
    return output_path


def render_ptx_vs_cuda_control(input_dir: Path, output_dir: Path) -> Path:
    rows = _read_csv(input_dir / "ptx_vs_cuda_control.csv")
    labels = [row["label"] for row in rows]
    ptx_correct = np.array([float(row["ptx_correct_rate"]) for row in rows], dtype=float)
    cuda_correct = np.array([float(row["cuda_correct_rate"]) for row in rows], dtype=float)
    ptx_win = np.array([float(row["ptx_win_rate"]) for row in rows], dtype=float)
    ratio = np.array([float(row["ptx_over_cuda_geomean"]) for row in rows], dtype=float)
    jointly_correct = np.array([int(float(row["jointly_correct"])) for row in rows], dtype=int)

    x = np.arange(len(labels), dtype=float)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.4),
        gridspec_kw={"height_ratios": [1.15, 1.0]},
        sharex=True,
    )

    bars1 = ax1.bar(x - width / 2, ptx_correct, width, color=PTX, label="PTX correctness")
    bars2 = ax1.bar(x + width / 2, cuda_correct, width, color=CUDA, label="CUDA correctness")
    _annotate_bars(ax1, bars1)
    _annotate_bars(ax1, bars2)
    ax1.set_title("Does More Control Help? PTX vs CUDA On Shared Tasks")
    ax1.set_ylabel("Correctness Rate")
    ax1.set_ylim(0, 1.10)
    _finalize_axes(ax1, "y")
    ax1.legend(frameon=False, ncol=2, loc="upper right")

    bars3 = ax2.bar(x, ptx_win, width=0.48, color=ACCENT, label="PTX head-to-head win rate")
    _annotate_bars(ax2, bars3)
    for xpos, geo, joint in zip(x, ratio, jointly_correct, strict=True):
        note = f"PTX/CUDA runtime = {geo:.2f}x\njoint={joint}"
        ax2.text(
            xpos,
            min(1.04, ptx_win.max() + 0.14),
            note,
            ha="center",
            va="bottom",
            fontsize=9,
            color=SUBTLE,
        )
    ax2.set_ylabel("PTX Win Rate")
    ax2.set_xlabel("Shared Interactive Slice")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.10)
    _finalize_axes(ax2, "y")
    fig.tight_layout()

    output_path = output_dir / "ptx_vs_cuda_control.png"
    _save(fig, output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PNG charts from the interactive CSV plot bundle.")
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "results" / "analysis" / "codex-interactive_figures"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "analysis" / "codex-interactive_figures"),
    )
    args = parser.parse_args()
    _setup_style()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        render_coverage_summary(input_dir, output_dir),
        render_coverage_summary_ptx_only(input_dir, output_dir),
        render_overlap_comparison(input_dir, output_dir),
        render_failure_breakdown(input_dir, output_dir),
        render_family_summary(input_dir, output_dir),
        render_family_summary_ptx_l1_selected(input_dir, output_dir),
        render_top_wins(input_dir, output_dir),
        render_hybrid_level3(input_dir, output_dir),
        render_ptx_vs_cuda_control(input_dir, output_dir),
    ]
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
