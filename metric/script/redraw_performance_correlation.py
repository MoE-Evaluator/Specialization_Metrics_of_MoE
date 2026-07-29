#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CANDIDATES = [
    ROOT / "最终数据表.xlsx",
    ROOT.parent / "最终数据表.xlsx",
]
DEFAULT_OUT_DIR = ROOT / "correlation"

# Performance values are taken from the paper table and aligned with the
# currently reported comparable MoE models/tasks.
PERFORMANCE_ROWS = [
    ("Qwen3-235B-Thinking", "Math", 81.5),
    ("GLM-4.6", "Math", 93.9),
    ("DeepSeek-R1-0528", "Math", 70.0),
    ("Qwen3-235B-Thinking", "Knowledge", 18.2),
    ("GLM-4.6", "Knowledge", 17.2),
    ("DeepSeek-R1-0528", "Knowledge", 17.7),
    ("Qwen3-235B-Thinking", "Code", 74.1),
    ("GLM-4.6", "Code", 82.8),
    ("DeepSeek-R1-0528", "Code", 68.7),
    ("GLM-4.6", "Code2", 68.0),
    ("DeepSeek-R1-0528", "Code2", 57.6),
]

METRICS = [
    "Routing_Specialization",
    "Normalized_Effective_Rank",
    "Domain_Isolation",
    "ngram_2_ratio_mean",
    "ngram_5_ratio_mean",
    "ngram_10_ratio_mean",
    "ngram_20_ratio_mean",
    "rademacher_1000_mean",
    "group_ngram_n2_ratio_mean",
    "group_ngram_n5_ratio_mean",
    "group_ngram_n10_ratio_mean",
    "group_ngram_n20_ratio_mean",
]

DISPLAY_NAMES = {
    "Routing_Specialization": "Routing Specialization",
    "Normalized_Effective_Rank": "Normalized Effective Rank",
    "Domain_Isolation": "Domain Isolation",
    "ngram_2_ratio_mean": "N-gram (n=2)",
    "ngram_5_ratio_mean": "N-gram (n=5)",
    "ngram_10_ratio_mean": "N-gram (n=10)",
    "ngram_20_ratio_mean": "N-gram (n=20)",
    "rademacher_1000_mean": r"RSS ($\sigma=10^{-4}$)",
    "group_ngram_n2_ratio_mean": "Group N-gram (n=2)",
    "group_ngram_n5_ratio_mean": "Group N-gram (n=5)",
    "group_ngram_n10_ratio_mean": "Group N-gram (n=10)",
    "group_ngram_n20_ratio_mean": "Group N-gram (n=20)",
}

MODEL_COLORS = {
    "Qwen3-235B-Thinking": "#2563eb",
    "GLM-4.6": "#dc2626",
    "DeepSeek-R1-0528": "#16a34a",
}

DOMAIN_MARKERS = {
    "Math": "o",
    "Knowledge": "s",
    "Code": "^",
    "Code2": "D",
}


def resolve_data_file(cli_path: str | None) -> Path:
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path

    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.is_file():
            return candidate

    searched = "\n".join(str(path) for path in DEFAULT_DATA_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate the metrics workbook. Checked:\n"
        f"{searched}\nUse --data-file to specify it explicitly."
    )


def load_merged_frame(data_file: Path) -> pd.DataFrame:
    metrics_df = pd.read_excel(data_file, sheet_name="Sheet2")
    metrics_df["模型"] = metrics_df["模型"].ffill()
    metrics_df["领域"] = (
        metrics_df["领域"].astype(str).str.replace("\n", "", regex=False).str.strip()
    )

    perf_df = pd.DataFrame(PERFORMANCE_ROWS, columns=["模型", "领域", "performance"])
    merged = metrics_df.merge(perf_df, on=["模型", "领域"], how="inner")
    merged = merged[["模型", "领域", "performance", *METRICS]].copy()
    return merged


def add_trend_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> float:
    r = float(np.corrcoef(x, y)[0, 1])
    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            xs,
            slope * xs + intercept,
            color="#ef4444",
            linestyle="--",
            linewidth=1.4,
        )
    return r


def scatter_grid(merged: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    axes = axes.flatten()

    for ax in axes[len(METRICS) :]:
        ax.axis("off")

    for ax, metric in zip(axes, METRICS):
        x = merged[metric].to_numpy(dtype=float)
        y = merged["performance"].to_numpy(dtype=float)
        r = add_trend_line(ax, x, y)

        for _, row in merged.iterrows():
            ax.scatter(
                row[metric],
                row["performance"],
                s=70,
                color=MODEL_COLORS[row["模型"]],
                marker=DOMAIN_MARKERS[row["领域"]],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
            )

        ax.set_title(f"{DISPLAY_NAMES[metric]}\n(r={r:.3f})", fontsize=11)
        ax.set_xlabel(DISPLAY_NAMES[metric], fontsize=9)
        ax.set_ylabel("Performance (%)", fontsize=9)
        ax.grid(alpha=0.25)

    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=9,
            label=model,
        )
        for model, color in MODEL_COLORS.items()
    ]
    domain_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#374151",
            linestyle="None",
            markersize=8,
            label=domain,
        )
        for domain, marker in DOMAIN_MARKERS.items()
    ]
    fig.legend(
        handles=model_handles + domain_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle("All Metrics vs Performance", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out_dir / "all_metrics_scatter-1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def correlation_bar(merged: pd.DataFrame, out_dir: Path) -> None:
    correlations = []
    for metric in METRICS:
        r = float(
            np.corrcoef(
                merged[metric].to_numpy(dtype=float),
                merged["performance"].to_numpy(dtype=float),
            )[0, 1]
        )
        correlations.append((metric, r))

    corr_df = pd.DataFrame(correlations, columns=["metric", "r"]).sort_values("r")
    colors = ["#ef4444" if r < 0 else "#16a34a" for r in corr_df["r"]]

    fig, ax = plt.subplots(figsize=(15, 9))
    bars = ax.barh(corr_df["metric"], corr_df["r"], color=colors, alpha=0.9)
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_xlabel("Pearson Correlation with Performance", fontsize=12)
    ax.set_title("Performance Correlation by Metric", fontsize=22, pad=14)
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, corr_df["r"]):
        x = bar.get_width()
        ha = "left" if value >= 0 else "right"
        offset = 0.01 if value >= 0 else -0.01
        ax.text(
            x + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=ha,
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "individual_correlations_bar-1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def rss_focus_scatter(merged: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
    metric = "rademacher_1000_mean"
    x = merged[metric].to_numpy(dtype=float)
    y = merged["performance"].to_numpy(dtype=float)
    r = add_trend_line(ax, x, y)

    for _, row in merged.iterrows():
        ax.scatter(
            row[metric],
            row["performance"],
            s=78,
            color=MODEL_COLORS[row["模型"]],
            marker=DOMAIN_MARKERS[row["领域"]],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
        )

    ax.set_title(f"{DISPLAY_NAMES[metric]}\n(r={r:.3f})", fontsize=12)
    ax.set_xlabel(DISPLAY_NAMES[metric], fontsize=10)
    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.grid(alpha=0.25)
    fig.suptitle("Performance vs RSS", fontsize=20, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_rss_scatter-1.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw metric-performance correlation figures, including RSS."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to the workbook containing Sheet2 metric values.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for generated correlation figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = resolve_data_file(args.data_file)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merged_frame(data_file)
    scatter_grid(merged, out_dir)
    correlation_bar(merged, out_dir)
    rss_focus_scatter(merged, out_dir)
    print(f"Performance correlation figures regenerated successfully in: {out_dir}")


if __name__ == "__main__":
    main()
