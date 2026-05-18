"""
Diagnostic utilities for inspecting raw attribution values across modalities.

Usage:
    from explanation_diagnostics import plot_modality_diagnostics

    # summary is a DocumentInstanceExplanationSummary
    plot_modality_diagnostics(summary)

    # or with pre/post reduction comparison
    plot_modality_diagnostics(summary, show_pre_reduction=True)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from docxeval.analysis.explanation_analysis.utils.colors import (
    green_transparent_purple,
    red_transparent_blue,
)


def plot_modality_diagnostics(
    summary,  # DocumentInstanceExplanationSummary
    show_pre_reduction: bool = True,
    figsize_per_row: tuple[float, float] = (18, 4),
) -> tuple[plt.Figure, plt.Figure, list[dict]]:
    """
    Plot raw attribution distributions and sums for each modality.

    Returns:
        fig_detail: per-modality histograms, box plots, stats
        fig_summary: cross-modality bar charts
        all_stats: list of stat dicts (one per modality × stage),
                   ready for CSV / DataFrame
    """

    feature_keys = summary.explanation_state.feature_keys
    raw_explanations = summary.explanation_state.explanations.value

    # ---------- collect tensors ----------
    stages: list[tuple[str, dict[str, torch.Tensor]]] = []

    if show_pre_reduction:
        pre = {
            key: exp for key, exp in zip(feature_keys, raw_explanations, strict=True)
        }
        stages.append(("pre-reduction", pre))

    reduced = summary.reduced_explanations()
    post = {key: exp for key, exp in zip(feature_keys, reduced, strict=True)}
    stages.append(("post-reduction", post))

    # ---------- figure layout ----------
    n_modalities = len(feature_keys)
    n_stages = len(stages)
    nrows = n_modalities * n_stages
    ncols = 3

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_row[0], figsize_per_row[1] * nrows),
        squeeze=False,
    )

    row = 0
    all_stats = []

    for stage_name, tensors in stages:
        for key in feature_keys:
            t = tensors[key]
            flat = t.detach().cpu().float().numpy().ravel()

            stats = _compute_stats(flat)
            stats["modality"] = key
            stats["stage"] = stage_name
            stats["shape"] = str(list(t.shape))
            all_stats.append(stats)

            color = _modality_color(key)

            # --- histogram ---
            ax_hist = axes[row, 0]
            ax_hist.hist(
                flat,
                bins=80,
                color=color,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.3,
            )
            ax_hist.set_title(f"{key}  [{stage_name}]  — histogram", fontsize=11)
            ax_hist.set_xlabel("attribution value")
            ax_hist.set_ylabel("count")
            ax_hist.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

            # --- box plot ---
            ax_box = axes[row, 1]
            ax_box.boxplot(
                flat,
                vert=False,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.6),
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
            )
            ax_box.set_title(f"{key}  [{stage_name}]  — box plot", fontsize=11)
            ax_box.set_xlabel("attribution value")
            ax_box.set_yticklabels([])

            # --- stats text ---
            ax_txt = axes[row, 2]
            ax_txt.axis("off")
            text_block = (
                f"shape:    {list(t.shape)}\n"
                f"numel:    {flat.size}\n"
                f"─────────────────────\n"
                f"sum:      {stats['sum']:+.6f}\n"
                f"abs_sum:  {stats['abs_sum']:.6f}\n"
                f"mean:     {stats['mean']:+.6f}\n"
                f"std:      {stats['std']:.6f}\n"
                f"min:      {stats['min']:+.6f}\n"
                f"max:      {stats['max']:+.6f}\n"
                f"─────────────────────\n"
                f"% positive: {stats['pct_pos']:.1f}%\n"
                f"% negative: {stats['pct_neg']:.1f}%\n"
                f"% zero:     {stats['pct_zero']:.1f}%\n"
            )
            ax_txt.text(
                0.05,
                0.95,
                text_block,
                transform=ax_txt.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.15),
            )

            row += 1

    fig_summary = _plot_grand_summary(all_stats)
    plt.tight_layout()
    return fig, fig_summary, all_stats


def _plot_grand_summary(
    all_stats: list[dict],
) -> plt.Figure:
    """
    Bar chart comparing total sum and abs_sum across modalities and stages.
    This is the key chart: if global normalization is valid, you want to
    understand how the raw sums compare before you flatten everything.
    """
    stages = sorted(set(s["stage"] for s in all_stats))
    modalities = list(dict.fromkeys(s["modality"] for s in all_stats))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- bar chart: raw sum per modality ---
    _grouped_bar(
        axes[0],
        all_stats,
        metric="sum",
        title="Raw Sum  (signed)",
        modalities=modalities,
        stages=stages,
    )

    # --- bar chart: abs sum per modality ---
    _grouped_bar(
        axes[1],
        all_stats,
        metric="abs_sum",
        title="Absolute Sum",
        modalities=modalities,
        stages=stages,
    )

    # --- bar chart: fraction of total abs_sum per modality (post-reduction only) ---
    ax_frac = axes[2]
    post_stats = [s for s in all_stats if s["stage"] == "post-reduction"]
    total_abs = sum(s["abs_sum"] for s in post_stats)
    if total_abs > 0:
        names = [s["modality"] for s in post_stats]
        fracs = [s["abs_sum"] / total_abs * 100 for s in post_stats]
        colors = [_modality_color(n) for n in names]
        bars = ax_frac.bar(names, fracs, color=colors, alpha=0.8, edgecolor="white")
        for bar, frac in zip(bars, fracs):
            ax_frac.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{frac:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax_frac.set_title("Modality Share of Total |attr|  (post-reduction)", fontsize=11)
    ax_frac.set_ylabel("% of total absolute attribution")
    ax_frac.set_ylim(0, 110)

    # grand total annotation
    fig.suptitle(
        f"Grand total signed sum (post-reduction): {sum(s['sum'] for s in post_stats):+.6f}    |    "
        f"Grand total |sum| (post-reduction): {total_abs:.6f}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    return fig


# ─────────────────────────── helpers ───────────────────────────


def _compute_stats(flat: np.ndarray) -> dict:
    return {
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "pct_pos": float((flat > 0).sum() / flat.size * 100),
        "pct_neg": float((flat < 0).sum() / flat.size * 100),
        "pct_zero": float((flat == 0).sum() / flat.size * 100),
    }


_COLOR_MAP = {
    "token_embeddings": "#4C78A8",
    "position_embeddings": "#F58518",
    "layout_embeddings": "#54A24B",
    "image": "#E45756",
}


def _modality_color(key: str) -> str:
    return _COLOR_MAP.get(key, "#72B7B2")


def _grouped_bar(
    ax: plt.Axes,
    all_stats: list[dict],
    metric: str,
    title: str,
    modalities: list[str],
    stages: list[str],
):
    x = np.arange(len(modalities))
    width = 0.35
    n_stages = len(stages)
    offsets = np.linspace(-width / 2, width / 2, n_stages)

    for i, stage in enumerate(stages):
        vals = []
        for mod in modalities:
            match = [
                s for s in all_stats if s["modality"] == mod and s["stage"] == stage
            ]
            vals.append(match[0][metric] if match else 0)
        colors = [_modality_color(m) for m in modalities]
        alpha = 0.5 if stage == "pre-reduction" else 0.9
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width * 0.9,
            color=colors,
            alpha=alpha,
            edgecolor="white",
            label=stage,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.5 if val >= 0 else -0.5),
                f"{val:.4f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)


def score_to_color_map(explanation_score: float, color_map="red_transparent_blue"):
    if color_map == "red_transparent_blue":
        rgba = red_transparent_blue(explanation_score)
    elif color_map == "green_transparent_purple":
        rgba = green_transparent_purple(explanation_score)
    return rgba
