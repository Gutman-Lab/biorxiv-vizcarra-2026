"""Plotting helpers for linear-probe and CNN analysis reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks.utils.io import maybe_save_fig, safe_filename
from notebooks.utils.labels import COMPARISON_GROUPS, model_label
from notebooks.utils.tables import (
    _confusion_block,
    _metrics_block,
    get_result,
    results_to_frame,
    sorted_results,
)

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
except ImportError:
    sns = None


def plot_result_metrics(
    results: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = ("macro_f1", "balanced_accuracy"),
    split: Literal["hold-out", "validation"] = "hold-out",
    class_names: Sequence[str] = (),
    sort_by: str | None = "macro_f1",
    label_col: str = "model_label",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    title_color: str = "black",
    legend: bool = True,
    save_name: str | None = None,
    save_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    df = results_to_frame(results, split=split, class_names=class_names)
    if df.empty:
        raise ValueError("No non-skipped results to plot.")

    plot_df = df.copy()
    if sort_by is not None:
        plot_df = plot_df.sort_values(sort_by, ascending=True)

    n_models = len(plot_df)
    n_metrics = len(metrics)
    if figsize is None:
        figsize = (11, max(4, 0.45 * n_models + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n_models)
    bar_height = 0.8 / n_metrics
    colors = plt.cm.tab10(np.linspace(0, 0.7, n_metrics))
    is_cnn = (
        plot_df["method"].eq("CNN fine-tune")
        | plot_df["model_family"].eq("cnn_finetune")
        | plot_df["model_id"].astype(str).str.startswith("cnn/")
    ).to_numpy()

    # Matplotlib 3.11+ hatch is independently colored; default matches facecolor
    # and is invisible unless hatchcolor is set explicitly.
    with plt.rc_context({"hatch.linewidth": 1.8}):
        for i, metric in enumerate(metrics):
            offset = (i - (n_metrics - 1) / 2) * bar_height
            values = plot_df[metric].to_numpy(dtype=float)
            for j, (val, cnn) in enumerate(zip(values, is_cnn)):
                bar_kwargs: dict[str, Any] = {
                    "height": bar_height,
                    "color": colors[i],
                    "edgecolor": "0.15" if cnn else "white",
                    "linewidth": 0.8 if cnn else 0.5,
                }
                if cnn:
                    bar_kwargs["hatch"] = "///"
                    bar_kwargs["hatchcolor"] = "0.1"
                bars = ax.barh(
                    y[j] + offset,
                    val,
                    label=metric.replace("_", " ") if j == 0 else None,
                    **bar_kwargs,
                )
                if np.isfinite(val):
                    bar = bars[0]
                    ax.text(
                        val + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}",
                        va="center",
                        ha="left",
                        fontsize=8,
                    )

    labels = [f"{row[label_col]} ({row['method'].split()[0]})" for _, row in plot_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title(title or f"Model comparison - {split}", color=title_color)
    if legend:
        ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save_name:
        maybe_save_fig(fig, save_name, save_dir=save_dir)
    return fig, ax, plot_df


def plot_probe_metrics(*args, **kwargs):
    return plot_result_metrics(*args, **kwargs)


def plot_per_class_f1(
    results: list[dict[str, Any]],
    *,
    split: Literal["hold-out", "validation"] = "hold-out",
    class_names: Sequence[str],
    models: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    title_color: str = "black",
    save_name: str | None = None,
    save_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    df = results_to_frame(results, split=split, class_names=class_names)
    if models is not None:
        df = df[df["model_id"].isin(models)]
    df = df.sort_values("macro_f1", ascending=True)

    melted = df.melt(
        id_vars=["model_id", "model_label", "method", "model_family"],
        value_vars=[f"{cls}_f1" for cls in class_names],
        var_name="class_name",
        value_name="f1",
    )
    melted["class_name"] = melted["class_name"].str.replace("_f1", "", regex=False)
    melted["label"] = melted["model_label"] + " (" + melted["method"].str.split().str[0] + ")"

    if figsize is None:
        figsize = (12, max(4, 0.42 * len(df) + 2))

    fig, ax = plt.subplots(figsize=figsize)
    if sns is not None:
        sns.barplot(
            data=melted,
            y="label",
            x="f1",
            hue="class_name",
            hue_order=list(class_names),
            ax=ax,
            orient="h",
        )
    else:
        for cls in class_names:
            subset = melted[melted["class_name"] == cls]
            ax.barh(subset["label"], subset["f1"], label=cls, alpha=0.8)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1")
    ax.set_ylabel("")
    ax.set_title(title or f"Per-class F1 - {split}", color=title_color)
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    if save_name:
        maybe_save_fig(fig, save_name, save_dir=save_dir)
    return fig, ax, melted


def plot_confusion_matrix(
    cm: dict[str, Any],
    *,
    title: str,
    class_names: Sequence[str] = (),
    normalize: Literal[False, "true", "pred", "all"] = False,
    ax: plt.Axes | None = None,
    cmap: str = "Blues",
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    matrix = np.asarray(cm.get("matrix", []), dtype=float)
    labels = list(cm.get("labels", class_names))
    is_normalized = normalize is not False
    created_ax = ax is None

    if matrix.size == 0:
        raise ValueError("Confusion matrix is empty")

    if is_normalized:
        axis = {"true": 1, "pred": 0, "all": None}[normalize]
        with np.errstate(invalid="ignore"):
            denom = matrix.sum(axis=axis, keepdims=True)
            plot_matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
        annot_fmt = ".2f"
        cbar_label = "proportion"
    else:
        plot_matrix = matrix.astype(int)
        annot_fmt = "d"
        cbar_label = "count"

    if created_ax:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure

    if sns is not None:
        sns.heatmap(
            plot_matrix,
            annot=True,
            fmt=annot_fmt,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": cbar_label},
        )
    else:
        im = ax.imshow(plot_matrix, cmap=cmap)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        for i in range(plot_matrix.shape[0]):
            for j in range(plot_matrix.shape[1]):
                value = plot_matrix[i, j]
                text = f"{value:.2f}" if is_normalized else f"{int(value)}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if created_ax:
        fig.tight_layout()
    return fig, ax, plot_matrix


def plot_result_confusion_matrix(
    results: list[dict[str, Any]],
    model_id: str,
    *,
    split: Literal["hold-out", "validation"] = "hold-out",
    class_names: Sequence[str] = (),
    normalize: Literal[False, "true", "pred", "all"] = False,
    save_name: str | None = None,
    save_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    result = get_result(results, model_id)
    cm = _confusion_block(result, split)
    macro_f1 = _metrics_block(result, split).get("macro_f1", float("nan"))
    title = f"{model_label(result)}\n{split} | macro-F1={macro_f1:.3f}"
    fig, ax, matrix = plot_confusion_matrix(cm, title=title, class_names=class_names, normalize=normalize)
    if save_name:
        maybe_save_fig(fig, save_name, save_dir=save_dir)
    return fig, ax, matrix


def plot_probe_confusion_matrix(probe_results: list[dict[str, Any]], model_id: str, **kwargs):
    return plot_result_confusion_matrix(probe_results, model_id, **kwargs)


def save_all_confusion_matrices(
    results: list[dict[str, Any]],
    *,
    split: Literal["hold-out", "validation"] = "hold-out",
    normalize: Literal[False, "true"] = False,
    save_dir: Path | None = None,
    class_names: Sequence[str] = (),
) -> list[Path | None]:
    if save_dir is None:
        raise ValueError("Pass save_dir before calling save_all_confusion_matrices()")

    split_slug = split.replace("/", "-")
    norm_slug = "normalized" if normalize is not False else "counts"
    saved: list[Path | None] = []

    for result in sorted_results(results, split=split):
        model_id = result["model_id"]
        filename = f"confusion_matrices/{split_slug}/{norm_slug}/{safe_filename(model_id)}.png"
        fig, _, _ = plot_result_confusion_matrix(
            results,
            model_id,
            split=split,
            normalize=normalize,
            class_names=class_names,
        )
        saved.append(maybe_save_fig(fig, filename, save_dir=save_dir))
        plt.close(fig)

    print(f"Saved {len(saved)} confusion matrix figure(s) to {save_dir}")
    return saved


def plot_validation_vs_holdout(
    results: list[dict[str, Any]],
    *,
    metric: str = "macro_f1",
    class_names: Sequence[str] = (),
    save_name: str | None = None,
    save_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    val_df = results_to_frame(results, split="validation", class_names=class_names)
    hold_df = results_to_frame(results, split="hold-out", class_names=class_names)
    merged = val_df[["model_id", "model_label", "method", "model_family", metric]].merge(
        hold_df[["model_id", metric]],
        on="model_id",
        suffixes=("_val", "_holdout"),
    )

    fig, ax = plt.subplots(figsize=(6.8, 6))
    families = merged["model_family"].unique()
    palette = plt.cm.tab10(np.linspace(0, 1, len(families)))
    for color, family in zip(palette, families):
        subset = merged[merged["model_family"] == family]
        ax.scatter(
            subset[f"{metric}_val"],
            subset[f"{metric}_holdout"],
            label=family,
            s=80,
            alpha=0.85,
            color=color,
        )
        for _, row in subset.iterrows():
            ax.annotate(row["model_label"], (row[f"{metric}_val"], row[f"{metric}_holdout"]), fontsize=7, alpha=0.85)

    lims = [0, 1.02]
    ax.plot(lims, lims, "k--", alpha=0.35, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Validation {metric}")
    ax.set_ylabel(f"Hold-out {metric}")
    ax.set_title("Validation vs hold-out")
    ax.legend(title="Family", loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_name:
        maybe_save_fig(fig, save_name, save_dir=save_dir)
    return fig, ax, merged


def plot_compute_time_vs_performance(
    leaderboard: pd.DataFrame,
    *,
    x_col: str = "available_compute_seconds",
    y_col: str = "holdout_macro_f1",
    save_name: str | None = None,
    save_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    plot_df = leaderboard.dropna(subset=[x_col, y_col]).copy()
    plot_df = plot_df[plot_df[x_col] > 0]

    fig, ax = plt.subplots(figsize=(8, 5.8))
    for family, subset in plot_df.groupby("model_family"):
        ax.scatter(subset[x_col], subset[y_col], label=family, s=90, alpha=0.85)
        for _, row in subset.iterrows():
            ax.annotate(row["model_label"], (row[x_col], row[y_col]), fontsize=7, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("Available compute time (seconds, log scale)")
    ax.set_ylabel("Hold-out macro F1")
    ax.set_title("Compute time vs performance")
    ax.legend(title="Family", loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    if save_name:
        maybe_save_fig(fig, save_name, save_dir=save_dir)
    return fig, ax, plot_df


def plot_model_dataset_heatmap(
    leaderboard: pd.DataFrame,
    *,
    value_col: str = "holdout_macro_f1",
    save_name: str | None = None,
    save_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    plot_df = leaderboard.dropna(subset=[value_col]).copy()
    if plot_df.empty:
        raise ValueError("No cross-dataset rows to plot")
    plot_df["model_display"] = np.where(
        plot_df["method"].eq("CNN fine-tune"),
        plot_df["model_label"],
        plot_df["model_id"],
    )
    pivot = plot_df.pivot_table(
        index="model_display",
        columns="dataset_display",
        values=value_col,
        aggfunc="max",
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * pivot.shape[1] + 3), max(4, 0.38 * pivot.shape[0] + 1.5)))
    if sns is not None:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": value_col})
    else:
        im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]), pivot.index)
        fig.colorbar(im, ax=ax, label=value_col)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Model")
    ax.set_title(value_col.replace("_", " ").title())
    fig.tight_layout()
    if save_name:
        maybe_save_fig(fig, save_name, save_dir=save_dir)
    return fig, ax, pivot


def plot_best_method_by_dataset(
    summary: pd.DataFrame,
    *,
    save_name: str | None = None,
    save_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    plot_df = summary.dropna(subset=["holdout_macro_f1"]).copy()
    if plot_df.empty:
        raise ValueError("No cross-dataset summary rows to plot")
    order = [label for label, _ in COMPARISON_GROUPS if label in set(plot_df["comparison_group"])]
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * plot_df["dataset_display"].nunique() + 4), 5.5))
    if sns is not None:
        sns.barplot(
            data=plot_df,
            x="dataset_display",
            y="holdout_macro_f1",
            hue="comparison_group",
            hue_order=order,
            ax=ax,
        )
    else:
        for group, subset in plot_df.groupby("comparison_group"):
            ax.bar(subset["dataset_display"], subset["holdout_macro_f1"], label=group, alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Hold-out macro F1")
    ax.set_title("Best result by method family")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Comparison", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if save_name:
        maybe_save_fig(fig, save_name, save_dir=save_dir)
    return fig, ax, plot_df


def plot_cnn_embedding_delta(
    delta_df: pd.DataFrame,
    *,
    save_name: str | None = None,
    save_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    if "cnn_minus_best_embedding_macro_f1" not in delta_df:
        raise ValueError("Delta table does not include CNN and best embedding columns")
    plot_df = delta_df.dropna(subset=["cnn_minus_best_embedding_macro_f1"]).copy()
    if plot_df.empty:
        raise ValueError("No CNN-vs-embedding deltas to plot")
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(plot_df) + 2), 4.8))
    colors = np.where(plot_df["cnn_minus_best_embedding_macro_f1"] >= 0, "#2ca02c", "#d62728")
    ax.bar(plot_df["dataset_display"], plot_df["cnn_minus_best_embedding_macro_f1"], color=colors, alpha=0.85)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_ylabel("Macro-F1 delta")
    ax.set_title("CNN fine-tune minus best embedding probe")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    if save_name:
        maybe_save_fig(fig, save_name, save_dir=save_dir)
    return fig, ax, plot_df

