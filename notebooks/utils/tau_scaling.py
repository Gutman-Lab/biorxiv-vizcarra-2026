"""Tau data-scaling loaders and plots."""
from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks.utils.io import load_cnn_results, load_linear_probe_results, load_run_args
from notebooks.utils.paths import cnn_run_dir, linear_probe_run_dir
from notebooks.utils.tables import results_to_frame

TAU_DATA_PERCENTAGES = (1, 5, 10, 25, 50, 100)
TAU_FULL_RUN_NAME = "tau-dataset-single-label"


def tau_scaling_run_name(percentage: int) -> str:
    return TAU_FULL_RUN_NAME if percentage == 100 else f"tau-data-scaling-{percentage}pct"


def load_tau_scaling_results(
    *,
    class_names: Sequence[str],
    percentages: Sequence[int] = TAU_DATA_PERCENTAGES,
) -> tuple[dict[int, list[dict[str, Any]]], pd.DataFrame, pd.DataFrame]:
    results_by_percentage: dict[int, list[dict[str, Any]]] = {}
    availability_rows: list[dict[str, Any]] = []

    for percentage in percentages:
        run_name = tau_scaling_run_name(percentage)
        linear_dir = linear_probe_run_dir(run_name)
        cnn_dir = cnn_run_dir(run_name)
        linear_args = load_run_args(linear_dir) if (linear_dir / "args.json").exists() else None
        cnn_args = load_run_args(cnn_dir) if (cnn_dir / "args.json").exists() else None

        metadata = {
            "run_key": "tau_scaling",
            "run_name": run_name,
            "dataset_display": "Tau inclusions",
            "modality": "IHC",
            "stain": "Tau IHC",
            "task_family": "neuropathology",
            "resolved_dataset": "tau-dataset",
        }
        linear_results = load_linear_probe_results(linear_dir, metadata=metadata)
        cnn_results_pct = load_cnn_results(cnn_dir, metadata=metadata, args=cnn_args)
        percentage_results = linear_results + cnn_results_pct
        for result in percentage_results:
            result["train_percentage"] = percentage
            result["scaling_run_name"] = run_name

        results_by_percentage[percentage] = percentage_results
        availability_rows.append(
            {
                "train_percentage": percentage,
                "run_name": run_name,
                "linear_models_complete": len(linear_results),
                "cnn_models_complete": len(cnn_results_pct),
                "linear_args_present": linear_args is not None,
                "cnn_args_present": cnn_args is not None,
            }
        )

    full_results = results_by_percentage.get(100, [])
    expected_linear = sum(r.get("analysis_source") == "linear_probe" for r in full_results)
    expected_cnn = sum(r.get("analysis_source") == "cnn" for r in full_results)
    availability = pd.DataFrame(availability_rows)
    availability["expected_linear_models"] = expected_linear
    availability["expected_cnn_models"] = expected_cnn

    full_train_count = next(
        (
            r.get("training_subset", {}).get("full_train_count")
            for percentage_results in results_by_percentage.values()
            for r in percentage_results
            if r.get("training_subset", {}).get("full_train_count") is not None
        ),
        None,
    )

    frames: list[pd.DataFrame] = []
    for percentage, percentage_results in results_by_percentage.items():
        frame = results_to_frame(percentage_results, split="hold-out", class_names=class_names)
        if frame.empty:
            continue
        frame["train_percentage"] = percentage
        frame["scaling_run_name"] = tau_scaling_run_name(percentage)
        selected_counts = {
            str(r["model_id"]): r.get("training_subset", {}).get("selected_train_count")
            for r in percentage_results
        }
        frame["selected_train_count"] = frame["model_id"].map(selected_counts)
        if percentage == 100 and full_train_count is not None:
            frame["selected_train_count"] = frame["selected_train_count"].fillna(full_train_count)
        frames.append(frame)

    scaling_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return results_by_percentage, availability, scaling_frame


def plot_tau_learning_curves(
    frame: pd.DataFrame,
    *,
    metric: str = "macro_f1",
    model_ids: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 7),
    percentages: Sequence[int] = TAU_DATA_PERCENTAGES,
):
    if frame.empty:
        raise ValueError("No completed Tau scaling results are available yet.")
    plot_df = frame.dropna(subset=[metric]).copy()
    if model_ids is not None:
        plot_df = plot_df[plot_df["model_id"].isin(model_ids)]
    if plot_df.empty:
        raise ValueError(f"No Tau scaling results available for {metric!r}.")

    x_positions = {percentage: i for i, percentage in enumerate(percentages)}
    fig, ax = plt.subplots(figsize=figsize)
    for model_id, group in plot_df.groupby("model_id"):
        group = group.sort_values("train_percentage")
        is_cnn = model_id.startswith("cnn/")
        ax.plot(
            group["train_percentage"].map(x_positions),
            group[metric],
            marker="o",
            linewidth=3 if is_cnn else 1.6,
            markersize=7 if is_cnn else 5,
            alpha=1.0 if is_cnn else 0.78,
            label=group["model_label"].iloc[0],
        )

    ax.set_xticks(range(len(percentages)), [f"{p}%" for p in percentages])
    ax.set_xlabel("Percentage of Tau training images")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Tau data scaling: hold-out {metric.replace('_', ' ')}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    return fig, ax, plot_df


def plot_tau_per_class_f1(
    frame: pd.DataFrame,
    model_ids: list[str],
    *,
    class_names: Sequence[str],
    percentages: Sequence[int] = TAU_DATA_PERCENTAGES,
):
    class_columns = [f"{class_name}_f1" for class_name in class_names]
    available_columns = [column for column in class_columns if column in frame.columns]
    plot_df = frame[frame["model_id"].isin(model_ids)].copy()
    long_df = plot_df.melt(
        id_vars=["train_percentage", "model_id", "model_label", "method"],
        value_vars=available_columns,
        var_name="class_metric",
        value_name="f1",
    ).dropna(subset=["f1"])
    long_df["class_name"] = long_df["class_metric"].str.removesuffix("_f1")
    if long_df.empty:
        raise ValueError("No per-class Tau scaling metrics are available yet.")

    x_positions = {percentage: i for i, percentage in enumerate(percentages)}
    fig, axes = plt.subplots(1, len(class_names), figsize=(17, 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, class_name in zip(axes, class_names):
        class_df = long_df[long_df["class_name"] == class_name]
        for model_id, group in class_df.groupby("model_id"):
            group = group.sort_values("train_percentage")
            ax.plot(
                group["train_percentage"].map(x_positions),
                group["f1"],
                marker="o",
                linewidth=2.5 if model_id.startswith("cnn/") else 1.5,
                label=group["model_label"].iloc[0],
            )
        ax.set_xticks(range(len(percentages)), [f"{p}%" for p in percentages], rotation=45)
        ax.set_title(class_name)
        ax.set_xlabel("Training data")
    axes[0].set_ylabel("Hold-out F1")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.suptitle("Tau class-specific learning curves")
    fig.tight_layout()
    return fig, axes, long_df


def tau_cnn_probe_gap(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for percentage, group in frame.groupby("train_percentage"):
        probes = group[group["method"] == "Frozen embedding + linear probe"].dropna(subset=["macro_f1"])
        cnns = group[group["method"] == "CNN fine-tune"].dropna(subset=["macro_f1"])
        if probes.empty and cnns.empty:
            continue
        best_probe = probes.nlargest(1, "macro_f1")
        best_cnn = cnns.nlargest(1, "macro_f1")
        probe_score = best_probe["macro_f1"].iloc[0] if not best_probe.empty else np.nan
        cnn_score = best_cnn["macro_f1"].iloc[0] if not best_cnn.empty else np.nan
        rows.append(
            {
                "train_percentage": percentage,
                "best_probe_model": best_probe["model_id"].iloc[0] if not best_probe.empty else None,
                "best_probe_macro_f1": probe_score,
                "cnn_macro_f1": cnn_score,
                "cnn_minus_best_probe": cnn_score - probe_score,
            }
        )
    return pd.DataFrame(rows).sort_values("train_percentage").reset_index(drop=True)


def plot_tau_cnn_probe_gap(
    gap_df: pd.DataFrame,
    *,
    percentages: Sequence[int] = TAU_DATA_PERCENTAGES,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    gap_complete = gap_df.dropna(subset=["best_probe_macro_f1", "cnn_macro_f1"])
    if gap_complete.empty:
        raise ValueError("No completed CNN vs probe gap rows to plot.")
    x_positions = {percentage: i for i, percentage in enumerate(percentages)}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        gap_complete["train_percentage"].map(x_positions),
        gap_complete["cnn_minus_best_probe"],
        marker="o",
        linewidth=2,
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(percentages)), [f"{p}%" for p in percentages])
    ax.set_xlabel("Percentage of Tau training images")
    ax.set_ylabel("CNN macro-F1 − best probe macro-F1")
    ax.set_title("Tau data scaling: CNN advantage over best embedding probe")
    fig.tight_layout()
    return fig, ax, gap_complete
