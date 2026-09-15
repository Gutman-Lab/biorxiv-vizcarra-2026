"""Flatten report JSONs into analysis tables and leaderboards."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from notebooks.utils.labels import (
    COMPARISON_GROUPS,
    ID_COLS,
    PER_LABEL_METRICS,
    SUMMARY_METRICS,
    method_label,
    model_family,
    model_label,
)
from notebooks.utils.paths import embedding_timing_md


def load_embedding_timings(md_path: Path | None = None) -> pd.DataFrame:
    """Parse ms/tile from reports/embedding_models.md summary table."""
    path = md_path if md_path is not None else embedding_timing_md()
    if not path.exists():
        return pd.DataFrame(columns=["model_id", "embed_dim", "ms_per_tile", "tiles_per_sec"])

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 6:
            continue
        model_id = parts[0].strip("`")
        try:
            ms_per_tile = float(parts[4])
        except ValueError:
            continue
        try:
            tiles_per_sec = float(parts[5])
        except ValueError:
            tiles_per_sec = np.nan
        rows.append(
            {
                "model_id": model_id,
                "embed_dim": parts[2],
                "ms_per_tile": ms_per_tile,
                "tiles_per_sec": tiles_per_sec,
            }
        )
    return pd.DataFrame(rows)


def _per_label_block(result: dict[str, Any], split: Literal["hold-out", "validation"]) -> dict[str, Any]:
    if split == "hold-out":
        return result.get("test_results", {}).get("per_label", {})
    return result.get("best_config", {}).get("validation_per_label", {})


def _metrics_block(result: dict[str, Any], split: Literal["hold-out", "validation"]) -> dict[str, Any]:
    if split == "hold-out":
        return result.get("test_results", {}).get("test_metrics", {})
    return result.get("best_config", {}).get("validation_metrics", {})


def _confusion_block(result: dict[str, Any], split: Literal["hold-out", "validation"]) -> dict[str, Any]:
    if split == "hold-out":
        return result.get("test_results", {}).get("confusion_matrix", {})
    return result.get("best_config", {}).get("validation_confusion_matrix", {})


def _split_sample_count(result: dict[str, Any], split: Literal["hold-out", "validation"]) -> int | None:
    if split == "hold-out":
        return result.get("test_results", {}).get("n_samples")
    support = result.get("best_config", {}).get("validation_label_support", {})
    return int(sum(support.values())) if support else None


def results_to_frame(
    results: list[dict[str, Any]],
    *,
    split: Literal["hold-out", "validation"] = "hold-out",
    class_names: Sequence[str] = (),
) -> pd.DataFrame:
    """Flatten report JSONs into one row per model/run."""
    rows: list[dict[str, Any]] = []

    for result in results:
        if result.get("skipped"):
            continue

        model_id = str(result["model_id"])
        metrics = _metrics_block(result, split)
        per_label = _per_label_block(result, split)
        best = result.get("best_config", {})
        final_training = result.get("final_training", {})
        timing = result.get("timing", {})

        row: dict[str, Any] = {
            "run_key": result.get("run_key"),
            "run_name": result.get("run_name"),
            "dataset_display": result.get("dataset_display", result.get("run_name")),
            "modality": result.get("modality"),
            "stain": result.get("stain"),
            "task_family": result.get("task_family"),
            "model_id": model_id,
            "model_label": model_label(result),
            "method": method_label(result),
            "model_family": model_family(model_id),
            "split": split,
            "total_samples": result.get("n_samples"),
            "split_samples": _split_sample_count(result, split),
            "best_lr": best.get("learning_rate"),
            "best_weight_decay": best.get("weight_decay"),
            "best_class_weight": best.get("class_weight"),
            "best_epoch": best.get("best_epoch"),
            "final_best_epoch": final_training.get("best_epoch"),
            "epochs_run": final_training.get("epochs_run"),
            "arch": result.get("arch"),
            "pretrained": result.get("pretrained") if result.get("analysis_source") == "cnn" else None,
            "tune_seconds": timing.get("tune_seconds"),
            "final_train_seconds": timing.get("final_train_seconds"),
        }
        row.update({k: metrics.get(k) for k in SUMMARY_METRICS})

        for cls in class_names:
            cls_metrics = per_label.get(cls, {})
            for metric in PER_LABEL_METRICS:
                row[f"{cls}_{metric}"] = cls_metrics.get(metric)
            row[f"{cls}_support"] = cls_metrics.get("support")

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def probe_results_to_frame(
    probe_results: list[dict[str, Any]],
    *,
    split: Literal["hold-out", "validation"] = "hold-out",
    class_names: Sequence[str] = (),
) -> pd.DataFrame:
    """Backward-compatible alias for ``results_to_frame``."""
    return results_to_frame(probe_results, split=split, class_names=class_names)


def build_leaderboard(
    results: list[dict[str, Any]],
    *,
    class_names: Sequence[str] = (),
    timing_md: Path | None = None,
) -> pd.DataFrame:
    holdout = results_to_frame(results, split="hold-out", class_names=class_names)
    validation = results_to_frame(results, split="validation", class_names=class_names)
    if holdout.empty:
        return holdout

    leaderboard = holdout.rename(
        columns={c: f"holdout_{c}" for c in holdout.columns if c not in ID_COLS}
    )

    val_metric_cols = ("macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall")
    merge_keys = ["model_id"]
    if "run_key" in leaderboard.columns and "run_key" in validation.columns:
        merge_keys = ["run_key", "model_id"]
    if validation.empty:
        for src in val_metric_cols:
            leaderboard[f"val_{src}"] = np.nan
    else:
        val_cols = merge_keys + [src for src in val_metric_cols if src in validation.columns]
        val_df = validation[val_cols].rename(columns={src: f"val_{src}" for src in val_metric_cols})
        leaderboard = leaderboard.merge(val_df, on=merge_keys, how="left")

    timings = load_embedding_timings(timing_md)
    if not timings.empty:
        leaderboard = leaderboard.merge(timings[["model_id", "ms_per_tile", "embed_dim"]], on="model_id", how="left")
    else:
        leaderboard["ms_per_tile"] = np.nan
        leaderboard["embed_dim"] = np.nan

    leaderboard["estimated_embedding_seconds"] = (
        leaderboard["ms_per_tile"] * leaderboard["holdout_total_samples"] / 1000.0
    )
    leaderboard["total_training_seconds"] = leaderboard[
        ["holdout_tune_seconds", "holdout_final_train_seconds"]
    ].sum(axis=1, min_count=1)
    leaderboard["available_compute_seconds"] = leaderboard[
        ["estimated_embedding_seconds", "total_training_seconds"]
    ].sum(axis=1, min_count=1)
    leaderboard["macro_f1_delta_val_to_holdout"] = (
        leaderboard["holdout_macro_f1"] - leaderboard["val_macro_f1"]
    )
    return leaderboard.sort_values("holdout_macro_f1", ascending=False).reset_index(drop=True)


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None or pd.isna(seconds):
        return ""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def _comparison_subset(df: pd.DataFrame, selector: str) -> pd.DataFrame:
    if selector == "embedding":
        return df[df["method"] == "Frozen embedding + linear probe"]
    if selector == "cnn":
        return df[df["method"] == "CNN fine-tune"]
    return df[df["model_family"] == selector]


def build_cross_dataset_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if leaderboard.empty:
        return pd.DataFrame(rows)

    for run_key, run_df in leaderboard.groupby("run_key", dropna=False):
        for group_label, selector in COMPARISON_GROUPS:
            subset = _comparison_subset(run_df, selector).dropna(subset=["holdout_macro_f1"])
            if subset.empty:
                continue
            best = subset.sort_values("holdout_macro_f1", ascending=False).iloc[0]
            rows.append(
                {
                    "run_key": run_key,
                    "run_name": best.get("run_name"),
                    "dataset_display": best.get("dataset_display"),
                    "modality": best.get("modality"),
                    "stain": best.get("stain"),
                    "task_family": best.get("task_family"),
                    "comparison_group": group_label,
                    "model_id": best["model_id"],
                    "model_label": best["model_label"],
                    "model_family": best["model_family"],
                    "holdout_macro_f1": best["holdout_macro_f1"],
                    "holdout_balanced_accuracy": best.get("holdout_balanced_accuracy"),
                    "val_macro_f1": best.get("val_macro_f1"),
                    "available_compute_seconds": best.get("available_compute_seconds"),
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["dataset_display", "comparison_group"]).reset_index(drop=True)


def build_cnn_vs_embedding_delta(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    pivot = summary.pivot_table(
        index=["run_key", "dataset_display", "modality", "task_family"],
        columns="comparison_group",
        values="holdout_macro_f1",
        aggfunc="max",
    ).reset_index()
    if "CNN fine-tune" in pivot and "Best embedding probe" in pivot:
        pivot["cnn_minus_best_embedding_macro_f1"] = pivot["CNN fine-tune"] - pivot["Best embedding probe"]
    return pivot.reset_index(drop=True)


def sorted_results(
    results: list[dict[str, Any]],
    *,
    split: Literal["hold-out", "validation"] = "hold-out",
) -> list[dict[str, Any]]:
    kept = [r for r in results if not r.get("skipped")]
    return sorted(kept, key=lambda r: _metrics_block(r, split).get("macro_f1", 0), reverse=True)


def sorted_probe_results(
    probe_results: list[dict[str, Any]],
    *,
    split: Literal["hold-out", "validation"] = "hold-out",
) -> list[dict[str, Any]]:
    return sorted_results(probe_results, split=split)


def get_result(results: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for result in results:
        if result.get("model_id") == model_id:
            return result
    raise KeyError(f"Unknown model_id: {model_id!r}")


def get_probe_result(probe_results: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    return get_result(probe_results, model_id)
