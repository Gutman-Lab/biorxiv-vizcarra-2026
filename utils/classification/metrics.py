"""Shared single-label classification metrics."""

from __future__ import annotations

from typing import Any

import torch


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
    )


def compute_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    """
    Build a confusion matrix with rows=true class and columns=predicted class.

    Returns ``{"labels": [...], "matrix": [[...], ...]}`` suitable for JSON.
    """
    y_true = y_true.flatten().to(dtype=torch.long)
    y_pred = y_pred.flatten().to(dtype=torch.long)
    indices = num_classes * y_true + y_pred
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    matrix = counts.reshape(num_classes, num_classes).tolist()
    return {"matrix": matrix}


def eval_single_label(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    class_names: list[str],
    metric_name: str,
) -> dict[str, Any]:
    """
    Compute aggregate and per-class metrics for single-label multiclass predictions.

    ``metric_name`` must be a key in the returned ``metrics`` dict (e.g. ``macro_f1``).
    The same value is also exposed as ``selection_score`` for hyperparameter tuning.
    """
    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError("class_names must not be empty")

    per_label: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    balanced_accs: list[float] = []
    label_support: dict[str, int] = {}

    for class_idx, name in enumerate(class_names):
        true_i = y_true == class_idx
        pred_i = y_pred == class_idx
        tp = int((true_i & pred_i).sum().item())
        fp = int((~true_i & pred_i).sum().item())
        fn = int((true_i & ~pred_i).sum().item())
        tn = int((~true_i & ~pred_i).sum().item())

        support = int(true_i.sum().item())
        label_support[name] = support
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced = (recall + specificity) / 2.0

        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        balanced_accs.append(balanced)
        per_label[name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced,
        }

    accuracy = float((y_true == y_pred).float().mean().item())
    metrics = {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
        "macro_precision": sum(precisions) / num_classes,
        "balanced_accuracy": sum(balanced_accs) / num_classes,
    }
    if metric_name not in metrics:
        raise ValueError(
            f"unknown metric_name {metric_name!r}; expected one of {sorted(metrics)}"
        )
    metrics["selection_score"] = metrics[metric_name]

    confusion = compute_confusion_matrix(y_true, y_pred, num_classes)
    confusion["labels"] = class_names

    return {
        "metrics": metrics,
        "per_label": per_label,
        "label_support": label_support,
        "confusion_matrix": confusion,
    }
