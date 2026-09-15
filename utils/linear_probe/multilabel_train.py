"""PyTorch multi-label linear probe training, evaluation, and hyperparameter search."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import TEST_SPLIT, TUNE_SPLIT
from utils.torch_device import get_torch_device


def scaled_learning_rate(learning_rate: float, batch_size: int) -> float:
    return learning_rate * batch_size / 256.0


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"macro_recall={metrics['macro_recall']:.4f}, "
        f"subset_accuracy={metrics['subset_accuracy']:.4f}"
    )


def _torch_device() -> torch.device:
    return get_torch_device()


def _num_labels(args: Namespace) -> int:
    return len(args.class_names)


def _samples_to_tensors(
    samples: list[dict[str, Any]], *, normalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("cannot build tensors from an empty sample list")

    embeddings = torch.from_numpy(
        np.asarray([s["embedding"] for s in samples], dtype=np.float32)
    )
    labels = torch.tensor(
        [s["labels"] for s in samples],
        dtype=torch.float32,
    )
    if normalize:
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return embeddings, labels


def _pos_weights(labels: torch.Tensor, mode: str) -> torch.Tensor | None:
    """Per-label positive weights for BCEWithLogitsLoss (neg_count / pos_count)."""
    if mode == "none":
        return None
    pos = labels.sum(dim=0)
    neg = labels.shape[0] - pos
    return neg / pos.clamp(min=1.0)


def _predict_labels(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).to(torch.int)


def _eval_multilabel(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    metric_name: str,
    args: Namespace,
) -> dict[str, Any]:
    y_true_i = y_true.to(torch.int)
    y_pred_i = y_pred.to(torch.int)
    class_names: list[str] = args.class_names
    num_labels = len(class_names)

    per_label: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    balanced_accs: list[float] = []

    label_support: dict[str, int] = {}
    for i, name in enumerate(class_names):
        true_i = y_true_i[:, i].bool()
        pred_i = y_pred_i[:, i].bool()
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

    tp_total = int((y_true_i.bool() & y_pred_i.bool()).sum().item())
    fp_total = int((~y_true_i.bool() & y_pred_i.bool()).sum().item())
    fn_total = int((y_true_i.bool() & ~y_pred_i.bool()).sum().item())
    micro_precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    subset_accuracy = float((y_true_i == y_pred_i).all(dim=1).float().mean().item())
    hamming_loss = float((y_true_i != y_pred_i).float().mean().item())

    metrics = {
        "macro_f1": sum(f1s) / num_labels,
        "macro_recall": sum(recalls) / num_labels,
        "macro_precision": sum(precisions) / num_labels,
        "balanced_accuracy": sum(balanced_accs) / num_labels,
        "micro_f1": micro_f1,
        "subset_accuracy": subset_accuracy,
        "hamming_loss": hamming_loss,
    }
    metrics["selection_score"] = metrics[metric_name]

    return {
        "metrics": metrics,
        "per_label": per_label,
        "label_support": label_support,
    }


def train_linear_probe(
    train_samples: list[dict[str, Any]],
    *,
    learning_rate: float,
    class_weight_mode: str,
    args: Namespace,
) -> nn.Module:
    torch.manual_seed(args.seed)
    train_x, train_y = _samples_to_tensors(train_samples, normalize=args.normalize_embeddings)
    device = _torch_device()
    num_labels = _num_labels(args)

    model = nn.Linear(train_x.shape[1], num_labels).to(device)
    nn.init.normal_(model.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.bias)

    pos_weight = _pos_weights(train_y, class_weight_mode)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=scaled_learning_rate(learning_rate, args.batch_size),
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model.train()
    for _ in range(args.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(batch_x), batch_y).backward()
            optimizer.step()
    return model


def evaluate_linear_probe(
    model: nn.Module,
    eval_samples: list[dict[str, Any]],
    *,
    metric_name: str,
    args: Namespace,
) -> dict[str, Any]:
    if not eval_samples:
        return {"metrics": {}, "per_label": {}, "n_samples": 0, "label_support": {}}

    eval_x, eval_y = _samples_to_tensors(eval_samples, normalize=args.normalize_embeddings)
    device = _torch_device()
    model = model.to(device).eval()

    loader = DataLoader(
        TensorDataset(eval_x, eval_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    preds: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_x, _ in loader:
            preds.append(_predict_labels(model(batch_x.to(device, non_blocking=True))))

    y_pred = torch.cat(preds, dim=0)
    result = _eval_multilabel(eval_y, y_pred, metric_name, args)
    result["n_samples"] = len(eval_samples)
    return result


def tune_hyperparameters(
    *,
    model_id: str,
    train_samples: list[dict[str, Any]],
    tune_samples: list[dict[str, Any]],
    args: Namespace,
) -> dict[str, Any]:
    if args.probe_type != "linear":
        raise NotImplementedError(
            f"probe type {args.probe_type!r} is not implemented yet; use --probe-type linear"
        )
    if not train_samples:
        raise SystemExit(f"[{model_id}] no training samples available.")
    if not tune_samples:
        raise SystemExit(f"[{model_id}] no {TUNE_SPLIT} samples available for tuning.")

    print(
        f"  [{model_id}] tuning on {TUNE_SPLIT} "
        f"({len(tune_samples):,} tiles, metric={args.metric})...",
        flush=True,
    )

    grid_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for lr in args.learning_rates:
        for cw in args.class_weights:
            model = train_linear_probe(
                train_samples, learning_rate=lr, class_weight_mode=cw, args=args
            )
            ev = evaluate_linear_probe(model, tune_samples, metric_name=args.metric, args=args)
            candidate = {
                "learning_rate": lr,
                "class_weight": cw,
                "scaled_learning_rate": scaled_learning_rate(lr, args.batch_size),
                "validation_metrics": ev["metrics"],
                "validation_per_label": ev["per_label"],
                "validation_label_support": ev["label_support"],
            }
            grid_results.append(candidate)
            print(f"    lr={lr:g}, class_weight={cw}: {format_metrics(ev['metrics'])}", flush=True)

            score = ev["metrics"].get("selection_score", float("-inf"))
            if best is None or score > best["validation_metrics"].get("selection_score", float("-inf")):
                best = candidate

    assert best is not None
    print(
        f"  [{model_id}] best on {TUNE_SPLIT}: lr={best['learning_rate']:g}, "
        f"class_weight={best['class_weight']}, {format_metrics(best['validation_metrics'])}",
        flush=True,
    )
    return {**best, "grid_results": grid_results}


def evaluate_probe(
    *,
    model_id: str,
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    best_config: dict[str, Any],
    args: Namespace,
) -> dict[str, Any]:
    if not test_samples:
        print(f"  [{model_id}] no {TEST_SPLIT} samples; skipping hold-out evaluation.", flush=True)
        return {"test_metrics": {}, "n_samples": 0}

    print(
        f"  [{model_id}] evaluating on {TEST_SPLIT} ({len(test_samples):,} tiles, "
        f"lr={best_config['learning_rate']:g}, class_weight={best_config['class_weight']})...",
        flush=True,
    )
    model = train_linear_probe(
        train_samples,
        learning_rate=best_config["learning_rate"],
        class_weight_mode=best_config["class_weight"],
        args=args,
    )
    ev = evaluate_linear_probe(model, test_samples, metric_name=args.metric, args=args)
    print(f"  [{model_id}] {TEST_SPLIT}: {format_metrics(ev['metrics'])}", flush=True)
    return {
        "test_metrics": ev["metrics"],
        "per_label": ev["per_label"],
        "label_support": ev["label_support"],
        "n_samples": ev["n_samples"],
    }
