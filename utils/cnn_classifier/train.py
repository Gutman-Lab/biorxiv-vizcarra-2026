"""Single-run CNN classifier training with validation checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.classification.metrics import eval_single_label, format_metrics
from utils.torch_device import get_torch_device


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    epochs: int | None = None,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    labels: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []

    phase = "train" if train else "val"
    if epoch is not None and epochs is not None:
        desc = f"{phase} {epoch}/{epochs}"
    elif epoch is not None:
        desc = f"{phase} {epoch}"
    else:
        desc = phase

    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch")
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_size = batch_y.shape[0]

            if train:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            else:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            labels.append(batch_y.detach().cpu())
            preds.append(logits.argmax(dim=1).detach().cpu())
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        raise ValueError("cannot run epoch on an empty DataLoader")

    mean_loss = total_loss / total_samples
    return mean_loss, torch.cat(labels, dim=0), torch.cat(preds, dim=0)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    hyperparams: dict[str, Any],
    val_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "hyperparams": hyperparams,
            "val_metrics": val_metrics,
        },
        path,
    )


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    class_names: list[str],
    checkpoint_path: Path,
    hyperparams: dict[str, Any],
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    class_weights: torch.Tensor | None = None,
    metric_name: str = "macro_f1",
    seed: int = 0,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float | None = None,
) -> dict[str, Any]:
    """
    Fine-tune a classifier for a fixed number of epochs.

    Each epoch trains on ``train_loader``, evaluates on ``val_loader``, and saves
    ``checkpoint_path`` when validation ``metric_name`` improves (default macro-F1).
    Stops early when the metric does not improve by at least ``early_stop_min_delta``
    for ``early_stop_patience`` epochs (0 patience disables early stopping).
    Uses Adam with a constant learning rate.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if not class_names:
        raise ValueError("class_names must not be empty")

    patience = (
        int(early_stop_patience)
        if early_stop_patience is not None
        else int(hyperparams.get("early_stop_patience", 0) or 0)
    )
    min_delta = (
        float(early_stop_min_delta)
        if early_stop_min_delta is not None
        else float(hyperparams.get("early_stop_min_delta", 0.0) or 0.0)
    )
    if patience < 0:
        raise ValueError(f"early_stop_patience must be non-negative, got {patience}")
    if min_delta < 0:
        raise ValueError(f"early_stop_min_delta must be non-negative, got {min_delta}")

    hyperparams = {
        **hyperparams,
        "epochs": epochs,
        "early_stop_patience": patience,
        "early_stop_min_delta": min_delta,
    }

    torch.manual_seed(seed)
    device = get_torch_device()
    model = model.to(device)

    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_epoch = 0
    best_score = float("-inf")
    best_val_metrics: dict[str, float] = {}
    training_history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            train=True,
            optimizer=optimizer,
            epoch=epoch,
            epochs=epochs,
        )
        val_loss, val_y, val_pred = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            train=False,
            epoch=epoch,
            epochs=epochs,
        )
        val_eval = eval_single_label(
            val_y,
            val_pred,
            class_names=class_names,
            metric_name=metric_name,
        )
        val_metrics = val_eval["metrics"]
        score = val_metrics["selection_score"]

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "val_per_label": val_eval["per_label"],
            "val_label_support": val_eval["label_support"],
            "val_confusion_matrix": val_eval["confusion_matrix"],
        }
        training_history.append(epoch_record)

        improved = score > best_score + min_delta
        if improved:
            best_epoch = epoch
            best_score = score
            best_val_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path,
                model=model,
                epoch=epoch,
                hyperparams=hyperparams,
                val_metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(
            f"  epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"{format_metrics(val_metrics)}{marker}",
            flush=True,
        )

        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            print(
                f"  early stop at epoch {epoch} "
                f"(no {metric_name} improvement for {patience} epochs)",
                flush=True,
            )
            break

    return {
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_checkpoint_path": str(checkpoint_path),
        "training_history": training_history,
        "epochs_run": len(training_history),
        "stopped_early": stopped_early,
    }
