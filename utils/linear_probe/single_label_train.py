"""PyTorch single-label (multiclass) linear probe training and evaluation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import TEST_SPLIT, TUNE_SPLIT
from utils.classification.labels import labels_to_class_index
from utils.classification.metrics import eval_single_label, format_metrics
from utils.linear_probe.common import best_checkpoint_path, last_checkpoint_path
from utils.linear_probe.multilabel_train import scaled_learning_rate
from utils.torch_device import get_torch_device


def _torch_device() -> torch.device:
    return get_torch_device()


def _num_classes(args: Namespace) -> int:
    return len(args.class_names)


def _samples_to_tensors(
    samples: list[dict[str, Any]], *, normalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("cannot build tensors from an empty sample list")

    embeddings = torch.from_numpy(
        np.asarray([s["embedding"] for s in samples], dtype=np.float32)
    )
    class_indices = torch.tensor(
        [labels_to_class_index(s["labels"]) for s in samples],
        dtype=torch.long,
    )
    if normalize:
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return embeddings, class_indices


def _class_weights(labels: torch.Tensor, mode: str, num_classes: int) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp(min=1.0) * num_classes)
    return weights


def _build_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    device = _torch_device()
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
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

    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch_x, batch_y in loader:
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

    if total_samples == 0:
        raise ValueError("cannot run epoch on an empty DataLoader")

    return total_loss / total_samples, torch.cat(labels, dim=0), torch.cat(preds, dim=0)


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


def _best_epoch_record(
    training_history: list[dict[str, Any]],
    best_epoch: int,
) -> dict[str, Any]:
    for record in training_history:
        if record["epoch"] == best_epoch:
            return record
    if training_history:
        return training_history[-1]
    return {}


def _grid_checkpoint_dir(
    checkpoint_root: Path,
    *,
    learning_rate: float,
    class_weight_mode: str,
) -> Path:
    name = f"lr_{learning_rate:g}_cw_{class_weight_mode}"
    return checkpoint_root / name


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    in_features: int,
    num_classes: int,
) -> nn.Module:
    device = _torch_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = nn.Linear(in_features, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def train_one_run(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    *,
    learning_rate: float,
    class_weight_mode: str,
    checkpoint_dir: Path,
    args: Namespace,
    epochs: int,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Train a linear probe with per-epoch validation checkpointing.

    Saves ``last.pt`` every epoch and ``best.pt`` when validation ``args.metric``
    improves (default macro-F1). Stops early when the metric does not improve by
    at least ``early_stop_min_delta`` for ``early_stop_patience`` epochs
    (0 patience disables early stopping).
    """
    if not train_samples:
        raise ValueError("train_samples must not be empty")
    if not val_samples:
        raise ValueError("val_samples must not be empty for checkpoint selection")
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if early_stop_patience < 0:
        raise ValueError(f"early_stop_patience must be non-negative, got {early_stop_patience}")
    if early_stop_min_delta < 0:
        raise ValueError(f"early_stop_min_delta must be non-negative, got {early_stop_min_delta}")

    torch.manual_seed(args.seed)
    device = _torch_device()
    num_classes = _num_classes(args)

    train_x, train_y = _samples_to_tensors(train_samples, normalize=args.normalize_embeddings)
    val_x, val_y = _samples_to_tensors(val_samples, normalize=args.normalize_embeddings)
    in_features = train_x.shape[1]

    model = nn.Linear(in_features, num_classes).to(device)
    nn.init.normal_(model.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.bias)

    class_weight = _class_weights(train_y, class_weight_mode, num_classes)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight.to(device) if class_weight is not None else None
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=scaled_learning_rate(learning_rate, args.batch_size),
        weight_decay=args.weight_decay,
    )
    train_loader = _build_loader(
        train_x,
        train_y,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = _build_loader(
        val_x,
        val_y,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    patience = early_stop_patience
    min_delta = early_stop_min_delta

    if hyperparams is None:
        hyperparams = {
            "learning_rate": learning_rate,
            "scaled_learning_rate": scaled_learning_rate(learning_rate, args.batch_size),
            "class_weight": class_weight_mode,
            "batch_size": args.batch_size,
            "epochs": epochs,
            "weight_decay": args.weight_decay,
            "metric": args.metric,
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
            "seed": args.seed,
        }

    best_path = best_checkpoint_path(checkpoint_dir)
    last_path = last_checkpoint_path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        )
        val_loss, val_labels, val_pred = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            train=False,
        )
        val_eval = eval_single_label(
            val_labels,
            val_pred,
            class_names=args.class_names,
            metric_name=args.metric,
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

        _save_checkpoint(
            last_path,
            model=model,
            epoch=epoch,
            hyperparams=hyperparams,
            val_metrics=val_metrics,
        )

        improved = score > best_score + min_delta
        if improved:
            best_epoch = epoch
            best_score = score
            best_val_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            _save_checkpoint(
                best_path,
                model=model,
                epoch=epoch,
                hyperparams=hyperparams,
                val_metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(
            f"    epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"{format_metrics(val_metrics)}{marker}",
            flush=True,
        )

        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            print(
                f"    early stop at epoch {epoch} "
                f"(no {args.metric} improvement for {patience} epochs)",
                flush=True,
            )
            break

    return {
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "training_history": training_history,
        "epochs_run": len(training_history),
        "stopped_early": stopped_early,
    }


def evaluate_linear_probe(
    model: nn.Module,
    eval_samples: list[dict[str, Any]],
    *,
    metric_name: str,
    args: Namespace,
) -> dict[str, Any]:
    if not eval_samples:
        return {
            "metrics": {},
            "per_label": {},
            "n_samples": 0,
            "label_support": {},
            "confusion_matrix": {},
        }

    eval_x, eval_y = _samples_to_tensors(eval_samples, normalize=args.normalize_embeddings)
    device = _torch_device()
    model = model.to(device).eval()

    loader = _build_loader(
        eval_x,
        eval_y,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    preds: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_x, _ in loader:
            preds.append(model(batch_x.to(device, non_blocking=True)).argmax(dim=1).cpu())

    y_pred = torch.cat(preds, dim=0)
    result = eval_single_label(
        eval_y,
        y_pred,
        class_names=args.class_names,
        metric_name=metric_name,
    )
    result["n_samples"] = len(eval_samples)
    return result


def tune_hyperparameters(
    *,
    model_id: str,
    train_samples: list[dict[str, Any]],
    tune_samples: list[dict[str, Any]],
    output_dir: Path,
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

    from utils.linear_probe.common import tune_checkpoint_root

    print(
        f"  [{model_id}] tuning on {TUNE_SPLIT} "
        f"({len(tune_samples):,} tiles, {args.tune_epochs} epochs/cell, "
        f"metric={args.metric})...",
        flush=True,
    )

    checkpoint_root = tune_checkpoint_root(output_dir, model_id)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    grid_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    tune_t0 = time.perf_counter()

    for lr in args.learning_rates:
        for cw in args.class_weights:
            checkpoint_dir = _grid_checkpoint_dir(
                checkpoint_root,
                learning_rate=lr,
                class_weight_mode=cw,
            )
            print(f"    lr={lr:g}, class_weight={cw}", flush=True)
            run_result = train_one_run(
                train_samples,
                tune_samples,
                learning_rate=lr,
                class_weight_mode=cw,
                checkpoint_dir=checkpoint_dir,
                args=args,
                epochs=args.tune_epochs,
                early_stop_patience=0,
                early_stop_min_delta=args.early_stop_min_delta,
            )
            best_record = _best_epoch_record(
                run_result["training_history"],
                run_result["best_epoch"],
            )
            candidate = {
                "learning_rate": lr,
                "class_weight": cw,
                "scaled_learning_rate": scaled_learning_rate(lr, args.batch_size),
                "best_epoch": run_result["best_epoch"],
                "best_checkpoint_path": run_result["best_checkpoint_path"],
                "last_checkpoint_path": run_result["last_checkpoint_path"],
                "validation_metrics": run_result["best_val_metrics"],
                "validation_per_label": best_record.get("val_per_label", {}),
                "validation_label_support": best_record.get("val_label_support", {}),
                "validation_confusion_matrix": best_record.get("val_confusion_matrix", {}),
                "training_history": run_result["training_history"],
            }
            grid_results.append(candidate)
            print(
                f"      -> {format_metrics(candidate['validation_metrics'])} "
                f"(best epoch {candidate['best_epoch']})",
                flush=True,
            )

            score = candidate["validation_metrics"].get("selection_score", float("-inf"))
            if best is None or score > best["validation_metrics"].get(
                "selection_score",
                float("-inf"),
            ):
                best = candidate

    assert best is not None
    tune_seconds = time.perf_counter() - tune_t0
    print(
        f"  [{model_id}] best on {TUNE_SPLIT}: lr={best['learning_rate']:g}, "
        f"class_weight={best['class_weight']}, "
        f"{format_metrics(best['validation_metrics'])} "
        f"(tuning {tune_seconds:.1f}s)",
        flush=True,
    )
    return {**best, "grid_results": grid_results, "tune_seconds": tune_seconds}


def evaluate_probe(
    *,
    model_id: str,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    best_config: dict[str, Any],
    output_dir: Path,
    args: Namespace,
) -> dict[str, Any]:
    if not test_samples:
        print(f"  [{model_id}] no {TEST_SPLIT} samples; skipping hold-out evaluation.", flush=True)
        return {"test_metrics": {}, "n_samples": 0}

    from utils.linear_probe.common import final_checkpoint_dir

    print(
        f"  [{model_id}] final training on train ({len(train_samples):,} tiles) with "
        f"{TUNE_SPLIT} monitoring ({len(val_samples):,} tiles, up to {args.epochs} epochs): "
        f"lr={best_config['learning_rate']:g}, class_weight={best_config['class_weight']}",
        flush=True,
    )
    checkpoint_dir = final_checkpoint_dir(output_dir, model_id)
    final_train_t0 = time.perf_counter()
    final_training = train_one_run(
        train_samples,
        val_samples,
        learning_rate=best_config["learning_rate"],
        class_weight_mode=best_config["class_weight"],
        checkpoint_dir=checkpoint_dir,
        args=args,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    final_train_seconds = time.perf_counter() - final_train_t0

    train_x, _ = _samples_to_tensors(train_samples, normalize=args.normalize_embeddings)
    model = _load_model_from_checkpoint(
        Path(final_training["best_checkpoint_path"]),
        in_features=train_x.shape[1],
        num_classes=_num_classes(args),
    )

    print(
        f"  [{model_id}] evaluating on {TEST_SPLIT} ({len(test_samples):,} tiles) "
        f"from checkpoint {Path(final_training['best_checkpoint_path']).name}...",
        flush=True,
    )
    ev = evaluate_linear_probe(model, test_samples, metric_name=args.metric, args=args)
    print(
        f"  [{model_id}] {TEST_SPLIT}: {format_metrics(ev['metrics'])} "
        f"(final train {final_train_seconds:.1f}s)",
        flush=True,
    )
    return {
        "test_metrics": ev["metrics"],
        "per_label": ev["per_label"],
        "label_support": ev["label_support"],
        "confusion_matrix": ev.get("confusion_matrix", {}),
        "n_samples": ev["n_samples"],
        "final_training": final_training,
        "final_train_seconds": final_train_seconds,
    }
