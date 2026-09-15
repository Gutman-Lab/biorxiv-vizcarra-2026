"""Hyperparameter grid search for single-label CNN classifiers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import time
from typing import Any

from torch.utils.data import DataLoader

from config import TUNE_SPLIT
from utils.classification.metrics import format_metrics
from utils.cnn_classifier.class_weights import compute_class_weights_from_samples
from utils.cnn_classifier.data import ImageSample, build_eval_dataloader, build_train_dataloader
from utils.cnn_classifier.models import build_classifier
from utils.cnn_classifier.sampler import build_train_sampler_from_samples
from utils.cnn_classifier.train import train_one_run
from utils.cnn_classifier.common import tune_checkpoint_root
from utils.cnn_classifier.transforms import build_eval_transform, build_train_transform
from utils.torch_device import get_torch_device

DEFAULT_LEARNING_RATES = (1e-4, 3e-4, 1e-3)
DEFAULT_WEIGHT_DECAYS = (1e-5, 1e-4)
DEFAULT_CLASS_WEIGHTS = ("none", "sqrt_inverse_freq", "effective_num")

TUNE_METRICS = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "macro_recall",
    "macro_precision",
)


def build_train_val_dataloaders(
    train_samples: list[ImageSample],
    val_samples: list[ImageSample],
    *,
    num_classes: int,
    image_size: int,
    batch_size: int,
    sampler_mode: str,
    num_workers: int,
) -> tuple[DataLoader[tuple[Any, int]], DataLoader[tuple[Any, int]]]:
    """Build train and validation DataLoaders for one training run."""
    pin_memory = get_torch_device().type == "cuda"
    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)
    train_sampler = build_train_sampler_from_samples(
        train_samples,
        sampler_mode,
        num_classes,
    )
    train_loader = build_train_dataloader(
        train_samples,
        transform=train_transform,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_eval_dataloader(
        val_samples,
        transform=eval_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _grid_checkpoint_path(
    checkpoint_dir: Path,
    *,
    learning_rate: float,
    weight_decay: float,
    class_weight_mode: str,
) -> Path:
    name = (
        f"lr_{learning_rate:g}_wd_{weight_decay:g}_cw_{class_weight_mode}.pt"
    )
    return checkpoint_dir / name


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


def _run_grid_cell(
    *,
    train_samples: list[ImageSample],
    val_samples: list[ImageSample],
    class_names: list[str],
    checkpoint_path: Path,
    args: Namespace,
    learning_rate: float,
    weight_decay: float,
    class_weight_mode: str,
) -> dict[str, Any]:
    num_classes = len(class_names)
    model = build_classifier(
        args.arch,
        num_classes,
        pretrained=args.pretrained,
    )
    train_loader, val_loader = build_train_val_dataloaders(
        train_samples,
        val_samples,
        num_classes=num_classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        sampler_mode=args.sampler,
        num_workers=args.num_workers,
    )
    class_weights = compute_class_weights_from_samples(
        train_samples,
        class_weight_mode,
        num_classes,
    )
    hyperparams = {
        "arch": args.arch,
        "pretrained": args.pretrained,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "class_weight": class_weight_mode,
        "sampler": args.sampler,
        "batch_size": args.batch_size,
        "epochs": args.tune_epochs,
        "image_size": args.image_size,
        "metric": args.metric,
        "early_stop_patience": 0,
        "early_stop_min_delta": args.early_stop_min_delta,
        "seed": args.seed,
    }
    run_result = train_one_run(
        model,
        train_loader,
        val_loader,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        hyperparams=hyperparams,
        epochs=args.tune_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights,
        metric_name=args.metric,
        seed=args.seed,
        early_stop_patience=0,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    best_record = _best_epoch_record(
        run_result["training_history"],
        run_result["best_epoch"],
    )
    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "class_weight": class_weight_mode,
        "best_epoch": run_result["best_epoch"],
        "best_checkpoint_path": run_result["best_checkpoint_path"],
        "validation_metrics": run_result["best_val_metrics"],
        "validation_per_label": best_record.get("val_per_label", {}),
        "validation_label_support": best_record.get("val_label_support", {}),
        "validation_confusion_matrix": best_record.get("val_confusion_matrix", {}),
    }


def tune_hyperparameters(
    *,
    train_samples: list[ImageSample],
    val_samples: list[ImageSample],
    output_dir: Path,
    args: Namespace,
) -> dict[str, Any]:
    """
    Grid search over learning rate, weight decay, and class-weight mode.

    Each configuration runs ``train_one_run`` on the train split with validation
    checkpointing on ``val_samples``. The winner is selected by validation
    ``args.metric`` (default macro-F1).
    """
    if not train_samples:
        raise SystemExit("No training samples available.")
    if not val_samples:
        raise SystemExit(
            f"No {TUNE_SPLIT} samples available for hyperparameter tuning."
        )

    class_names: list[str] = args.class_names
    if not class_names:
        raise ValueError("args.class_names must not be empty")

    print(
        f"Tuning on {TUNE_SPLIT} ({len(val_samples):,} tiles, "
        f"{args.tune_epochs} epochs/cell, metric={args.metric})...",
        flush=True,
    )

    checkpoint_dir = tune_checkpoint_root(output_dir, args.arch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    grid_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    tune_t0 = time.perf_counter()

    for learning_rate in args.learning_rates:
        for weight_decay in args.weight_decays:
            for class_weight_mode in args.class_weights:
                checkpoint_path = _grid_checkpoint_path(
                    checkpoint_dir,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    class_weight_mode=class_weight_mode,
                )
                print(
                    f"  lr={learning_rate:g}, weight_decay={weight_decay:g}, "
                    f"class_weight={class_weight_mode}",
                    flush=True,
                )
                candidate = _run_grid_cell(
                    train_samples=train_samples,
                    val_samples=val_samples,
                    class_names=class_names,
                    checkpoint_path=checkpoint_path,
                    args=args,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    class_weight_mode=class_weight_mode,
                )
                grid_results.append(candidate)
                print(
                    f"    -> {format_metrics(candidate['validation_metrics'])} "
                    f"(best epoch {candidate['best_epoch']})",
                    flush=True,
                )

                score = candidate["validation_metrics"].get(
                    "selection_score",
                    float("-inf"),
                )
                if best is None or score > best["validation_metrics"].get(
                    "selection_score",
                    float("-inf"),
                ):
                    best = candidate

    assert best is not None
    tune_seconds = time.perf_counter() - tune_t0
    print(
        f"Best on {TUNE_SPLIT}: lr={best['learning_rate']:g}, "
        f"weight_decay={best['weight_decay']:g}, "
        f"class_weight={best['class_weight']}, "
        f"{format_metrics(best['validation_metrics'])} "
        f"(tuning {tune_seconds:.1f}s)",
        flush=True,
    )
    return {**best, "grid_results": grid_results, "tune_seconds": tune_seconds}
