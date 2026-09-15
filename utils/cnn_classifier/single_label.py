"""Single-label CNN classifier orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TEST_SPLIT, TRAIN_SPLIT, TUNE_SPLIT
from utils.classification.labels import labels_to_class_index
from utils.classification.metrics import eval_single_label, format_metrics
from utils.cnn_classifier.class_weights import compute_class_weights_from_samples
from utils.cnn_classifier.common import (
    best_checkpoint_path,
    final_checkpoint_dir,
    save_result,
    tune_checkpoint_root,
)
from utils.cnn_classifier.data import ImageSample, build_eval_dataloader
from utils.cnn_classifier.models import build_classifier
from utils.cnn_classifier.train import train_one_run
from utils.cnn_classifier.transforms import build_eval_transform
from utils.cnn_classifier.tune import build_train_val_dataloaders, tune_hyperparameters
from utils.linear_probe.data import load_image_samples, split_samples
from utils.torch_device import get_torch_device


def _class_support(samples: list[ImageSample], class_names: list[str]) -> dict[str, int]:
    support = {name: 0 for name in class_names}
    for sample in samples:
        class_idx = labels_to_class_index(sample["labels"])
        support[class_names[class_idx]] += 1
    return support


def _split_class_support(
    train_samples: list[ImageSample],
    val_samples: list[ImageSample],
    test_samples: list[ImageSample],
    class_names: list[str],
) -> dict[str, dict[str, int]]:
    return {
        TRAIN_SPLIT: _class_support(train_samples, class_names),
        TUNE_SPLIT: _class_support(val_samples, class_names),
        TEST_SPLIT: _class_support(test_samples, class_names),
    }


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[Any, int]],
    *,
    class_names: list[str],
    metric_name: str,
) -> dict[str, Any]:
    device = get_torch_device()
    model = model.to(device).eval()

    labels: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(batch_y.cpu())

    if not labels:
        return {
            "metrics": {},
            "per_label": {},
            "label_support": {},
            "confusion_matrix": {},
            "n_samples": 0,
        }

    y_true = torch.cat(labels, dim=0)
    y_pred = torch.cat(preds, dim=0)
    result = eval_single_label(
        y_true,
        y_pred,
        class_names=class_names,
        metric_name=metric_name,
    )
    result["n_samples"] = len(loader.dataset)
    return result


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    num_classes: int,
    args: Namespace,
) -> nn.Module:
    device = get_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_classifier(
        args.arch,
        num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def train_with_best_config(
    *,
    train_samples: list[ImageSample],
    val_samples: list[ImageSample],
    best_config: dict[str, Any],
    output_dir: Path,
    args: Namespace,
) -> dict[str, Any]:
    """Retrain on train with validation monitoring using the tuned hyperparameters."""
    class_names: list[str] = args.class_names
    num_classes = len(class_names)
    learning_rate = best_config["learning_rate"]
    weight_decay = best_config["weight_decay"]
    class_weight_mode = best_config["class_weight"]

    print(
        f"Final training on {TRAIN_SPLIT} ({len(train_samples):,} tiles) with "
        f"{TUNE_SPLIT} monitoring ({len(val_samples):,} tiles, up to {args.epochs} epochs): "
        f"lr={learning_rate:g}, weight_decay={weight_decay:g}, "
        f"class_weight={class_weight_mode}",
        flush=True,
    )

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
    checkpoint_dir = final_checkpoint_dir(output_dir, args.arch)
    checkpoint_path = best_checkpoint_path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hyperparams = {
        "arch": args.arch,
        "pretrained": args.pretrained,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "class_weight": class_weight_mode,
        "sampler": args.sampler,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "metric": args.metric,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "seed": args.seed,
    }
    final_train_t0 = time.perf_counter()
    final_training = train_one_run(
        model,
        train_loader,
        val_loader,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        hyperparams=hyperparams,
        epochs=args.epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights,
        metric_name=args.metric,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    final_training["final_train_seconds"] = time.perf_counter() - final_train_t0
    print(f"Final training completed in {final_training['final_train_seconds']:.1f}s", flush=True)
    return final_training


def evaluate_on_test(
    *,
    test_samples: list[ImageSample],
    checkpoint_path: Path,
    args: Namespace,
) -> dict[str, Any]:
    """Load the best validation checkpoint and evaluate once on hold-out."""
    class_names: list[str] = args.class_names
    if not test_samples:
        print(f"No {TEST_SPLIT} samples; skipping hold-out evaluation.", flush=True)
        return {
            "test_metrics": {},
            "per_label": {},
            "label_support": {},
            "confusion_matrix": {},
            "n_samples": 0,
        }

    print(
        f"Evaluating on {TEST_SPLIT} ({len(test_samples):,} tiles) "
        f"from checkpoint {checkpoint_path.name}...",
        flush=True,
    )
    pin_memory = get_torch_device().type == "cuda"
    test_loader = build_eval_dataloader(
        test_samples,
        transform=build_eval_transform(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model = _load_model_from_checkpoint(
        checkpoint_path,
        num_classes=len(class_names),
        args=args,
    )
    evaluation = _evaluate_model(
        model,
        test_loader,
        class_names=class_names,
        metric_name=args.metric,
    )
    print(f"{TEST_SPLIT}: {format_metrics(evaluation['metrics'])}", flush=True)
    return {
        "test_metrics": evaluation["metrics"],
        "per_label": evaluation["per_label"],
        "label_support": evaluation["label_support"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "n_samples": evaluation["n_samples"],
    }


def run_cnn_classifier(
    *,
    dataset: str,
    tile_metadata: dict[str, dict[str, Any]],
    output_dir: Path,
    args: Namespace,
) -> dict[str, Any]:
    """
    Tune hyperparameters, retrain with the winner, and evaluate on hold-out.

    Trains on the train split only (validation is for monitoring and checkpoint
    selection). Does not merge train and validation for final training.
    """
    samples = load_image_samples(tile_metadata)
    if not samples:
        raise SystemExit("No image samples with filePath available.")

    train_samples, val_samples, test_samples = split_samples(samples)
    class_names: list[str] = args.class_names
    print(
        f"Samples: {TRAIN_SPLIT}={len(train_samples):,}, "
        f"{TUNE_SPLIT}={len(val_samples):,}, {TEST_SPLIT}={len(test_samples):,}",
        flush=True,
    )

    best_config = tune_hyperparameters(
        train_samples=train_samples,
        val_samples=val_samples,
        output_dir=output_dir,
        args=args,
    )
    final_training = train_with_best_config(
        train_samples=train_samples,
        val_samples=val_samples,
        best_config=best_config,
        output_dir=output_dir,
        args=args,
    )
    test_results = evaluate_on_test(
        test_samples=test_samples,
        checkpoint_path=best_checkpoint_path(final_checkpoint_dir(output_dir, args.arch)),
        args=args,
    )

    result = {
        "dataset": dataset,
        "label_mode": "single-label",
        "n_samples": len(samples),
        "split_counts": {
            TRAIN_SPLIT: len(train_samples),
            TUNE_SPLIT: len(val_samples),
            TEST_SPLIT: len(test_samples),
        },
        "split_class_support": _split_class_support(
            train_samples,
            val_samples,
            test_samples,
            class_names,
        ),
        "best_config": best_config,
        "final_training": final_training,
        "test_results": test_results,
        "timing": {
            "tune_seconds": best_config.get("tune_seconds"),
            "final_train_seconds": final_training.get("final_train_seconds"),
        },
    }
    save_result(output_dir, args.arch, result)
    return result
