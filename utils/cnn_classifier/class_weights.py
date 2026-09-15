"""Class-weight helpers for imbalanced CNN training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from utils.cnn_classifier.data import ImageSample, class_indices_from_samples

ClassWeightMode = Literal["none", "inverse_freq", "sqrt_inverse_freq", "effective_num"]

CLASS_WEIGHT_MODES: tuple[ClassWeightMode, ...] = (
    "none",
    "inverse_freq",
    "sqrt_inverse_freq",
    "effective_num",
)

DEFAULT_EFFECTIVE_NUM_BETA = 0.999


def validate_class_weight_mode(mode: str) -> ClassWeightMode:
    if mode not in CLASS_WEIGHT_MODES:
        supported = ", ".join(CLASS_WEIGHT_MODES)
        raise ValueError(f"class weight mode must be one of: {supported}; got {mode!r}")
    return mode  # type: ignore[return-value]


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    mean = weights.mean()
    if mean <= 0:
        raise ValueError("cannot normalize class weights with non-positive mean")
    return weights / mean


def _class_counts(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1-D tensor, got shape {tuple(labels.shape)}")
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= num_classes):
        raise ValueError(
            f"labels must be in [0, {num_classes}), "
            f"got min={labels.min().item()}, max={labels.max().item()}"
        )
    return torch.bincount(labels, minlength=num_classes).float()


def compute_class_weights(
    labels: torch.Tensor,
    mode: str,
    num_classes: int,
    *,
    beta: float = DEFAULT_EFFECTIVE_NUM_BETA,
) -> torch.Tensor | None:
    """
    Compute per-class CE weights from train-split labels.

    Modes ``inverse_freq``, ``sqrt_inverse_freq``, and ``effective_num`` are
    normalized to mean 1. Use train labels only; do not combine with weighted
    sampling in initial experiments.
    """
    validated_mode = validate_class_weight_mode(mode)
    if validated_mode == "none":
        return None

    counts = _class_counts(labels, num_classes)
    safe_counts = counts.clamp(min=1.0)
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot compute class weights from an empty label tensor")

    if validated_mode == "inverse_freq":
        weights = total / safe_counts
    elif validated_mode == "sqrt_inverse_freq":
        weights = torch.sqrt(total / safe_counts)
    else:
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, safe_counts))

    return _normalize_weights(weights)


def compute_class_weights_from_samples(
    samples: Sequence[ImageSample],
    mode: str,
    num_classes: int,
    *,
    beta: float = DEFAULT_EFFECTIVE_NUM_BETA,
) -> torch.Tensor | None:
    """Compute class weights from CNN image sample dicts (train split only)."""
    labels = class_indices_from_samples(samples)
    return compute_class_weights(labels, mode, num_classes, beta=beta)
