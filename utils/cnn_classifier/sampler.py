"""Training sampler helpers for imbalanced CNN datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch.utils.data import WeightedRandomSampler

from utils.cnn_classifier.data import ImageSample, class_indices_from_samples

SamplerMode = Literal["none", "weighted"]

SAMPLER_MODES: tuple[SamplerMode, ...] = ("none", "weighted")


def validate_sampler_mode(mode: str) -> SamplerMode:
    if mode not in SAMPLER_MODES:
        supported = ", ".join(SAMPLER_MODES)
        raise ValueError(f"sampler mode must be one of: {supported}; got {mode!r}")
    return mode  # type: ignore[return-value]


def _inverse_freq_sample_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1-D tensor, got shape {tuple(labels.shape)}")
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= num_classes):
        raise ValueError(
            f"labels must be in [0, {num_classes}), "
            f"got min={labels.min().item()}, max={labels.max().item()}"
        )
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    return (1.0 / counts)[labels]


def build_train_sampler(
    labels: torch.Tensor,
    mode: str,
    num_classes: int,
) -> WeightedRandomSampler | None:
    """
    Build a train-split sampler from per-sample class indices.

    ``weighted`` uses inverse-frequency weights (``1 / n_c``) independent of
    loss class-weighting mode. Returns ``None`` for natural sampling.
    """
    validated_mode = validate_sampler_mode(mode)
    if validated_mode == "none":
        return None

    if labels.numel() == 0:
        raise ValueError("cannot build a weighted sampler from an empty label tensor")

    sample_weights = _inverse_freq_sample_weights(labels, num_classes)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_train_sampler_from_samples(
    samples: Sequence[ImageSample],
    mode: str,
    num_classes: int,
) -> WeightedRandomSampler | None:
    """Build a train-split sampler from CNN image sample dicts."""
    labels = class_indices_from_samples(samples)
    return build_train_sampler(labels, mode, num_classes)
