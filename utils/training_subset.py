"""Deterministic nested subsampling of the training split."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from config import TRAIN_SPLIT


def validate_train_percentage(value: float) -> float:
    """Validate and normalize a training percentage in ``(0, 100]``."""
    percentage = float(value)
    if not 0 < percentage <= 100:
        raise ValueError(
            f"training percentage must be greater than 0 and at most 100, got {value!r}"
        )
    return percentage


def select_nested_training_subset(
    tile_metadata: dict[str, dict[str, Any]],
    *,
    percentage: float,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Keep a deterministic percentage of training tiles and all evaluation tiles.

    Training tile names are sorted before seeded shuffling, making the ordering
    independent of database row order. Selecting a prefix guarantees nesting:
    for the same metadata and seed, the 1% subset is contained in 5%, and so on.
    """
    percentage = validate_train_percentage(percentage)
    train_names = sorted(
        tile_name
        for tile_name, meta in tile_metadata.items()
        if meta["split"] == TRAIN_SPLIT
    )
    if not train_names:
        raise ValueError("cannot subsample training data: the training split is empty")

    rng = random.Random(seed)
    rng.shuffle(train_names)
    selected_count = min(
        len(train_names),
        max(1, math.ceil(len(train_names) * percentage / 100.0)),
    )
    selected_names = set(train_names[:selected_count])

    selected_metadata = {
        tile_name: meta
        for tile_name, meta in tile_metadata.items()
        if meta["split"] != TRAIN_SPLIT or tile_name in selected_names
    }
    fingerprint = hashlib.sha256(
        "\n".join(sorted(selected_names)).encode("utf-8")
    ).hexdigest()
    details = {
        "percentage": percentage,
        "seed": seed,
        "full_train_count": len(train_names),
        "selected_train_count": selected_count,
        "selected_tile_names_sha256": fingerprint,
        "selection": "sorted tile names, seeded shuffle, prefix",
    }
    return selected_metadata, details
