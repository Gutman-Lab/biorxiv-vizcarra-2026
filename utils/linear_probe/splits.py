"""Train/validation/hold-out split parsing and assignment."""

from __future__ import annotations

import random
from argparse import ArgumentParser
from collections import defaultdict

from config import SPLIT_NAMES, SPLIT_VALUES, TEST_SPLIT, TRAIN_SPLIT, TUNE_SPLIT

TRAIN_VAL_SPLIT_NAMES = (TRAIN_SPLIT, TUNE_SPLIT)

FULL_SPLIT_HELP = (
    "Train:validation:hold-out ratios as integers, e.g. 80:20:20. "
    "Groups by wsi_name (or tileName if wsi_name is empty). "
    "Groups are packed so each class's positive tile count follows "
    "these ratios. Reassigns every tile, including a predefined hold-out. "
    "Cannot be combined with --train-val-split."
)

TRAIN_VAL_SPLIT_HELP = (
    "Train:validation ratios as integers, e.g. 80:20. Keeps hold-out from "
    "the split column and packs remaining tiles into train/validation "
    "(same grouping and class-balance packing as --split). "
    "If omitted, uses TRAIN_VAL_SPLIT from the dataset config when set. "
    "Cannot be combined with --split."
)


def add_split_cli_arguments(parser: ArgumentParser) -> None:
    """Add mutually exclusive ``--split`` and ``--train-val-split`` flags."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--split",
        type=str,
        default=None,
        metavar="RATIOS",
        help=FULL_SPLIT_HELP,
    )
    group.add_argument(
        "--train-val-split",
        dest="train_val_split",
        type=str,
        default=None,
        metavar="RATIOS",
        help=TRAIN_VAL_SPLIT_HELP,
    )


def parse_ratio_string(split: str, n_parts: int) -> tuple[float, ...]:
    """Parse colon-separated ratios into ``n_parts`` values that sum to 1.0."""
    parts = split.strip().split(":")
    if n_parts == 3:
        hint = "train:val:hold-out"
    elif n_parts == 2:
        hint = "train:val"
    else:
        hint = f"{n_parts} colon-separated ratios"
    if len(parts) != n_parts:
        raise ValueError(f"split must have {hint}, got {split!r}")

    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid numeric split ratios in {split!r}") from exc

    if any(value < 0 for value in values):
        raise ValueError(f"split ratios must be non-negative, got {split!r}")

    total = sum(values)
    if total <= 0:
        raise ValueError(f"split ratios must sum to a positive value, got {split!r}")

    return tuple(value / total for value in values)


def parse_split_ratios(split: str) -> tuple[float, float, float]:
    """
    Parse a split string like ``80:20:20`` into normalized (train, val, test) ratios.

    Ratios are normalized to sum to 1.0 if they do not already.
    """
    values = parse_ratio_string(split, 3)
    return (values[0], values[1], values[2])


def parse_train_val_ratios(split: str) -> tuple[float, float]:
    """Parse ``80:20`` into normalized (train, val) ratios."""
    values = parse_ratio_string(split, 2)
    return (values[0], values[1])


def _positive_flags(labels: list[float] | tuple[float, ...], n_classes: int) -> list[int]:
    flags = [0] * n_classes
    for i, value in enumerate(labels):
        if i >= n_classes:
            break
        if (value or 0) > 0:
            flags[i] = 1
    return flags


def assign_group_splits(
    group_keys: list[str],
    *,
    seed: int,
    ratios: tuple[float, ...],
    label_vectors: list[list[float]] | None = None,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, str]:
    """
    Assign each group key to the names in ``split_names``.

    When ``label_vectors`` is aligned with ``group_keys``, groups are packed so
    each class's positive tile count (label > 0) follows ``ratios``. A
    multi-label tile counts toward every positive class. Negative-only groups
    fill remaining tile-count quota.

    Without labels, groups are shuffled and split by group count (legacy).
    """
    if len(ratios) != len(split_names):
        raise ValueError(
            f"ratios length {len(ratios)} != split_names length {len(split_names)}"
        )
    unique_keys = sorted({key for key in group_keys if key})
    if not unique_keys:
        return {}

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")

    rng = random.Random(seed)
    if not label_vectors:
        return _assign_by_group_count(
            unique_keys, ratios=ratios, rng=rng, split_names=split_names
        )

    if len(label_vectors) != len(group_keys):
        raise ValueError(
            f"label_vectors length {len(label_vectors)} != group_keys length {len(group_keys)}"
        )

    n_classes = max((len(vec) for vec in label_vectors), default=0)
    if n_classes == 0:
        return _assign_by_group_count(
            unique_keys, ratios=ratios, rng=rng, split_names=split_names
        )

    return _assign_balancing_positives(
        group_keys,
        label_vectors,
        unique_keys=unique_keys,
        ratios=ratios,
        n_classes=n_classes,
        rng=rng,
        split_names=split_names,
    )


def _group_key_for_rows(rows: list[dict]) -> str:
    if any(row.get("wsi_name") for row in rows):
        return "wsi_name"
    return "tileName"


def compute_row_splits(
    rows: list[dict],
    *,
    split: str,
    seed: int,
) -> tuple[str, dict[str, str], tuple[float, ...]]:
    """
    Group and pack rows the same way as linear probe / CNN ``--split``.

    Returns ``(group_key, group_splits, ratios)``. ``group_key`` is
    ``wsi_name`` when any row has one, otherwise ``tileName``.
    """
    ratios = parse_split_ratios(split)
    group_key = _group_key_for_rows(rows)
    keys = [row.get(group_key) for row in rows]
    label_vectors = [
        list(row["labels"]) if row.get("labels") is not None else []
        for row in rows
    ]
    group_splits = assign_group_splits(
        keys,
        seed=seed,
        ratios=ratios,
        label_vectors=label_vectors,
    )
    if not group_splits:
        raise ValueError(f"cannot assign splits: no usable {group_key!r} values")
    return group_key, group_splits, ratios


def compute_train_val_row_splits(
    rows: list[dict],
    *,
    train_val_split: str,
    seed: int,
) -> tuple[str, dict[str, str], tuple[float, ...]]:
    """
    Keep predefined hold-out; pack remaining rows into train / validation.

    Grouping matches ``compute_row_splits`` (``wsi_name`` if present, else
    ``tileName``). A group key that appears in both the pool and hold-out
    is an error.
    """
    ratios = parse_train_val_ratios(train_val_split)
    holdout_rows = [row for row in rows if row.get("split") == TEST_SPLIT]
    pool_rows = [row for row in rows if row.get("split") != TEST_SPLIT]
    if not holdout_rows:
        raise ValueError(
            "train_val_split requires a predefined hold-out in the split column; "
            "none found. Use --split for a full train:val:hold-out assignment."
        )
    if not pool_rows:
        raise ValueError(
            "train_val_split found hold-out tiles but nothing left to pack "
            "into train/validation"
        )

    invalid = sorted(
        {
            value
            for row in pool_rows
            if (value := row.get("split")) not in {None, "", *SPLIT_VALUES}
        }
    )
    if invalid:
        raise ValueError(
            "train_val_split: split column has invalid values in the train pool; "
            f"expected train, validation, or empty, got examples: {invalid[:5]}"
        )

    group_key = _group_key_for_rows(rows)
    pool_keys = [row.get(group_key) for row in pool_rows]
    holdout_keys = [row.get(group_key) for row in holdout_rows]
    overlap = {key for key in pool_keys if key} & {key for key in holdout_keys if key}
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            f"train_val_split: {len(overlap)} {group_key!r} value(s) appear in "
            f"both the train pool and hold-out, e.g. {examples}. "
            "Hold-out would leak into train/validation."
        )

    label_vectors = [
        list(row["labels"]) if row.get("labels") is not None else []
        for row in pool_rows
    ]
    group_splits = assign_group_splits(
        pool_keys,
        seed=seed,
        ratios=ratios,
        label_vectors=label_vectors,
        split_names=TRAIN_VAL_SPLIT_NAMES,
    )
    if not group_splits:
        raise ValueError(
            f"cannot assign train/validation: no usable {group_key!r} values "
            "in the train pool"
        )
    for key in holdout_keys:
        if key:
            group_splits[key] = TEST_SPLIT
    return group_key, group_splits, ratios


def _assign_by_group_count(
    unique_keys: list[str],
    *,
    ratios: tuple[float, ...],
    rng: random.Random,
    split_names: tuple[str, ...],
) -> dict[str, str]:
    shuffled = unique_keys.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    counts = [int(n * ratio) for ratio in ratios[:-1]]
    counts.append(n - sum(counts))

    split_map: dict[str, str] = {}
    idx = 0
    for name, count in zip(split_names, counts):
        for key in shuffled[idx : idx + count]:
            split_map[key] = name
        idx += count
    return split_map


def _assign_balancing_positives(
    group_keys: list[str],
    label_vectors: list[list[float]],
    *,
    unique_keys: list[str],
    ratios: tuple[float, ...],
    n_classes: int,
    rng: random.Random,
    split_names: tuple[str, ...],
) -> dict[str, str]:
    """
    Largest-first greedy pack: give each WSI to the split most behind on the
    classes this WSI contributes (weighted by remaining demand vs ratio targets).
    """
    group_n: dict[str, int] = defaultdict(int)
    group_pos: dict[str, list[int]] = {
        key: [0] * n_classes for key in unique_keys
    }
    for key, labels in zip(group_keys, label_vectors):
        if not key:
            continue
        group_n[key] += 1
        for i, flag in enumerate(_positive_flags(labels, n_classes)):
            group_pos[key][i] += flag

    total_n = sum(group_n[key] for key in unique_keys)
    total_pos = [
        sum(group_pos[key][c] for key in unique_keys) for c in range(n_classes)
    ]
    targets_n = {name: ratios[i] * total_n for i, name in enumerate(split_names)}
    targets_pos = {
        name: [ratios[i] * total_pos[c] for c in range(n_classes)]
        for i, name in enumerate(split_names)
    }

    current_n = {name: 0.0 for name in split_names}
    current_pos = {name: [0.0] * n_classes for name in split_names}

    ordered = unique_keys.copy()
    rng.shuffle(ordered)
    ordered.sort(key=lambda key: (-sum(group_pos[key]), -group_n[key]))

    split_map: dict[str, str] = {}
    for key in ordered:
        best_split = split_names[0]
        best_score: tuple[float, float] | None = None
        pos_sum = sum(group_pos[key])
        for split_name in split_names:
            if pos_sum > 0:
                pos_demand = sum(
                    (targets_pos[split_name][c] - current_pos[split_name][c])
                    * group_pos[key][c]
                    for c in range(n_classes)
                )
                n_demand = (targets_n[split_name] - current_n[split_name]) * group_n[
                    key
                ]
                score = (pos_demand, n_demand)
            else:
                n_demand = targets_n[split_name] - current_n[split_name]
                score = (n_demand, 0.0)
            if best_score is None or score > best_score:
                best_score = score
                best_split = split_name

        split_map[key] = best_split
        current_n[best_split] += group_n[key]
        for c in range(n_classes):
            current_pos[best_split][c] += group_pos[key][c]

    return split_map


def format_ratio_label(ratios: tuple[float, ...]) -> str:
    """Human-readable ratio string, e.g. ``60:20:20`` or ``80:20``."""
    return ":".join(str(int(round(r * 100))) for r in ratios)


def split_name_breakdown(counts: dict[str, int]) -> str:
    return ", ".join(f"{name}={counts.get(name, 0):,}" for name in SPLIT_NAMES)


def format_positive_split_breakdown(
    *,
    group_splits: dict[str, str],
    group_keys: list[str],
    label_vectors: list[list[float]],
    class_names: list[str],
    ratios: tuple[float, ...] | None,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> str:
    """Per-split tile counts and per-class positives vs ratio targets."""
    n_classes = len(class_names)
    named = set(split_names)
    split_n = {name: 0 for name in split_names}
    split_pos = {name: [0] * n_classes for name in split_names}
    total_pos = [0] * n_classes

    for key, labels in zip(group_keys, label_vectors):
        split_name = group_splits.get(key) if key else None
        if split_name not in named:
            continue
        flags = _positive_flags(labels, n_classes)
        for c, flag in enumerate(flags):
            total_pos[c] += flag
        split_n[split_name] += 1
        for c, flag in enumerate(flags):
            split_pos[split_name][c] += flag

    lines: list[str] = []
    for i, split_name in enumerate(split_names):
        parts = [f"{split_n[split_name]:,} tiles"]
        for c, class_name in enumerate(class_names):
            actual = split_pos[split_name][c]
            if ratios is None:
                parts.append(f"{class_name}={actual:,}")
            else:
                target = ratios[i] * total_pos[c]
                parts.append(f"{class_name}={actual:,} (target {target:.1f})")
        lines.append(f"  {split_name}: " + " | ".join(parts))
    return "\n".join(lines)
