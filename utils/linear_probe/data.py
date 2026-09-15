"""Load tile labels, paths, and embeddings for probe and classifier evaluation."""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from typing import Any, Literal

import pixeltable as pxt

from config import EMBEDDINGS_TABLE_NAME, SPLIT_NAMES, SPLIT_VALUES, TEST_SPLIT, TRAIN_SPLIT, TUNE_SPLIT
from dataset_configs import get_train_val_split
from utils.linear_probe.classes import (
    excluded_class_names,
    has_excluded_positive,
    remap_labels,
    validate_excluded_class_policy,
)
from utils.linear_probe.splits import (
    TRAIN_VAL_SPLIT_NAMES,
    compute_row_splits,
    compute_train_val_row_splits,
    format_positive_split_breakdown,
    format_ratio_label,
    split_name_breakdown,
)

SplitMode = Literal["column", "computed", "train_val"]


def binarize_labels(raw_labels: list[float] | tuple[float, ...], num_labels: int) -> list[int]:
    """Binarize a variable-length float label vector to length num_labels."""
    values = list(raw_labels) if raw_labels is not None else []
    if len(values) != num_labels:
        raise ValueError(f"expected {num_labels} label values, got {len(values)}")
    return [int((v or 0) > 0) for v in values]


def resolve_probe_models(
    embeddings_table: pxt.Table,
    dataset: str,
    model_ids: list[str] | None,
) -> list[str]:
    if model_ids:
        return model_ids

    print("Discovering embedding models in Pixeltable...", flush=True)
    t0 = time.perf_counter()
    rows = (
        embeddings_table.select(embeddings_table.model_id)
        .where(embeddings_table.dataset == dataset)
        .collect()
    )
    available = sorted({row["model_id"] for row in rows if row.get("model_id")})
    print(
        f"Found {len(available)} model(s) in {time.perf_counter() - t0:.1f}s.",
        flush=True,
    )
    if not available:
        raise SystemExit(
            f"No embeddings found in {EMBEDDINGS_TABLE_NAME!r} for dataset {dataset!r}. "
            "Run the dataset embeddings script first."
        )
    return available


def _split_counts(tile_metadata: dict[str, dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(meta["split"] for meta in tile_metadata.values()))


def _subsample_tile_metadata(
    tile_metadata: dict[str, dict[str, Any]],
    max_tiles: int,
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if len(tile_metadata) <= max_tiles:
        return tile_metadata

    by_split: dict[str, list[str]] = defaultdict(list)
    for tile_name, meta in tile_metadata.items():
        split: str = meta["split"] or "unknown"
        by_split[split].append(tile_name)

    splits = sorted(by_split)
    rng = random.Random(seed)

    if max_tiles < len(splits):
        chosen = rng.sample(list(tile_metadata), max_tiles)
        return {name: tile_metadata[name] for name in chosen}

    allocations = {split: 1 for split in splits}
    remaining = max_tiles - len(splits)
    total = len(tile_metadata)

    for split in splits:
        pool_size = len(by_split[split])
        share = int(round(remaining * pool_size / total))
        add = min(share, pool_size - allocations[split])
        allocations[split] += add
        remaining -= add

    while remaining > 0:
        progressed = False
        for split in splits:
            if remaining == 0:
                break
            if allocations[split] < len(by_split[split]):
                allocations[split] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected: list[str] = []
    for split, tile_names in by_split.items():
        selected.extend(rng.sample(tile_names, allocations[split]))

    return {name: tile_metadata[name] for name in selected}


def _resolve_split_mode(
    rows: list[dict[str, Any]],
    *,
    split: str | None,
    train_val_split: str | None,
    seed: int,
) -> tuple[SplitMode, str | None, dict[str, str], tuple[float, ...] | None]:
    """
    Decide how to assign splits.

    - ``split`` set (e.g. ``80:20:20``): group by ``wsi_name``, or ``tileName``
      if all ``wsi_name`` values are empty. Groups are packed so each class's
      positive tile count follows the ratios (a multi-label tile counts in
      every positive class). Negative-only groups fill remaining tile quota.
      This reassigns every tile, including a predefined hold-out.
    - ``train_val_split`` set (e.g. ``80:20``): keep hold-out from the split
      column; pack remaining tiles into train/validation the same way.
    - both None: use the ``split`` column on each row; error if not properly set.
    """
    if split is not None and train_val_split is not None:
        raise ValueError("cannot combine --split and --train-val-split")

    if split is not None:
        group_key, group_splits, ratios = compute_row_splits(
            rows, split=split, seed=seed
        )
        return "computed", group_key, group_splits, ratios

    if train_val_split is not None:
        group_key, group_splits, ratios = compute_train_val_row_splits(
            rows, train_val_split=train_val_split, seed=seed
        )
        return "train_val", group_key, group_splits, ratios

    splits_in_table = [row.get("split") for row in rows if row.get("tileName")]
    if not splits_in_table:
        raise ValueError("cannot use split column: no tiles found")

    invalid = [value for value in splits_in_table if value not in SPLIT_VALUES]
    if invalid:
        unique_invalid = sorted({value for value in invalid})
        raise ValueError(
            "split column is missing or has invalid values; "
            f"expected {sorted(SPLIT_VALUES)}, got examples: {unique_invalid[:5]}. "
            "Provide --split to compute splits from wsi_name/tileName, or "
            "--train-val-split to keep hold-out and pack train/validation."
        )

    return "column", None, {}, None


def _resolve_tile_split(
    *,
    row: dict[str, Any],
    split_mode: SplitMode,
    group_key: str | None,
    group_splits: dict[str, str],
) -> str | None:
    if split_mode == "column":
        value = row.get("split")
        return value if value in SPLIT_VALUES else None

    key = row.get(group_key) if group_key else None
    if not key:
        return None
    return group_splits.get(key)


def load_tile_metadata(
    tiles_table: pxt.Table,
    *,
    dataset: str,
    source_class_names: list[str],
    probe_class_names: list[str],
    split: str | None = None,
    train_val_split: str | None = None,
    excluded_class_policy: str = "remove",
    seed: int = 0,
    max_tiles: int | None = None,
    include_file_path: bool = False,
) -> dict[str, dict[str, Any]]:
    excluded_class_policy = validate_excluded_class_policy(excluded_class_policy)
    dropped_classes = excluded_class_names(source_class_names, probe_class_names)

    print(
        f"Loading tile metadata for {dataset!r} "
        f"(tileName, wsi_name, labels{', filePath' if include_file_path else ''})...",
        flush=True,
    )
    if probe_class_names != source_class_names:
        print(
            f"Probe classes ({len(probe_class_names)}): {', '.join(probe_class_names)} "
            f"(from dataset CLASSES: {', '.join(source_class_names)})",
            flush=True,
        )
        if dropped_classes:
            print(
                f"Excluded classes: {', '.join(dropped_classes)} "
                f"(policy={excluded_class_policy})",
                flush=True,
            )
    t0 = time.perf_counter()

    select_cols = [
        tiles_table.tileName,
        tiles_table.wsi_name,
        tiles_table.labels,
        tiles_table.split,
    ]
    if include_file_path:
        select_cols.append(tiles_table.filePath)

    rows = (
        tiles_table.select(*select_cols)
        .where(tiles_table.dataset == dataset)
        .collect()
    )

    if not rows:
        raise ValueError(f"no tiles found in table for dataset {dataset!r}")

    if split is None and train_val_split is None:
        train_val_split = get_train_val_split(dataset)

    split_mode, group_key, group_splits, ratios = _resolve_split_mode(
        rows, split=split, train_val_split=train_val_split, seed=seed
    )

    if split_mode in {"computed", "train_val"}:
        assert ratios is not None and group_key is not None
        group_counts = dict(Counter(group_splits.values()))
        keys = [row.get(group_key) for row in rows]
        label_vectors = [
            list(row["labels"]) if row.get("labels") is not None else []
            for row in rows
        ]
        if split_mode == "train_val":
            print(
                "Kept hold-out from table. Packed remaining into train/validation "
                f"({format_ratio_label(ratios)}, seed={seed}) by {group_key}: "
                f"{len(group_splits):,} groups — {split_name_breakdown(group_counts)}",
                flush=True,
            )
            print(
                format_positive_split_breakdown(
                    group_splits=group_splits,
                    group_keys=keys,
                    label_vectors=label_vectors,
                    class_names=source_class_names,
                    ratios=ratios,
                    split_names=TRAIN_VAL_SPLIT_NAMES,
                ),
                flush=True,
            )
            print(
                format_positive_split_breakdown(
                    group_splits=group_splits,
                    group_keys=keys,
                    label_vectors=label_vectors,
                    class_names=source_class_names,
                    ratios=None,
                    split_names=(TEST_SPLIT,),
                ),
                flush=True,
            )
        else:
            print(
                f"Computed split ({format_ratio_label(ratios)}, seed={seed}) "
                f"by {group_key}: {len(group_splits):,} groups — "
                f"{split_name_breakdown(group_counts)}",
                flush=True,
            )
            print(
                format_positive_split_breakdown(
                    group_splits=group_splits,
                    group_keys=keys,
                    label_vectors=label_vectors,
                    class_names=source_class_names,
                    ratios=ratios,
                ),
                flush=True,
            )
    else:
        print("Using split column from tiles table.", flush=True)

    tile_metadata: dict[str, dict[str, Any]] = {}
    skipped_split = 0
    skipped_excluded = 0
    skipped_file_path = 0

    for row in rows:
        tile_name = row.get("tileName")
        if not tile_name:
            continue

        raw_labels = row.get("labels")
        if raw_labels is None:
            raise ValueError(
                f"tile {tile_name!r} in dataset {dataset!r} has null labels; "
                "labels are required on every row."
            )

        raw_label_list = list(raw_labels)
        if (
            excluded_class_policy == "remove"
            and dropped_classes
            and has_excluded_positive(
                raw_label_list, source_class_names, probe_class_names
            )
        ):
            skipped_excluded += 1
            continue

        try:
            selected = remap_labels(
                raw_label_list, source_class_names, probe_class_names
            )
            labels = binarize_labels(selected, len(probe_class_names))
        except ValueError as exc:
            raise ValueError(
                f"tile {tile_name!r} in dataset {dataset!r}: {exc}"
            ) from exc

        resolved_split = _resolve_tile_split(
            row=row,
            split_mode=split_mode,
            group_key=group_key,
            group_splits=group_splits,
        )
        if resolved_split is None:
            skipped_split += 1
            continue

        entry: dict[str, Any] = {
            "split": resolved_split,
            "wsi_name": row.get("wsi_name"),
            "labels": labels,
        }
        if include_file_path:
            file_path = row.get("filePath")
            if not file_path:
                skipped_file_path += 1
                continue
            entry["filePath"] = file_path

        tile_metadata[tile_name] = entry

    if skipped_split:
        print(f"Skipped {skipped_split:,} rows with no resolved split.", flush=True)

    if skipped_excluded:
        print(
            f"Skipped {skipped_excluded:,} rows with positive excluded-class labels "
            f"(policy=remove).",
            flush=True,
        )

    if skipped_file_path:
        print(f"Skipped {skipped_file_path:,} rows with missing filePath.", flush=True)

    if not tile_metadata:
        raise ValueError(
            f"no tiles with valid splits for dataset {dataset!r}; "
            "check --split or the split column."
        )

    if max_tiles is not None:
        tile_metadata = _subsample_tile_metadata(tile_metadata, max_tiles, seed=seed)
        counts = _split_counts(tile_metadata)
        print(
            f"Subsampled to {len(tile_metadata):,} tiles (stratified, seed={seed}): "
            f"{split_name_breakdown(counts)}",
            flush=True,
        )

    counts = _split_counts(tile_metadata)
    print(
        f"Loaded {len(tile_metadata):,} tile labels in {time.perf_counter() - t0:.1f}s "
        f"({split_name_breakdown(counts)}).",
        flush=True,
    )
    return tile_metadata


def load_image_samples(
    tile_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build CNN sample dicts from tile metadata loaded with ``include_file_path=True``.

    Each sample includes tileName, split, wsi_name, labels, and filePath.
    """
    samples: list[dict[str, Any]] = []
    missing_path = 0

    for tile_name, meta in tile_metadata.items():
        file_path = meta.get("filePath")
        if not file_path:
            missing_path += 1
            continue
        samples.append(
            {
                "tileName": tile_name,
                "split": meta["split"],
                "wsi_name": meta.get("wsi_name"),
                "labels": meta["labels"],
                "filePath": file_path,
            }
        )

    if missing_path:
        raise ValueError(
            f"{missing_path:,} tile(s) in metadata are missing filePath; "
            "call load_tile_metadata(..., include_file_path=True)."
        )

    return samples


def load_embeddings_for_model(
    embeddings_table: pxt.Table,
    model_id: str,
    dataset: str,
    tile_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    allowed_tile_names = set(tile_metadata)
    print(f"  [{model_id}] loading embeddings from Pixeltable...", flush=True)
    t0 = time.perf_counter()

    predicate = (embeddings_table.dataset == dataset) & (embeddings_table.model_id == model_id)
    if len(allowed_tile_names) <= 5000:
        predicate = predicate & embeddings_table.tileName.isin(list(allowed_tile_names))

    rows = (
        embeddings_table.select(
            embeddings_table.tileName,
            embeddings_table.embedding,
        )
        .where(predicate)
        .collect()
    )

    samples: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        tile_name = row.get("tileName")
        embedding = row.get("embedding")
        if not tile_name or embedding is None:
            continue
        if tile_name not in allowed_tile_names:
            skipped += 1
            continue
        meta = tile_metadata[tile_name]
        samples.append(
            {
                "tileName": tile_name,
                "split": meta["split"],
                "wsi_name": meta["wsi_name"],
                "labels": meta["labels"],
                "embedding": embedding,
            }
        )

    elapsed = time.perf_counter() - t0
    extra = f", skipped {skipped:,} without tile metadata" if skipped else ""
    print(
        f"  [{model_id}] loaded {len(samples):,} embeddings in {elapsed:.1f}s{extra}.",
        flush=True,
    )
    return samples


def split_samples(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = [s for s in samples if s["split"] == TRAIN_SPLIT]
    tune = [s for s in samples if s["split"] == TUNE_SPLIT]
    test = [s for s in samples if s["split"] == TEST_SPLIT]
    return train, tune, test
