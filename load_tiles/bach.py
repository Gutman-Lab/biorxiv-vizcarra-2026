"""
Load ICIAR 2018 BACH tiles into Pixeltable.

Expected on-disk layout under --source (one folder per class):

  Benign/
  InSitu/
  Invasive/
  Normal/

Assigns a reproducible train / validation / hold-out split **stratified by
class** (default ``80:20:20``, normalized to sum to 1). Inserts the same
Pixeltable row shape as the other dataset loaders:

  dataset, tileName, filePath, img, wsi_name, caseId, labels, split

``labels`` is a one-hot float vector aligned with ``CLASSES``.

Idempotent per (dataset, tileName): only new tiles are inserted.

Example:

  uv run python -m load_tiles bach \\
    --source /path/to/bach
"""

from __future__ import annotations

import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from config import (
    SPLIT_NAMES,
    TEST_SPLIT,
    TILES_TABLE_NAME,
    TILES_TABLE_SCHEMA,
    TRAIN_SPLIT,
    TUNE_SPLIT,
    VALID_EXTENSIONS,
)
from dataset_configs.bach import CLASSES, DATASET, DEFAULT_SPLIT_RATIOS
from utils import get_table_or_create
from utils.linear_probe.splits import parse_split_ratios
from utils.utils import image_paths_from_dir


def parse_args():
    parser = ArgumentParser(
        description="Load BACH histology tiles into Pixeltable (stratified splits)."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="BACH root containing Benign/, InSitu/, Invasive/, Normal/",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT_RATIOS,
        help=(
            "Train:validation:hold-out ratios stratified by class "
            f"(default: {DEFAULT_SPLIT_RATIOS}; ratios are normalized)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for stratified split shuffle (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Insert batch size (default: 500)",
    )
    return parser.parse_args()


def _image_glob_patterns() -> tuple[str, ...]:
    return tuple(f"*.{ext}" for ext in VALID_EXTENSIONS)


def _require_class_dirs(source: Path) -> None:
    missing = [name for name in CLASSES if not (source / name).is_dir()]
    if missing:
        raise SystemExit(
            "Source is missing class directories:\n  "
            + "\n  ".join(str(source / name) for name in missing)
        )


def _one_hot(class_name: str) -> list[float]:
    return [1.0 if name == class_name else 0.0 for name in CLASSES]


def _collect_images_by_class(source: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    for class_name in CLASSES:
        class_dir = source / class_name
        paths = sorted(
            path
            for path in image_paths_from_dir(
                class_dir,
                patterns=_image_glob_patterns(),
                recursive=False,
            )
            if path.name.lower() != "thumbs.db"
        )
        by_class[class_name] = paths
        print(f"  {class_name}: {len(paths):,} images", flush=True)
    return by_class


def _stratified_split_indices(
    n: int,
    *,
    ratios: tuple[float, float, float],
    seed: int,
    salt: str,
) -> list[str]:
    """
    Assign each of ``n`` items a split name.

    Uses a per-class salt so classes shuffle independently but reproducibly.
    """
    if n == 0:
        return []

    rng = random.Random(f"{seed}:{salt}")
    order = list(range(n))
    rng.shuffle(order)

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # Remainder goes to hold-out so counts always sum to n.
    n_test = n - n_train - n_val

    # Keep all three splits non-empty when possible.
    if n >= 3:
        if n_test == 0 and ratios[2] > 0 and n_train > 1:
            n_train -= 1
            n_test = 1
        if n_val == 0 and ratios[1] > 0 and n_train > 1:
            n_train -= 1
            n_val = 1

    assignments = [TRAIN_SPLIT] * n
    idx = 0
    for i in order[idx : idx + n_train]:
        assignments[i] = TRAIN_SPLIT
    idx += n_train
    for i in order[idx : idx + n_val]:
        assignments[i] = TUNE_SPLIT
    idx += n_val
    for i in order[idx:]:
        assignments[i] = TEST_SPLIT
    return assignments


def _build_rows(
    by_class: dict[str, list[Path]],
    *,
    ratios: tuple[float, float, float],
    seed: int,
    existing_tile_names: set[str],
) -> list[dict]:
    rows: list[dict] = []
    split_counts: dict[str, int] = defaultdict(int)
    class_split_counts: dict[str, dict[str, int]] = {
        cls: defaultdict(int) for cls in CLASSES
    }

    for class_name, paths in by_class.items():
        splits = _stratified_split_indices(
            len(paths),
            ratios=ratios,
            seed=seed,
            salt=class_name,
        )
        labels = _one_hot(class_name)

        for path, split in zip(paths, splits):
            tile_name = path.name
            if tile_name in existing_tile_names:
                continue

            file_path = str(path.resolve())
            rows.append(
                {
                    "dataset": DATASET,
                    "tileName": tile_name,
                    "filePath": file_path,
                    "img": file_path,
                    "wsi_name": path.stem,
                    "caseId": class_name,
                    "labels": labels,
                    "split": split,
                }
            )
            split_counts[split] += 1
            class_split_counts[class_name][split] += 1

    print("Stratified split counts (new rows only):", flush=True)
    print(
        "  overall: "
        + ", ".join(f"{name}={split_counts[name]:,}" for name in SPLIT_NAMES),
        flush=True,
    )
    for class_name in CLASSES:
        counts = class_split_counts[class_name]
        print(
            f"  {class_name}: "
            + ", ".join(f"{name}={counts[name]:,}" for name in SPLIT_NAMES),
            flush=True,
        )
    return rows


def load_table(
    source_dir: str | Path,
    *,
    split: str = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
    batch_size: int = 500,
) -> tuple[int, int]:
    """
    Idempotent load: insert only files not already in table for this dataset.
    Returns (already_in_table, inserted).
    """
    source_path = Path(source_dir).resolve()
    _require_class_dirs(source_path)
    ratios = parse_split_ratios(split)

    table = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)
    rows = (
        table.select(table.tileName)
        .where(table.dataset == DATASET)
        .collect()
    )
    existing_tile_names = {r["tileName"] for r in rows if r.get("tileName")}
    print(f"Found {len(existing_tile_names)} existing tiles for {DATASET!r}.")

    print("Scanning class directories...", flush=True)
    by_class = _collect_images_by_class(source_path)
    total_images = sum(len(paths) for paths in by_class.values())
    if total_images == 0:
        raise SystemExit(f"No images found under {source_path}")

    print(
        f"Total images: {total_images:,} | "
        f"split ratios (normalized): "
        f"{ratios[0]:.3f}:{ratios[1]:.3f}:{ratios[2]:.3f} "
        f"(from {split!r}, seed={seed})",
        flush=True,
    )

    data_to_insert = _build_rows(
        by_class,
        ratios=ratios,
        seed=seed,
        existing_tile_names=existing_tile_names,
    )
    print(f"Found {len(data_to_insert)} new rows to insert.")

    inserted = 0
    for i in range(0, len(data_to_insert), batch_size):
        batch = data_to_insert[i : i + batch_size]
        table.insert(batch)
        inserted += len(batch)

    if inserted:
        print(f"Inserted {inserted} new rows.")

    return len(existing_tile_names), inserted


def main() -> None:
    args = parse_args()

    print("Loading BACH tiles to PixelTable.\n")
    print(f"Source: {args.source}")
    print(f"Dataset slug: {DATASET}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("\nScanning and comparing to existing rows...")

    n_existing, n_inserted = load_table(
        args.source,
        split=args.split,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print(f"Already in table: {n_existing}")
    print(f"Inserted: {n_inserted}")


if __name__ == "__main__":
    main()
