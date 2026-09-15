"""
Load Vizcarra-2023 tau tiles into Pixeltable.

Run after ``vizcarra-2023-create-tiles.py``. Expected on-disk layout under --source:

  metadata.csv
  tiles/
    <wsi_name>/
      <tile>.png

Inserts the same Pixeltable row shape as the other dataset loaders:

  dataset, tileName, filePath, img, wsi_name, caseId, labels, split

Idempotent per (dataset, tileName): only new tiles are inserted.

Example:

  uv run python -m load_tiles tau \\
    --source /path/to/vizcarra-2023-tiles
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from config import TILES_TABLE_NAME, TILES_TABLE_SCHEMA, VALID_EXTENSIONS
from dataset_configs.tau import CLASSES, DATASET
from utils import get_table_or_create
from utils.utils import image_paths_from_dir

REQUIRED_CSV_COLUMNS = ("imagename", *CLASSES, "split", "wsi_name", "caseId")


def parse_args():
    parser = ArgumentParser(
        description="Load Vizcarra-2023 tau tile images into Pixeltable."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Tile dataset root from vizcarra-2023-create-tiles.py",
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


def _index_tiles_by_imagename(tiles_root: Path) -> dict[str, str]:
    """
    Walk ``tiles/`` once and map ``{wsi}/{tile}`` → absolute file path.

    Matches the ``imagename`` column written by vizcarra-2023-create-tiles.py.
    """
    rel_to_abs: dict[str, str] = {}
    for path in image_paths_from_dir(
        tiles_root,
        patterns=_image_glob_patterns(),
        recursive=True,
    ):
        imagename = path.relative_to(tiles_root).as_posix()
        rel_to_abs[imagename] = str(path.resolve())
    return rel_to_abs


def _rows_to_insert(
    df: pd.DataFrame,
    *,
    imagename_to_abs: dict[str, str],
    existing_tile_names: set[str],
) -> list[dict]:
    """Join metadata rows to on-disk files and build Pixeltable insert dicts."""
    work = df.copy()
    work["filePath"] = work["imagename"].map(imagename_to_abs)
    work = work[work["filePath"].notna()].copy()

    work["tileName"] = work["imagename"].map(lambda name: Path(name).name)

    if existing_tile_names:
        work = work[~work["tileName"].isin(existing_tile_names)]

    if work.empty:
        return []

    work["img"] = work["filePath"]
    work["labels"] = work[list(CLASSES)].astype(float).values.tolist()
    work["dataset"] = DATASET
    work["split"] = work["split"].astype(str)
    work["wsi_name"] = work["wsi_name"].astype(str)
    work["caseId"] = work["caseId"].astype(str)

    return work[
        [
            "dataset",
            "tileName",
            "filePath",
            "img",
            "wsi_name",
            "caseId",
            "labels",
            "split",
        ]
    ].to_dict("records")


def load_table(
    source_dir: str | Path,
    batch_size: int = 500,
) -> tuple[int, int]:
    """
    Idempotent load: insert only files not already in table for this dataset.
    Returns (already_in_table, inserted).
    """
    source_path = Path(source_dir).resolve()
    metadata_path = source_path / "metadata.csv"
    tiles_root = source_path / "tiles"

    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata.csv under {source_path}")
    if not tiles_root.is_dir():
        raise SystemExit(f"Missing tiles/ directory under {source_path}")

    table = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)

    rows = (
        table.select(table.tileName)
        .where(table.dataset == DATASET)
        .collect()
    )
    existing_tile_names = {r["tileName"] for r in rows if r.get("tileName")}
    print(f"Found {len(existing_tile_names)} existing tiles for {DATASET!r}.")

    df = pd.read_csv(metadata_path)
    missing = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(
            f"metadata.csv missing required columns: {', '.join(missing)}"
        )

    n_csv_rows = len(df)
    print(f"CSV rows: {n_csv_rows}")
    print("Indexing on-disk tile images under tiles/...")

    imagename_to_abs = _index_tiles_by_imagename(tiles_root)
    print(f"Indexed {len(imagename_to_abs):,} image files.")

    data_to_insert = _rows_to_insert(
        df,
        imagename_to_abs=imagename_to_abs,
        existing_tile_names=existing_tile_names,
    )

    missing_files = n_csv_rows - len(
        df[df["imagename"].isin(imagename_to_abs)]
    )
    if missing_files:
        print(f"Warning: {missing_files} metadata rows have no matching tile file.")

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

    print("Loading tau tiles to PixelTable.\n")
    print(f"Source: {args.source}")
    print(f"Dataset slug: {DATASET}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("\nScanning and comparing to existing rows...")

    n_existing, n_inserted = load_table(
        args.source,
        batch_size=args.batch_size,
    )

    print(f"Already in table: {n_existing}")
    print(f"Inserted: {n_inserted}")


if __name__ == "__main__":
    main()
