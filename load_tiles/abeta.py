"""
Load ABeta tiles into Pixeltable.

Expected on-disk layout under --source:

  train.csv, validation.csv, hold-out.csv
  train/<slide>/<tiles...>
  validation/<slide>/<tiles...>
  hold-out/<slide>/<tiles...>

Inserts the same Pixeltable row shape as the other dataset loaders:

  dataset, tileName, filePath, img, wsi_name, caseId, labels, split

Idempotent per (dataset, tileName): only new tiles are inserted.

Example:

  uv run python -m load_tiles abeta \\
    --source /path/to/abeta-tiles
"""

import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from config import TILES_TABLE_NAME, TILES_TABLE_SCHEMA, VALID_EXTENSIONS
from dataset_configs.abeta import CLASSES, DATASET, EXCLUDE_IF_POSITIVE

REQUIRED_CSV_COLUMNS = (*CLASSES, *EXCLUDE_IF_POSITIVE)

from utils import get_table_or_create
from utils.utils import image_paths_from_dir


def parse_args():
    parser = ArgumentParser(
        description="Load ABeta tile images into Pixeltable."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Tile root directory",
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


def _index_images_by_rel_path(source_path: Path) -> dict[str, str]:
    """
    Walk the tile tree once and map `{split}/{imagename}` → absolute file path.

    Avoids per-row is_file() / resolve() calls (80k+ syscalls on large CSVs).
    """
    rel_to_abs: dict[str, str] = {}
    for path in image_paths_from_dir(
        source_path,
        patterns=_image_glob_patterns(),
        recursive=True,
    ):
        rel = path.relative_to(source_path).as_posix()
        rel_to_abs[rel] = str(path.resolve())
    return rel_to_abs


def _wsi_name_from_tile_name(tile_name: str) -> str:
    """
    Parse WSI name from a tile filename.

    Example: NA4009-02_AB_25_32_22.jpg → NA4009-02_AB_25
    (last two underscore-separated segments are pyvips row/col).

    Negative tiles prefixed with neg_ drop that prefix from wsi_name only, e.g.
    neg_NA4009-02_AB_25_32_22.jpg → NA4009-02_AB_25
    """
    stem = Path(tile_name).stem
    if stem.startswith("neg_"):
        stem = stem[4:]
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return stem


def _rows_to_insert(
    df: pd.DataFrame,
    *,
    rel_to_abs: dict[str, str],
    existing_tile_names: set[str],
) -> list[dict]:
    """Join CSV rows to on-disk files and build insert dicts with float class scores."""
    work = df.copy()
    work["rel"] = (
        work["path_split"].astype(str) + "/" + work["imagename"].astype(str)
    )
    work["filePath"] = work["rel"].map(rel_to_abs)
    work = work[work["filePath"].notna()]

    work["tileName"] = work["imagename"].map(lambda name: Path(name).name)

    if existing_tile_names:
        work = work[~work["tileName"].isin(existing_tile_names)]

    if work.empty:
        return []

    work["img"] = work["filePath"]
    work["wsi_name"] = work["tileName"].map(_wsi_name_from_tile_name)
    work["caseId"] = work["imagename"].str.split("/").str[0]
    work["labels"] = work[CLASSES].values.tolist()
    work["dataset"] = DATASET
    work["split"] = None

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
    Idempotent load: insert only files not already in table.
    Returns (already_in_table, inserted).
    """
    source_path = Path(source_dir).resolve()

    table = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)

    # Get existing tile names in the tile table.
    rows = (
        table.select(table.tileName)
        .where(table.dataset == DATASET)
        .collect()
    )
    existing_tile_names = {r["tileName"] for r in rows if r.get("tileName")}

    print(f"Found {len(existing_tile_names)} existing tiles in the table.")

    # Load all the .csv files in the source directory.
    csv_fps = list(source_path.rglob("*.csv"))

    # Loop through each CSV file.
    dfs = []  # stack the dfs.

    for csv_fp in csv_fps:
        df = pd.read_csv(csv_fp)

        missing = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
        if missing:
            print(
                f"Skipping {csv_fp.name} because it is missing required columns: "
                f"{', '.join(missing)}."
            )
            continue

        # path_split is the on-disk folder (train/validation/hold-out), not the dataset slug.
        df["path_split"] = [csv_fp.stem] * len(df)

        dfs.append(df)

    # Concatenate the dfs.
    df = pd.concat(dfs, ignore_index=True)

    n_csv_rows = len(df)
    exclude_mask = (df["notsure"] > 0) | (df["flag"] > 0)
    n_excluded = int(exclude_mask.sum())
    df = df[~exclude_mask]

    print(f"CSV rows: {n_csv_rows}")
    print(f"Excluded {n_excluded} rows with notsure or flag > 0.")
    print(f"Rows after exclusion filter: {len(df)}")
    print("Indexing on-disk tile images (one directory walk)...")

    rel_to_abs = _index_images_by_rel_path(source_path)
    data_to_insert = _rows_to_insert(
        df,
        rel_to_abs=rel_to_abs,
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


def main():
    args = parse_args()

    print("Loading data to PixelTable.\n")

    print(f"Source: {args.source}")

    print("\nScanning and comparing to existing rows...")

    n_existing, n_inserted = load_table(
        args.source,
        batch_size=args.batch_size,
    )

    print(f"Already in table: {n_existing}")
    print(f"Inserted: {n_inserted}")


if __name__ == "__main__":
    main()
