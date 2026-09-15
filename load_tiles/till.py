"""
Load TCGA TIL tiles into Pixeltable.

Expected on-disk layout under --source (TCGA-TILs root):

  images-tcga-tils-metadata.csv
  images-tcga-tils/
    <study>/
      train|val|test/
        til-positive|til-negative/
          <barcode>_<md5>.png

Metadata columns: partition, study, barcode, label, path, md5
``path`` is relative to the CSV (i.e. to --source). Partition names map to
Pixeltable splits:

  train → train
  val   → validation
  test  → hold-out

Splits are taken from the metadata CSV only (no ratio CLI). Tile paths are
joined from metadata without per-file existence checks (Zenodo layout assumed).

Inserts the same Pixeltable row shape as the other dataset loaders:

  dataset, tileName, filePath, img, wsi_name, caseId, labels, split

``labels`` is a float vector aligned with ``CLASSES`` (``til-positive`` only).
``til-negative`` tiles are stored as all-zero vectors; downstream single-label
training prepends the implicit ``negative`` class.

``wsi_name`` / ``caseId`` are the TCGA participant ``barcode`` (leakage unit).

Idempotent per (dataset, tileName): only new tiles are inserted.

Example:

  uv run python -m load_tiles till \\
    --source /path/to/tcga-tils
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from config import TILES_TABLE_NAME, TILES_TABLE_SCHEMA
from dataset_configs.till import (
    CLASSES,
    DATASET,
    METADATA_FILENAME,
    NEGATIVE_LABEL,
    PARTITION_TO_SPLIT,
)
from utils import get_table_or_create

REQUIRED_CSV_COLUMNS = ("partition", "study", "barcode", "label", "path", "md5")
ALLOWED_METADATA_LABELS = frozenset({*CLASSES, NEGATIVE_LABEL})


def parse_args():
    parser = ArgumentParser(
        description="Load TCGA TIL tile images into Pixeltable."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "TCGA-TILs root containing images-tcga-tils/ and "
            f"{METADATA_FILENAME}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Insert batch size (default: 500)",
    )
    return parser.parse_args()


def _labels_vector(label: str) -> list[float]:
    """Map metadata label to a CLASSES-aligned vector (zeros = negative)."""
    return [1.0 if name == label else 0.0 for name in CLASSES]


def _rows_to_insert(
    df: pd.DataFrame,
    *,
    source_path: Path,
    existing_tile_names: set[str],
) -> list[dict]:
    """Build Pixeltable insert dicts from metadata (paths relative to source)."""
    work = df.copy()
    work["filePath"] = work["path"].map(
        lambda p: str(source_path / str(p))
    )
    work["tileName"] = work["path"].map(lambda p: Path(str(p)).name)

    if existing_tile_names:
        work = work[~work["tileName"].isin(existing_tile_names)]

    if work.empty:
        return []

    unknown_labels = sorted(set(work["label"].astype(str)) - ALLOWED_METADATA_LABELS)
    if unknown_labels:
        raise SystemExit(
            f"Unknown label values in metadata: {', '.join(unknown_labels)}. "
            f"Expected one of: {', '.join(sorted(ALLOWED_METADATA_LABELS))}"
        )

    unknown_partitions = sorted(
        set(work["partition"].astype(str)) - set(PARTITION_TO_SPLIT)
    )
    if unknown_partitions:
        raise SystemExit(
            f"Unknown partition values in metadata: {', '.join(unknown_partitions)}. "
            f"Expected one of: {', '.join(PARTITION_TO_SPLIT)}"
        )

    work["img"] = work["filePath"]
    work["wsi_name"] = work["barcode"].astype(str)
    work["caseId"] = work["barcode"].astype(str)
    work["labels"] = work["label"].astype(str).map(_labels_vector)
    work["dataset"] = DATASET
    work["split"] = work["partition"].astype(str).map(PARTITION_TO_SPLIT)

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
    *,
    batch_size: int = 500,
) -> tuple[int, int]:
    """
    Idempotent load: insert only files not already in table for this dataset.
    Returns (already_in_table, inserted).
    """
    source_path = Path(source_dir).resolve()
    metadata_path = source_path / METADATA_FILENAME

    if not metadata_path.is_file():
        raise SystemExit(
            f"Missing required metadata CSV: {metadata_path}\n"
            f"Expected {METADATA_FILENAME} under --source."
        )

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
            f"{metadata_path.name} missing required columns: {', '.join(missing)}"
        )

    print(f"CSV rows: {len(df):,}")

    data_to_insert = _rows_to_insert(
        df,
        source_path=source_path,
        existing_tile_names=existing_tile_names,
    )
    print(f"Found {len(data_to_insert):,} new rows to insert.")

    inserted = 0
    for i in range(0, len(data_to_insert), batch_size):
        batch = data_to_insert[i : i + batch_size]
        table.insert(batch)
        inserted += len(batch)
        if inserted and (
            inserted % (batch_size * 20) == 0 or inserted == len(data_to_insert)
        ):
            print(f"  inserted {inserted:,} / {len(data_to_insert):,}", flush=True)

    if inserted:
        print(f"Inserted {inserted:,} new rows.")

    return len(existing_tile_names), inserted


def main() -> None:
    args = parse_args()

    print("Loading TCGA TIL tiles to PixelTable.\n")
    print(f"Source: {args.source}")
    print(f"Dataset slug: {DATASET}")
    print(f"Classes: {', '.join(CLASSES)} (metadata negatives → all-zero labels)")
    print("\nScanning and comparing to existing rows...")

    n_existing, n_inserted = load_table(
        args.source,
        batch_size=args.batch_size,
    )

    print(f"Already in table: {n_existing}")
    print(f"Inserted: {n_inserted}")


if __name__ == "__main__":
    main()
