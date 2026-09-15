"""
Create a classification tile dataset from Vizcarra-2023 ROIs.

Reads ``rois.csv``, ``images/``, ``labels/``, and ``boundaries/`` from --source,
assigns WSI-level train/validation and hold-out (type==test) splits, tiles every
ROI, and writes:

  <output_dir>/
    metadata.csv
    tiles/
      <wsi_name>/
        <tile>.png

Example:

  uv run python vizcarra-2023-create-tiles.py \\
    --source /path/to/vizcarra-2023 \\
    --output /path/to/vizcarra-2023-tiles
"""

from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from utils.vizcarra_tiling import (
    CLASS_NAMES,
    assign_roi_splits,
    records_to_metadata,
    tile_roi,
)


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return n


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Tile Vizcarra-2023 ROIs into a classification dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Vizcarra source root containing rois.csv, images/, labels/, boundaries/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory (metadata.csv + tiles/<wsi>/...)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Output tile size at target magnification (default: 256)",
    )
    parser.add_argument(
        "--source-magnification",
        type=float,
        default=40.0,
        help="ROI image magnification (default: 40)",
    )
    parser.add_argument(
        "--target-magnification",
        type=float,
        default=40.0,
        help="Desired tile magnification (default: 40)",
    )
    parser.add_argument(
        "--object-fraction-threshold",
        type=float,
        default=0.5,
        help="Min object-area fraction for a positive class label (default: 0.5)",
    )
    parser.add_argument(
        "--min-roi-fraction",
        type=float,
        default=0.25,
        help="Drop tiles whose ROI coverage is below this fraction (default: 0.25)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="WSI-level train share among non-test ROIs (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Seed for WSI train/validation shuffle (default: 2023)",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=("png", "jpg", "jpeg"),
        help="Tile image format (default: png)",
    )
    parser.add_argument(
        "--nproc",
        type=_positive_int,
        default=32,
        help="Number of parallel ROI worker processes (default: 32)",
    )
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Optional cap on ROIs for smoke tests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def _require_source_layout(source: Path) -> None:
    required = [
        source / "rois.csv",
        source / "images",
        source / "labels",
        source / "boundaries",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Source directory is missing required paths:\n  "
            + "\n  ".join(missing)
        )


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tiles_dir = output / "tiles"
    metadata_path = output / "metadata.csv"

    if not overwrite:
        existing = []
        if tiles_dir.exists() and any(tiles_dir.iterdir()):
            existing.append(str(tiles_dir))
        if metadata_path.exists():
            existing.append(str(metadata_path))
        if existing:
            raise SystemExit(
                "Output already has tile data. Pass --overwrite to continue, or "
                "choose a new --output:\n  " + "\n  ".join(existing)
            )

    tiles_dir.mkdir(parents=True, exist_ok=True)


def _tile_one_roi(job: dict[str, Any]) -> dict[str, Any]:
    """
    Worker entrypoint for one ROI.

    Returns a picklable dict: status, optional error/message, optional records.
    """
    roi_path = Path(job["roi_path"])
    if not roi_path.is_file():
        return {
            "status": "skipped",
            "message": f"missing image: {roi_path}",
            "records": None,
        }

    label_path = Path(job["label_path"])
    try:
        tiles_df = tile_roi(
            roi_path,
            label_path if label_path.is_file() else None,
            job["boundary_path"],
            tile_size=job["tile_size"],
            source_magnification=job["source_magnification"],
            target_magnification=job["target_magnification"],
            object_fraction_threshold=job["object_fraction_threshold"],
            min_roi_fraction=job["min_roi_fraction"],
            output_root=job["output_root"],
            split=job["split"],
            wsi_name=job["wsi_name"],
            case_id=job["case_id"],
            image_format=job["image_format"],
            return_images=False,
        )
    except Exception as exc:  # noqa: BLE001 — surface per-ROI failures without killing the pool
        return {
            "status": "error",
            "message": f"{roi_path.name}: {exc}",
            "records": None,
        }

    if tiles_df.empty:
        return {"status": "empty", "message": None, "records": None}

    return {
        "status": "ok",
        "message": None,
        "records": tiles_df.to_dict("records"),
    }


def _build_jobs(source: Path, output: Path, rois_df: pd.DataFrame, args: Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in rois_df.itertuples(index=False):
        filename = str(row.filename)
        stem = Path(filename).stem
        jobs.append(
            {
                "roi_path": str(source / "images" / filename),
                "label_path": str(source / "labels" / f"{stem}.txt"),
                "boundary_path": str(source / "boundaries" / f"{stem}.txt"),
                "output_root": str(output),
                "split": row.split,
                "wsi_name": row.wsi_name,
                "case_id": row.case,
                "tile_size": args.tile_size,
                "source_magnification": args.source_magnification,
                "target_magnification": args.target_magnification,
                "object_fraction_threshold": args.object_fraction_threshold,
                "min_roi_fraction": args.min_roi_fraction,
                "image_format": args.image_format,
            }
        )
    return jobs


def _consume_result(
    result: dict[str, Any],
    *,
    tile_frames: list[pd.DataFrame],
    counters: dict[str, int],
) -> None:
    if result["status"] == "skipped":
        counters["skipped"] += 1
        tqdm.write(f"  skip {result['message']}", file=None)
    elif result["status"] == "error":
        counters["errors"] += 1
        tqdm.write(f"  error {result['message']}", file=None)
    elif result["status"] == "ok" and result["records"]:
        counters["ok"] += 1
        tile_frames.append(pd.DataFrame(result["records"]))
    elif result["status"] == "empty":
        counters["empty"] += 1


def create_tiles(args: Namespace) -> Path:
    source = args.source.resolve()
    output = args.output.resolve()
    _require_source_layout(source)
    _prepare_output(output, overwrite=args.overwrite)

    rois_df = pd.read_csv(source / "rois.csv")
    required_cols = {"filename", "case", "wsi_name", "type"}
    missing_cols = required_cols - set(rois_df.columns)
    if missing_cols:
        raise SystemExit(
            f"rois.csv missing columns: {', '.join(sorted(missing_cols))}"
        )

    n_missing_filename = int(rois_df["filename"].isna().sum())
    if n_missing_filename:
        print(
            f"Dropping {n_missing_filename} ROI row(s) with missing filename.",
            flush=True,
        )
        rois_df = rois_df[rois_df["filename"].notna()].copy()
        rois_df["filename"] = rois_df["filename"].astype(str)

    if rois_df.empty:
        raise SystemExit("No ROI rows with a valid filename.")

    rois_df = assign_roi_splits(
        rois_df,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    if args.max_rois is not None:
        rois_df = rois_df.head(args.max_rois).copy()

    split_counts = rois_df["split"].value_counts().to_dict()
    nproc = min(args.nproc, len(rois_df)) if len(rois_df) else args.nproc
    print(
        f"ROIs: {len(rois_df):,} | "
        f"train={split_counts.get('train', 0):,}, "
        f"validation={split_counts.get('validation', 0):,}, "
        f"hold-out={split_counts.get('hold-out', 0):,} | "
        f"nproc={nproc}",
        flush=True,
    )
    print(
        f"Tiling at {args.target_magnification:g}x "
        f"(source {args.source_magnification:g}x, tile_size={args.tile_size}) "
        f"→ {output}",
        flush=True,
    )

    jobs = _build_jobs(source, output, rois_df, args)
    tile_frames: list[pd.DataFrame] = []
    counters = {"ok": 0, "skipped": 0, "errors": 0, "empty": 0}

    # Update on every completed ROI (not throttled).
    pbar = tqdm(
        total=len(jobs),
        desc="ROIs",
        unit="roi",
        miniters=1,
        mininterval=0,
        smoothing=0,
    )

    try:
        if nproc == 1:
            for job in jobs:
                result = _tile_one_roi(job)
                _consume_result(result, tile_frames=tile_frames, counters=counters)
                pbar.set_postfix(
                    ok=counters["ok"],
                    skip=counters["skipped"],
                    err=counters["errors"],
                    refresh=True,
                )
                pbar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=nproc) as executor:
                futures = [executor.submit(_tile_one_roi, job) for job in jobs]
                for future in as_completed(futures):
                    result = future.result()
                    _consume_result(result, tile_frames=tile_frames, counters=counters)
                    pbar.set_postfix(
                        ok=counters["ok"],
                        skip=counters["skipped"],
                        err=counters["errors"],
                        refresh=True,
                    )
                    pbar.update(1)
    finally:
        pbar.close()

    n_skipped = counters["skipped"]
    n_errors = counters["errors"]

    if not tile_frames:
        raise SystemExit("No tiles were produced.")

    all_tiles = pd.concat(tile_frames, ignore_index=True)
    metadata = records_to_metadata(all_tiles)
    metadata_path = output / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    class_pos = {name: int(metadata[name].sum()) for name in CLASS_NAMES}
    print(
        f"Wrote {len(metadata):,} tiles "
        f"({n_skipped} ROI(s) skipped, {n_errors} error(s)) → {metadata_path}",
        flush=True,
    )
    print(
        "Split tile counts: "
        + ", ".join(
            f"{split}={count:,}"
            for split, count in metadata["split"]
            .value_counts()
            .sort_index()
            .items()
        ),
        flush=True,
    )
    print(
        "Positive tile counts: "
        + ", ".join(f"{name}={count:,}" for name, count in class_pos.items()),
        flush=True,
    )
    print(f"WSIs with tiles: {metadata['wsi_name'].nunique():,}", flush=True)
    return metadata_path


def main() -> None:
    create_tiles(parse_args())


if __name__ == "__main__":
    main()
