"""Tile Vizcarra-2023 ROIs into multilabel classification tiles.

Logic mirrors ``notebooks/vizcarra-2023-dataset-setup.ipynb``. Output layout:

  <output_dir>/
    metadata.csv
    tiles/
      <wsi_name>/
        <tile>.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

# Vizcarra ROIs are often >> PIL's default ~89MP decompression-bomb limit.
# These are trusted local dataset files, not untrusted uploads.
Image.MAX_IMAGE_PIXELS = None

CLASS_NAMES = ("pre_nft", "inft")


def _safe_path_part(value: Any) -> str:
    value = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _read_numeric_values(path: Path | str) -> list[float]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    values: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip().replace(",", " ")
        if not line:
            continue
        values.extend(float(part) for part in line.split())
    return values


def read_boundary_polygon(boundary_path: Path | str, image_size: tuple[int, int]) -> Polygon:
    """Read the ROI polygon from normalized or absolute corner coordinates."""
    width, height = image_size
    image_bounds = box(0, 0, width, height)
    values = _read_numeric_values(boundary_path)

    if len(values) < 6:
        return image_bounds
    if len(values) % 2 != 0:
        raise ValueError(f"Boundary file has an odd number of coordinates: {boundary_path}")

    coords = list(zip(values[::2], values[1::2]))
    if max(abs(value) for value in values) <= 1.0:
        coords = [(x * width, y * height) for x, y in coords]

    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    polygon = polygon.intersection(image_bounds)
    return polygon if not polygon.is_empty else image_bounds


def _box_from_label_values(
    values: Sequence[float],
    image_size: tuple[int, int],
    label_format: str = "auto",
) -> tuple[int, Any, str]:
    width, height = image_size
    cls = int(values[0])
    x1_or_cx, y1_or_cy, x2_or_w, y2_or_h = values[1:5]
    valid_formats = {"auto", "normalized_yolo", "absolute_yolo", "absolute_xyxy"}
    if label_format not in valid_formats:
        raise ValueError(f"label_format must be one of {sorted(valid_formats)}")

    is_normalized = all(0 <= value <= 1 for value in (x1_or_cx, y1_or_cy, x2_or_w, y2_or_h))
    if label_format == "normalized_yolo" or (label_format == "auto" and is_normalized):
        cx = x1_or_cx * width
        cy = y1_or_cy * height
        bw = x2_or_w * width
        bh = y2_or_h * height
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        encoding = "normalized_yolo"
    elif label_format == "absolute_xyxy" or (
        label_format == "auto" and x2_or_w > x1_or_cx and y2_or_h > y1_or_cy
    ):
        x1, y1, x2, y2 = x1_or_cx, y1_or_cy, x2_or_w, y2_or_h
        encoding = "absolute_xyxy"
    else:
        cx, cy, bw, bh = x1_or_cx, y1_or_cy, x2_or_w, y2_or_h
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        encoding = "absolute_yolo"

    geom = box(x1, y1, x2, y2).intersection(box(0, 0, width, height))
    return cls, geom, encoding


def read_label_boxes(
    label_path: Path | str | None,
    image_size: tuple[int, int],
    *,
    label_format: str = "auto",
) -> pd.DataFrame:
    """Read ROI labels as a DataFrame in source-image pixel coordinates."""
    label_path = Path(label_path) if label_path is not None else None
    rows: list[dict[str, Any]] = []

    if label_path is not None and label_path.exists() and label_path.stat().st_size > 0:
        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            line = line.strip().replace(",", " ")
            if not line:
                continue

            parts = [float(part) for part in line.split()]
            if len(parts) < 5:
                raise ValueError(f"Expected 5 label values on {label_path}:{line_number}")

            cls, geom, encoding = _box_from_label_values(parts[:5], image_size, label_format)
            if cls not in (0, 1) or geom.is_empty or geom.area == 0:
                continue

            rows.append(
                {
                    "class_id": cls,
                    "encoding": encoding,
                    "object_area": geom.area,
                    "geometry": geom,
                }
            )

    return pd.DataFrame(rows, columns=["class_id", "encoding", "object_area", "geometry"])


def _iter_geometries(geom: Any) -> Iterable[Polygon]:
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            yield from _iter_geometries(part)


def _polygon_mask_for_tile(roi_polygon: Any, tile_geom: Any, tile_size: int) -> Image.Image:
    local_roi = translate(
        roi_polygon.intersection(tile_geom),
        xoff=-tile_geom.bounds[0],
        yoff=-tile_geom.bounds[1],
    )
    mask = Image.new("L", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(mask)

    for polygon in _iter_geometries(local_roi):
        coords = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        if len(coords) >= 3:
            draw.polygon(coords, fill=255)

    return mask


def _crop_and_mask_tile(
    image: Image.Image,
    roi_polygon: Any,
    tile_geom: Any,
    source_tile_size: int,
    outside_value: tuple[int, int, int] | int,
) -> Image.Image:
    x1, y1, x2, y2 = (int(round(value)) for value in tile_geom.bounds)
    tile = Image.new(image.mode, (source_tile_size, source_tile_size), outside_value)

    crop_box = (
        max(0, x1),
        max(0, y1),
        min(image.width, x2),
        min(image.height, y2),
    )
    if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
        tile.paste(image.crop(crop_box), (crop_box[0] - x1, crop_box[1] - y1))

    mask = _polygon_mask_for_tile(roi_polygon, tile_geom, source_tile_size)
    gray = Image.new(image.mode, (source_tile_size, source_tile_size), outside_value)
    return Image.composite(tile, gray, mask)


def _tile_box_records(
    label_boxes: pd.DataFrame,
    tile_geom: Any,
    source_tile_size: int,
    output_tile_size: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    if label_boxes.empty:
        return [], [0, 0]

    scale = output_tile_size / source_tile_size
    rows: list[dict[str, Any]] = []
    labels = [0, 0]
    x0, y0, _, _ = tile_geom.bounds

    intersections = label_boxes.copy()
    intersections["intersection"] = intersections["geometry"].map(
        lambda geom: geom.intersection(tile_geom)
    )
    intersections = intersections[
        intersections["intersection"].map(lambda geom: not geom.is_empty)
    ].copy()

    for row in intersections.itertuples():
        fraction = row.intersection.area / row.object_area if row.object_area else 0.0
        included = fraction >= threshold
        if included:
            labels[int(row.class_id)] = 1

        bx1, by1, bx2, by2 = row.intersection.bounds
        rows.append(
            {
                "class_id": int(row.class_id),
                "class_name": CLASS_NAMES[int(row.class_id)],
                "fraction": float(fraction),
                "included": bool(included),
                "x1": (bx1 - x0) * scale,
                "y1": (by1 - y0) * scale,
                "x2": (bx2 - x0) * scale,
                "y2": (by2 - y0) * scale,
            }
        )

    return rows, labels


def split_train_val_by_wsi(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    wsi_col: str = "wsi_name",
    seed: int = 2023,
    train_name: str = "train",
    val_name: str = "validation",
) -> pd.DataFrame:
    """Assign a reproducible train/validation split without leaking WSIs."""
    split_df = df.copy()
    wsi_names = pd.Series(split_df[wsi_col].dropna().unique()).sample(
        frac=1,
        random_state=seed,
    )
    n_train = int(round(len(wsi_names) * train_fraction))
    train_wsis = set(wsi_names.iloc[:n_train])

    split_df["split"] = np.where(
        split_df[wsi_col].isin(train_wsis),
        train_name,
        val_name,
    )
    return split_df


def assign_roi_splits(
    rois_df: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    seed: int = 2023,
) -> pd.DataFrame:
    """
    Map Vizcarra ROI ``type`` to train / validation / hold-out.

    Rows with ``type == "test"`` become hold-out. Remaining ROIs are split at
    the WSI level into train and validation.
    """
    test_df = rois_df[rois_df["type"] == "test"].copy()
    train_val_df = rois_df[rois_df["type"] != "test"].copy()

    frames: list[pd.DataFrame] = []
    if not train_val_df.empty:
        frames.append(
            split_train_val_by_wsi(
                train_val_df,
                train_fraction=train_fraction,
                seed=seed,
            )
        )
    if not test_df.empty:
        frames.append(test_df.assign(split="hold-out"))

    if not frames:
        return rois_df.assign(split=pd.Series(dtype=str))

    return pd.concat(frames, ignore_index=True)


def tile_roi(
    roi_path: Path | str,
    label_path: Path | str | None,
    boundary_path: Path | str,
    *,
    tile_size: int = 512,
    source_magnification: float = 40.0,
    target_magnification: float = 20.0,
    object_fraction_threshold: float = 0.5,
    min_roi_fraction: float = 0.0,
    output_root: Path | str | None = None,
    split: str | None = None,
    wsi_name: str | None = None,
    case_id: str | None = None,
    dataset: str = "vizcarra-2023",
    image_format: str = "png",
    label_format: str = "auto",
    include_partial_tiles: bool = True,
    outside_value: tuple[int, int, int] = (192, 192, 192),
    return_images: bool = False,
) -> pd.DataFrame:
    """
    Tile one ROI into non-overlapping classifier tiles and multilabel targets.

    When ``output_root`` is set, tiles are written under::

        <output_root>/tiles/<wsi_name>/<tile>.<ext>

    ``tile_size`` is the output size at ``target_magnification``. For 40x ROIs
    and 20x output, the source crop is 1024 px and the saved tile is 512 px.
    """
    roi_path = Path(roi_path)
    label_path = Path(label_path) if label_path is not None else None
    boundary_path = Path(boundary_path)

    magnification_scale = source_magnification / target_magnification
    source_tile_size = int(round(tile_size * magnification_scale))
    if source_tile_size <= 0:
        raise ValueError("source_tile_size must be positive")

    image = Image.open(roi_path).convert("RGB")
    image_size = (image.width, image.height)
    roi_polygon = read_boundary_polygon(boundary_path, image_size)
    label_boxes = read_label_boxes(label_path, image_size, label_format=label_format)

    output_root = Path(output_root) if output_root is not None else None
    safe_wsi = _safe_path_part(wsi_name or roi_path.stem)
    if output_root is not None:
        tile_dir = output_root / "tiles" / safe_wsi
        tile_dir.mkdir(parents=True, exist_ok=True)
    else:
        tile_dir = None

    if include_partial_tiles:
        x_starts = range(0, image.width, source_tile_size)
        y_starts = range(0, image.height, source_tile_size)
    else:
        x_starts = range(0, max(image.width - source_tile_size + 1, 0), source_tile_size)
        y_starts = range(0, max(image.height - source_tile_size + 1, 0), source_tile_size)

    records: list[dict[str, Any]] = []
    for row_idx, y0 in enumerate(y_starts):
        for col_idx, x0 in enumerate(x_starts):
            tile_geom = box(x0, y0, x0 + source_tile_size, y0 + source_tile_size)
            roi_area = roi_polygon.intersection(tile_geom).area
            roi_fraction = roi_area / tile_geom.area if tile_geom.area else 0.0
            if roi_area == 0 or roi_fraction < min_roi_fraction:
                continue

            tile_image = _crop_and_mask_tile(
                image, roi_polygon, tile_geom, source_tile_size, outside_value
            )
            if source_tile_size != tile_size:
                tile_image = tile_image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

            tile_boxes, labels = _tile_box_records(
                label_boxes,
                tile_geom,
                source_tile_size,
                tile_size,
                object_fraction_threshold,
            )

            tile_name = f"{roi_path.stem}_r{row_idx:03d}_c{col_idx:03d}.{image_format}"
            file_path = None
            imagename = f"{safe_wsi}/{tile_name}"
            if tile_dir is not None:
                file_path = tile_dir / tile_name
                tile_image.save(file_path)
                file_path = str(file_path.resolve())

            record: dict[str, Any] = {
                "dataset": dataset,
                "imagename": imagename,
                "tileName": tile_name,
                "filePath": file_path,
                "wsi_name": wsi_name,
                "caseId": case_id,
                "labels": labels,
                CLASS_NAMES[0]: labels[0],
                CLASS_NAMES[1]: labels[1],
                "split": split,
                "roi_filename": roi_path.name,
                "label_filename": (
                    label_path.name if label_path is not None and label_path.exists() else None
                ),
                "boundary_filename": boundary_path.name,
                "tile_row": row_idx,
                "tile_col": col_idx,
                "source_x": x0,
                "source_y": y0,
                "source_tile_size": source_tile_size,
                "tile_size": tile_size,
                "roi_fraction": roi_fraction,
                "boxes": tile_boxes,
            }
            if return_images:
                record["image"] = tile_image

            records.append(record)

    return pd.DataFrame(records)


METADATA_COLUMNS = (
    "id",
    "imagename",
    *CLASS_NAMES,
    "split",
    "wsi_name",
    "caseId",
    "roi_filename",
    "label_filename",
    "boundary_filename",
    "tile_row",
    "tile_col",
    "source_x",
    "source_y",
    "source_tile_size",
    "tile_size",
    "roi_fraction",
)


def records_to_metadata(tiles_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten tile records into an abeta-style metadata table."""
    if tiles_df.empty:
        return pd.DataFrame(columns=list(METADATA_COLUMNS))

    meta = tiles_df.copy()
    meta.insert(0, "id", range(len(meta)))
    keep = [col for col in METADATA_COLUMNS if col in meta.columns]
    return meta[keep].reset_index(drop=True)
