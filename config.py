"""Shared Pixeltable schemas and constants used across all datasets."""

from pathlib import Path

import pixeltable as pxt

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"

TILES_TABLE_NAME = "tiles"
EMBEDDINGS_TABLE_NAME = "embeddings"

TRAIN_SPLIT = "train"
TUNE_SPLIT = "validation"
TEST_SPLIT = "hold-out"
SPLIT_NAMES = (TRAIN_SPLIT, TUNE_SPLIT, TEST_SPLIT)
SPLIT_VALUES = frozenset(SPLIT_NAMES)

TILES_TABLE_SCHEMA = {
    "schema": {
        "dataset": pxt.Required[pxt.String],
        "tileName": pxt.Required[pxt.String],
        "filePath": pxt.Required[pxt.String],
        "img": pxt.Image,
        "wsi_name": pxt.String,
        "caseId": pxt.String,
        "labels": pxt.Required[pxt.Array[(None,), pxt.Float]],
        "split": pxt.String,
        "brainRegion": pxt.String,
    },
    "primary_key": ["dataset", "tileName"],
    "description": (
        "Tile images and labels. PK (dataset, tileName). labels required; "
        "split, wsi_name, caseId, and brainRegion optional."
    ),
}

EMBEDDINGS_TABLE_SCHEMA = {
    "schema": {
        "dataset": pxt.Required[pxt.String],
        "tileName": pxt.Required[pxt.String],
        "model_id": pxt.Required[pxt.String],
        "embedding": pxt.Array[(None,), pxt.Float],
        "inf_time": pxt.Float,
    },
    "primary_key": ["dataset", "tileName", "model_id"],
    "description": "Per-tile embedding vectors keyed by (dataset, tileName, model_id).",
}

VALID_EXTENSIONS = ["png", "jpg", "jpeg", "tif", "tiff"]
