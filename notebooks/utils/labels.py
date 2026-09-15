"""Model labeling helpers shared by analysis tables and plots."""
from __future__ import annotations

from typing import Any

MODEL_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("MahmoodLab/", "pathology_fm"),
    ("owkin/", "pathology_fm"),
    ("paige-ai/", "pathology_fm"),
    ("prov-gigapath/", "pathology_fm"),
    ("MountSinaiCompPath/", "pathology_fm"),
    ("upath", "pathology_fm"),
    ("openai/clip", "general_fm"),
    ("facebook/dinov2", "general_fm"),
    ("sscd_disc_large", "general_fm"),
    ("timm/", "frozen_cnn_embedding"),
)

SUMMARY_METRICS = ("macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall")
PER_LABEL_METRICS = ("precision", "recall", "f1")
ID_COLS = (
    "run_key",
    "run_name",
    "dataset_display",
    "modality",
    "stain",
    "task_family",
    "model_id",
    "model_label",
    "method",
    "model_family",
)

COMPARISON_GROUPS: tuple[tuple[str, str], ...] = (
    ("Best embedding probe", "embedding"),
    ("Best pathology FM", "pathology_fm"),
    ("Best general FM", "general_fm"),
    ("Frozen CNN embedding", "frozen_cnn_embedding"),
    ("CNN fine-tune", "cnn"),
)


def model_family(model_id: str) -> str:
    if model_id.startswith("cnn/"):
        return "cnn_finetune"
    for prefix, family in MODEL_FAMILY_RULES:
        if model_id.startswith(prefix) or model_id == prefix:
            return family
    return "other"


def method_label(result: dict[str, Any]) -> str:
    return "CNN fine-tune" if result.get("analysis_source") == "cnn" else "Frozen embedding + linear probe"


def model_label(result: dict[str, Any]) -> str:
    if result.get("model_label"):
        return str(result["model_label"])
    model_id = str(result["model_id"])
    return model_id.split("/")[-1]
