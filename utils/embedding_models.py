"""Registered embedding models for incremental tile embedding."""

from typing import Any, Callable

# model_id -> config. Mirrors ImageEmbeddingModelComparison WSU/export/models/*.
EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    # General vision (CLIP via Pixeltable HF UDF)
    "openai/clip-vit-base-patch32": {
        "udf": "clip",
        "loader": "clip",
        "embed_dim": 512,
        "description": "OpenAI CLIP ViT-B/32 — fast default for smoke tests",
        "requires_hf_token": False,
        "label": "CLIP ViT-B/32"
    },
    "openai/clip-vit-base-patch16": {
        "udf": "clip",
        "loader": "clip",
        "embed_dim": 512,
        "description": "OpenAI CLIP ViT-B/16",
        "requires_hf_token": False,
        "label": "CLIP ViT-B/16"
    },
    "openai/clip-vit-large-patch14": {
        "udf": "clip",
        "loader": "clip",
        "embed_dim": 768,
        "description": "OpenAI CLIP ViT-L/14 — higher quality, slower",
        "requires_hf_token": False,
        "label": "CLIP ViT-L/14"
    },
    # Pathology foundation models (PyTorch/HF; TensorRT not required)
    "MahmoodLab/conch": {
        "udf": "pathology",
        "loader": "conch",
        "embed_dim": 512,
        "description": "CONCH ViT-B/16 (MahmoodLab) — requires conch package + HF_TOKEN",
        "requires_hf_token": True,
        "extra_packages": ["conch"],
        "label": "CONCH"
    },
    "MahmoodLab/UNI": {
        "udf": "pathology",
        "loader": "hf",
        "embed_dim": 1024,
        "description": "UNI pathology foundation model (ViT-L/16)",
        "requires_hf_token": True,
        "label": "UNI"
    },
    "MahmoodLab/UNI2-h": {
        "udf": "pathology",
        "loader": "timm",
        "timm_hub": "hf-hub:MahmoodLab/UNI2-h",
        "embed_dim": 1536,
        "description": "UNI2-h pathology foundation model",
        "requires_hf_token": True,
        "extra_packages": ["timm"],
        "label": "UNI2-h"
    },
    "facebook/dinov2-large": {
        "udf": "pathology",
        "loader": "hf",
        "embed_dim": 1024,
        "description": "Meta DINOv2 ViT-L/14",
        "requires_hf_token": False,
        "label": "DINOv2-L"
    },
    "facebook/dinov2-base": {
        "udf": "pathology",
        "loader": "hf",
        "embed_dim": 768,
        "description": "Meta DINOv2 ViT-B/14 — faster general-vision baseline",
        "requires_hf_token": False,
        "label": "DINOv2-B"
    },
    "owkin/phikon-v2": {
        "udf": "pathology",
        "loader": "hf",
        "embed_dim": 1024,
        "description": "Owkin Phikon v2 pathology foundation model (ViT-B)",
        "requires_hf_token": False,
        "label": "Phikon-v2"
    },
    "timm/resnet50.a1_in1k": {
        "udf": "pathology",
        "loader": "timm",
        "timm_hub": "resnet50.a1_in1k",
        "embed_dim": 2048,
        "description": "ImageNet ResNet-50 baseline (timm)",
        "requires_hf_token": False,
        "extra_packages": ["timm"],
        "label": "ResNet50 (Frozen)"
    },
    "paige-ai/Virchow": {
        "udf": "pathology",
        "loader": "timm",
        "timm_hub": "hf-hub:paige-ai/Virchow",
        "embed_dim": 2560,
        "description": "Paige Virchow pathology foundation model (CLS+Mean)",
        "requires_hf_token": True,
        "extra_packages": ["timm"],
        "label": "Virchow"
    },
    "paige-ai/Virchow2": {
        "udf": "pathology",
        "loader": "timm",
        "timm_hub": "hf-hub:paige-ai/Virchow2",
        "embed_dim": 2560,
        "description": "Paige Virchow2 pathology foundation model (CLS+Mean)",
        "requires_hf_token": True,
        "extra_packages": ["timm"],
        "label": "Virchow2"
    },
    "prov-gigapath/prov-gigapath": {
        "udf": "pathology",
        "loader": "timm",
        "timm_hub": "hf_hub:prov-gigapath/prov-gigapath",
        "embed_dim": 1536,
        "description": "Prov-GigaPath pathology foundation model",
        "requires_hf_token": True,
        "extra_packages": ["timm"],
        "label": "Prov-GigaPath"
    },
    "sscd_disc_large": {
        "udf": "pathology",
        "loader": "torchscript",
        "embed_dim": 1024,
        "checkpoint_path": "./custom_models/sscd_disc_large.torchscript.pt",
        "description": "SSCD copy-detection embedding (TorchScript checkpoint)",
        "requires_hf_token": False,
        "label": "SSCD"
    },
}

DEFAULT_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

_SUPPORTED_UDFS = frozenset({"clip", "pathology"})


def list_models() -> list[str]:
    return sorted(EMBEDDING_MODELS.keys())


def resolve_models(model_ids: list[str] | None = None) -> list[str]:
    """
    Validate and return model ids to run.

    When model_ids is None or empty, returns all registered models in definition order.
    """
    if not model_ids:
        return list(EMBEDDING_MODELS.keys())

    unknown = [
        model_id for model_id in model_ids if model_id not in EMBEDDING_MODELS
    ]
    if unknown:
        known = ", ".join(list_models())
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Known models: {known}"
        )

    missing_udf = [
        model_id
        for model_id in model_ids
        if not EMBEDDING_MODELS[model_id].get("udf")
    ]
    if missing_udf:
        raise ValueError(
            f"Model(s) missing embedding UDF config: {', '.join(missing_udf)}"
        )

    unsupported = [
        model_id
        for model_id in model_ids
        if EMBEDDING_MODELS[model_id]["udf"] not in _SUPPORTED_UDFS
    ]
    if unsupported:
        udf_names = ", ".join(sorted(_SUPPORTED_UDFS))
        raise ValueError(
            f"Model(s) with unsupported UDF: {', '.join(unsupported)}. "
            f"Supported UDFs: {udf_names}"
        )

    seen: set[str] = set()
    ordered: list[str] = []
    for model_id in model_ids:
        if model_id not in seen:
            seen.add(model_id)
            ordered.append(model_id)
    return ordered


def get_model_config(model_id: str) -> dict[str, Any]:
    if model_id not in EMBEDDING_MODELS:
        known = ", ".join(list_models())
        raise ValueError(
            f"Unknown model_id {model_id!r}. Known models: {known}"
        )
    return EMBEDDING_MODELS[model_id]


def get_embed_fn(model_id: str, cfg: dict[str, Any]) -> Callable:
    """Return a Pixeltable UDF bound to model_id."""
    udf_name = cfg["udf"]

    if udf_name == "clip":
        from pixeltable.functions.huggingface import clip

        return clip.using(model_id=model_id)

    if udf_name == "pathology":
        from utils.pathology_udfs import pathology_embed

        return pathology_embed.using(model_id=model_id)

    raise ValueError(f"Unsupported udf {udf_name!r} for model {model_id!r}")
