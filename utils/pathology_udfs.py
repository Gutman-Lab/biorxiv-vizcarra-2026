"""
Pixeltable UDFs for pathology foundation model embeddings.

Loaders mirror ImageEmbeddingModelComparison WSU/export/models/* and
WSU/inference/hf_llm.py (CONCH), using PyTorch/Hugging Face inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_LOADER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LOADER_DIR.parent

try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

import PIL.Image
import pixeltable as pxt
from pixeltable.func import Batch
from pixeltable.functions.util import resolve_torch_device

_MODEL_CACHE: dict[tuple[str, str], _EmbedBundle] = {}


@dataclass
class _EmbedBundle:
    forward: Callable[[list[PIL.Image.Image]], list[list[float]]]
    embed_dim: int | None


def _device():
    explicit = os.environ.get("PIXELEMBED_TORCH_DEVICE")
    if explicit:
        return resolve_torch_device(explicit, allow_mps=False)
    return resolve_torch_device("auto", allow_mps=False)


def _pil_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _l2_normalize(vec) -> list[float]:
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )


def _hf_auth_kwargs() -> dict[str, str]:
    token = _hf_token()
    return {"token": token} if token else {}


def _require_hf_token(model_id: str) -> None:
    from utils.embedding_models import get_model_config

    cfg = get_model_config(model_id)
    if cfg.get("requires_hf_token") and not _hf_token():
        raise OSError(
            f"Model {model_id!r} is gated on Hugging Face. "
            "Request access on huggingface.co, then set HF_TOKEN in .env or run: huggingface-cli login"
        )


def _pool_timm_features(feats) -> Any:
    """Return a 2-D (batch, dim) tensor from timm model output."""
    import torch
    import torch.nn.functional as F

    if isinstance(feats, tuple):
        feats = feats[0]
    if feats.ndim == 4:
        return F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
    if feats.ndim == 3:
        # (batch, tokens, dim) — use CLS token
        return feats[:, 0]
    if feats.ndim > 2:
        return torch.flatten(feats, 1)
    return feats


def _pool_virchow_features(feats, model_id: str) -> Any:
    """
    Paige Virchow CLS+Mean embedding (2560-d): concat CLS token with mean patch tokens.
    Virchow2 has 4 register tokens after CLS — patch tokens start at index 5.
    """
    import torch

    if isinstance(feats, tuple):
        feats = feats[0]
    class_token = feats[:, 0]
    patch_start = 5 if model_id == "paige-ai/Virchow2" else 1
    patch_tokens = feats[:, patch_start:]
    return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)


def _load_hf_bundle(model_id: str, *, embed_dim: int | None) -> _EmbedBundle:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    _require_hf_token(model_id)
    auth = _hf_auth_kwargs()
    device = _device()
    processor = AutoImageProcessor.from_pretrained(model_id, **auth)
    model = AutoModel.from_pretrained(model_id, **auth)
    model.eval()
    model.to(device)

    def forward(images: list[PIL.Image.Image]) -> list[list[float]]:
        rgb = [_pil_to_rgb(img) for img in images]
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            if (
                hasattr(outputs, "pooler_output")
                and outputs.pooler_output is not None
            ):
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            feats = feats.detach().cpu().float().numpy()
        return [_l2_normalize(feats[i]) for i in range(feats.shape[0])]

    return _EmbedBundle(forward=forward, embed_dim=embed_dim)


def _timm_kwargs_for(model_id: str) -> dict[str, Any]:
    from timm.layers.mlp import SwiGLUPacked
    from torch.nn import SiLU

    if model_id == "MahmoodLab/UNI2-h":
        return {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": SwiGLUPacked,
            "act_layer": SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": False,
        }
    if model_id in ("paige-ai/Virchow", "paige-ai/Virchow2"):
        return {
            "mlp_layer": SwiGLUPacked,
            "act_layer": SiLU,
            "dynamic_img_size": False,
        }
    if model_id == "timm/resnet50.a1_in1k":
        return {"num_classes": 0, "global_pool": "avg"}
    return {"num_classes": 0, "dynamic_img_size": False}


def _load_timm_bundle(
    model_id: str, *, timm_hub: str | None, embed_dim: int | None
) -> _EmbedBundle:
    import timm
    import torch
    from timm.data import create_transform, resolve_data_config

    _require_hf_token(model_id)
    token = _hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)

    device = _device()
    hub = timm_hub or f"hf-hub:{model_id}"
    kwargs = _timm_kwargs_for(model_id)
    model = timm.create_model(hub, pretrained=True, **kwargs)
    model.eval()
    model.to(device)
    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config, is_training=False)

    pool_fn = (
        (lambda feats: _pool_virchow_features(feats, model_id))
        if model_id in ("paige-ai/Virchow", "paige-ai/Virchow2")
        else _pool_timm_features
    )

    def forward(images: list[PIL.Image.Image]) -> list[list[float]]:
        tensors = torch.stack(
            [transform(_pil_to_rgb(img)) for img in images]
        ).to(device)
        with torch.inference_mode():
            if model_id in ("paige-ai/Virchow", "paige-ai/Virchow2"):
                raw = model(tensors)
            elif hasattr(model, "forward_features"):
                raw = model.forward_features(tensors)
            else:
                raw = model(tensors)
            feats = pool_fn(raw)
            feats = feats.detach().cpu().float().numpy()
        return [_l2_normalize(feats[i]) for i in range(feats.shape[0])]

    return _EmbedBundle(forward=forward, embed_dim=embed_dim)


def _load_conch_bundle(
    model_id: str, *, embed_dim: int | None
) -> _EmbedBundle:
    """
    CONCH loader — matches ImageEmbeddingModelComparison:
      WSU/export/models/export_conch.py
      WSU/inference/hf_llm.py setup_conch_model()
    """
    import torch
    from conch.open_clip_custom import create_model_from_pretrained

    _require_hf_token(model_id)
    device = _device()
    token = _hf_token()

    # Same call pattern as export_conch.py / hf_llm.py
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path="hf_hub:MahmoodLab/conch",
        hf_auth_token=token,
        force_image_size=(224, 224),
    )
    model.eval()
    model.to(device)

    def forward(images: list[PIL.Image.Image]) -> list[list[float]]:
        tensors = torch.stack(
            [preprocess(_pil_to_rgb(img)) for img in images]
        ).to(device)
        with torch.inference_mode():
            # visual() returns (pooled, tokens) when output_tokens=True in CONCH config;
            # encode_image() unpacks and applies contrastive projection.
            feats = model.encode_image(tensors, normalize=False)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.detach().cpu().float().numpy()
        return [_l2_normalize(feats[i]) for i in range(feats.shape[0])]

    return _EmbedBundle(forward=forward, embed_dim=embed_dim)


def _ensure_iec_root() -> str:
    """Add ImageEmbeddingModelComparison to sys.path for WSU custom models (UPath)."""
    import sys

    from utils.paths import DEFAULT_IMAGE_EMBEDDING_COMPARISON_ROOT

    root = os.environ.get("IMAGE_EMBEDDING_COMPARISON_ROOT")
    if not root:
        root = str(DEFAULT_IMAGE_EMBEDDING_COMPARISON_ROOT)
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"ImageEmbeddingModelComparison not found at {root!r}. "
            "Clone it next to this repository or set IMAGE_EMBEDDING_COMPARISON_ROOT."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")


def _load_upath_bundle(
    model_id: str, *, checkpoint_path: str, embed_dim: int | None
) -> _EmbedBundle:
    import torch
    from torchvision.transforms import v2

    _ensure_iec_root()
    from WSU.models.upath.UPath import UNetSwinTransformerEmbeddings

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"UPath checkpoint not found: {checkpoint_path}. "
            "Set UPATH_CHECKPOINT or place weights at ./custom_models/upath.pt"
        )

    device = _device()
    model = UNetSwinTransformerEmbeddings()
    model.load_state_dict(
        _load_checkpoint_state_dict(checkpoint_path), strict=False
    )
    model.eval()
    model.to(device)

    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    def forward(images: list[PIL.Image.Image]) -> list[list[float]]:
        batch = torch.stack(
            [transform(_pil_to_rgb(img)) for img in images]
        ).to(device)
        with torch.inference_mode():
            patch_feats = model(batch)  # (B, 256, 1024)
            feats = patch_feats.mean(dim=1).detach().cpu().float().numpy()
        return [_l2_normalize(feats[i]) for i in range(feats.shape[0])]

    return _EmbedBundle(forward=forward, embed_dim=embed_dim)


def _load_neurofm_bundle(
    model_id: str, *, hf_repo: str, embed_dim: int | None
) -> _EmbedBundle:
    """Mount Sinai neuroFM_HE20x — ViT-L/14 neuropathology FM on Hugging Face."""
    import torch
    from huggingface_hub import hf_hub_download
    from torchvision.transforms import v2

    _ensure_iec_root()
    from WSU.models.neurofm_he20x_transformer import vit_large

    device = _device()
    auth = _hf_auth_kwargs()
    vit_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "init_values": 1.0e-05,
        "ffn_layer": "swiglufused",
        "block_chunks": 4,
        "qkv_bias": True,
        "proj_bias": True,
        "ffn_bias": True,
    }
    model = vit_large(**vit_kwargs)
    ckpt_path = hf_hub_download(hf_repo, "pytorch_model.bin", **auth)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and any(
        k.startswith("encoder.") for k in state
    ):
        state = {
            k.replace("encoder.", "", 1): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    def forward(images: list[PIL.Image.Image]) -> list[list[float]]:
        batch = torch.stack(
            [transform(_pil_to_rgb(img)) for img in images]
        ).to(device)
        with torch.inference_mode():
            feats = model(batch)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.detach().cpu().float().numpy()
        return [_l2_normalize(feats[i]) for i in range(feats.shape[0])]

    return _EmbedBundle(forward=forward, embed_dim=embed_dim)


def _load_torchscript_bundle(
    *, checkpoint_path: str, embed_dim: int | None
) -> _EmbedBundle:
    import torch
    from torchvision.transforms import v2

    device = _device()
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"SSCD checkpoint not found: {checkpoint_path}. "
            "Set SSCD_CHECKPOINT or place sscd_disc_large.torchscript.pt on disk."
        )
    model = torch.jit.load(checkpoint_path, map_location=device)
    model.eval()
    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    def forward(images: list[PIL.Image.Image]) -> list[list[float]]:
        batch = torch.stack(
            [transform(_pil_to_rgb(img)) for img in images]
        ).to(device)
        with torch.inference_mode():
            feats = model(batch)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.detach().cpu().float().numpy()
        return [_l2_normalize(feats[i]) for i in range(feats.shape[0])]

    return _EmbedBundle(forward=forward, embed_dim=embed_dim)


def _get_bundle(model_id: str, cfg: dict[str, Any]) -> _EmbedBundle:
    cache_key = (model_id, _device())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    loader = cfg["loader"]
    embed_dim = cfg.get("embed_dim")
    if loader == "hf":
        bundle = _load_hf_bundle(model_id, embed_dim=embed_dim)
    elif loader == "timm":
        bundle = _load_timm_bundle(
            model_id, timm_hub=cfg.get("timm_hub"), embed_dim=embed_dim
        )
    elif loader == "conch":
        bundle = _load_conch_bundle(model_id, embed_dim=embed_dim)
    elif loader == "upath":
        path = cfg.get("checkpoint_path") or os.environ.get(
            "UPATH_CHECKPOINT", ""
        )
        bundle = _load_upath_bundle(
            model_id, checkpoint_path=path, embed_dim=embed_dim
        )
    elif loader == "neurofm":
        bundle = _load_neurofm_bundle(
            model_id, hf_repo=cfg.get("hf_repo", model_id), embed_dim=embed_dim
        )
    elif loader == "torchscript":
        path = cfg.get("checkpoint_path") or os.environ.get(
            "SSCD_CHECKPOINT", ""
        )
        bundle = _load_torchscript_bundle(
            checkpoint_path=path, embed_dim=embed_dim
        )
    else:
        raise ValueError(
            f"Unsupported loader {loader!r} for model {model_id!r}"
        )

    _MODEL_CACHE[cache_key] = bundle
    return bundle


@pxt.udf(batch_size=32)
def pathology_embed(
    image: Batch[PIL.Image.Image], *, model_id: str
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Pathology foundation model image embeddings (UNI, CONCH, DINOv2, Virchow, etc.).

    Model config is resolved from utils.embedding_models.EMBEDDING_MODELS.
    """
    from utils.embedding_models import get_model_config

    cfg = get_model_config(model_id)
    bundle = _get_bundle(model_id, cfg)
    vectors = bundle.forward(list(image))
    return vectors
