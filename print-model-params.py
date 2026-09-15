#!/usr/bin/env python3
"""
Print parameter counts and native input tile sizes for embedding models.

Walks ``utils/embedding_models.py`` and instantiates each frozen backbone the
same way ``dataset-embeddings.py`` / the linear-probe pipeline does. Also
reports the linear-probe head size (``nn.Linear(embed_dim, num_classes)``).

Usage (from repo root):

  uv run python print-model-params.py
  uv run python print-model-params.py --models MahmoodLab/UNI facebook/dinov2-base
  uv run python print-model-params.py --dataset tau --no-multi-label
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

from utils.embedding_models import EMBEDDING_MODELS, resolve_models


@dataclass
class InspectResult:
    model_id: str
    loader: str
    n_params: int | None
    input_h: int | None
    input_w: int | None
    embed_dim: int | None
    error: str | None = None
    notes: str = ""

    @property
    def input_size_label(self) -> str:
        if self.input_h is None or self.input_w is None:
            return "—"
        if self.input_h == self.input_w:
            return f"{self.input_h}×{self.input_w}"
        return f"{self.input_h}×{self.input_w}"


def _count_params(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _format_count(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000_000:
        compact = f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        compact = f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        compact = f"{n / 1_000:.1f}K"
    else:
        compact = str(n)
    return f"{n:,} ({compact})"


def _hf_auth_kwargs() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"token": token} if token else {}


def _require_hf_token(cfg: dict[str, Any], model_id: str) -> None:
    if not cfg.get("requires_hf_token"):
        return
    if _hf_auth_kwargs():
        return
    raise OSError(
        f"Model {model_id!r} is gated on Hugging Face. "
        "Set HF_TOKEN in .env or run: huggingface-cli login"
    )


def _size_from_processor(processor: Any) -> tuple[int, int] | None:
    parsed = _parse_size_value(getattr(processor, "crop_size", None))
    if parsed is not None:
        return parsed
    parsed = _parse_size_value(getattr(processor, "size", None))
    if parsed is not None:
        return parsed
    data_config = getattr(processor, "data_config", None)
    if isinstance(data_config, dict):
        parsed = _parse_size_value(data_config.get("input_size"))
        if parsed is not None:
            return parsed
    return None


def _size_from_model_config(config: Any) -> tuple[int, int] | None:
    for attr in ("img_size", "image_size"):
        val = getattr(config, attr, None)
        if isinstance(val, int):
            return (val, val)
        parsed = _parse_size_value(val)
        if parsed is not None:
            return parsed
    pretrained_cfg = getattr(config, "pretrained_cfg", None)
    if isinstance(pretrained_cfg, dict):
        parsed = _parse_size_value(pretrained_cfg.get("input_size"))
        if parsed is not None:
            return parsed
    vision = getattr(config, "vision_config", None)
    if vision is not None:
        return _size_from_model_config(vision)
    return None


def _as_size_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        try:
            mapped = value.to_dict()
        except TypeError:
            mapped = None
        if isinstance(mapped, dict):
            return mapped
    if hasattr(value, "height") or hasattr(value, "shortest_edge"):
        return {
            "height": getattr(value, "height", None),
            "width": getattr(value, "width", None),
            "shortest_edge": getattr(value, "shortest_edge", None),
        }
    return None


def _parse_size_value(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[-2]), int(value[-1]))
    mapping = _as_size_mapping(value)
    if not mapping:
        return None
    height, width = mapping.get("height"), mapping.get("width")
    if height is not None and width is not None:
        return (int(height), int(width))
    if mapping.get("shortest_edge") is not None:
        edge = int(mapping["shortest_edge"])
        return (edge, edge)
    return None


def _resolve_checkpoint(path: str) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    rooted = _PROJECT_ROOT / path
    if rooted.is_file():
        return str(rooted)
    return str(candidate)


def _inspect_clip(model_id: str, cfg: dict[str, Any]) -> InspectResult:
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModelWithProjection

    processor = CLIPImageProcessor.from_pretrained(model_id)
    clip_config = CLIPConfig.from_pretrained(model_id)
    model = CLIPVisionModelWithProjection(clip_config.vision_config)
    model.eval()
    size = _size_from_processor(processor)
    n_params = _count_params(model)
    del model
    h, w = size if size is not None else (None, None)
    return InspectResult(
        model_id=model_id,
        loader="clip",
        n_params=n_params,
        input_h=h,
        input_w=w,
        embed_dim=cfg.get("embed_dim"),
        notes="CLIP vision encoder + projection (tile embedding path)",
    )


def _inspect_hf(model_id: str, cfg: dict[str, Any]) -> InspectResult:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    _require_hf_token(cfg, model_id)
    auth = _hf_auth_kwargs()
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, **auth)
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=True, **auth
        )
        size = _size_from_processor(processor)
    except Exception:
        size = None
    if size is None:
        size = _size_from_model_config(config)
    model = AutoModel.from_config(config, trust_remote_code=True)
    model.eval()
    n_params = _count_params(model)
    del model
    h, w = size if size is not None else (None, None)
    return InspectResult(
        model_id=model_id,
        loader="hf",
        n_params=n_params,
        input_h=h,
        input_w=w,
        embed_dim=cfg.get("embed_dim"),
    )


def _inspect_timm(model_id: str, cfg: dict[str, Any]) -> InspectResult:
    import timm
    from timm.data import resolve_data_config

    from utils.pathology_udfs import _timm_kwargs_for

    _require_hf_token(cfg, model_id)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)

    hub = cfg.get("timm_hub") or f"hf-hub:{model_id}"
    kwargs = _timm_kwargs_for(model_id)
    try:
        model = timm.create_model(hub, pretrained=False, **kwargs)
    except Exception:
        model = timm.create_model(hub, pretrained=True, **kwargs)
    model.eval()
    data_config = resolve_data_config({}, model=model)
    input_size = data_config.get("input_size")
    if isinstance(input_size, (list, tuple)) and len(input_size) >= 3:
        h, w = int(input_size[-2]), int(input_size[-1])
    elif isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        h, w = int(input_size[0]), int(input_size[1])
    else:
        h = w = int(kwargs["img_size"]) if "img_size" in kwargs else None
    n_params = _count_params(model)
    del model
    return InspectResult(
        model_id=model_id,
        loader="timm",
        n_params=n_params,
        input_h=h,
        input_w=w,
        embed_dim=cfg.get("embed_dim"),
    )


def _inspect_conch(model_id: str, cfg: dict[str, Any]) -> InspectResult:
    from conch.open_clip_custom import create_model_from_pretrained

    _require_hf_token(cfg, model_id)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model, _preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path="hf_hub:MahmoodLab/conch",
        hf_auth_token=token,
        force_image_size=(224, 224),
    )
    model.eval()
    visual = getattr(model, "visual", None)
    n_visual = _count_params(visual) if visual is not None else None
    n_total = _count_params(model)
    n_params = n_visual if n_visual is not None else n_total
    notes = "CONCH visual encoder (encode_image path)"
    if n_visual is not None and n_visual != n_total:
        notes += f"; full CONCH {n_total:,}"
    del model
    return InspectResult(
        model_id=model_id,
        loader="conch",
        n_params=n_params,
        input_h=224,
        input_w=224,
        embed_dim=cfg.get("embed_dim"),
        notes=notes,
    )


def _inspect_torchscript(model_id: str, cfg: dict[str, Any]) -> InspectResult:
    import torch

    path = cfg.get("checkpoint_path") or os.environ.get("SSCD_CHECKPOINT", "")
    resolved = _resolve_checkpoint(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"SSCD checkpoint not found: {resolved}. "
            "Set SSCD_CHECKPOINT or place sscd_disc_large.torchscript.pt on disk."
        )
    model = torch.jit.load(resolved, map_location="cpu")
    model.eval()
    n_params = _count_params(model)
    del model
    return InspectResult(
        model_id=model_id,
        loader="torchscript",
        n_params=n_params,
        input_h=224,
        input_w=224,
        embed_dim=cfg.get("embed_dim"),
        notes="Resize in pathology UDF is 224×224",
    )


def _inspect_neurofm(model_id: str, cfg: dict[str, Any]) -> InspectResult:
    import torch
    from huggingface_hub import hf_hub_download

    from utils.pathology_udfs import _ensure_iec_root, _hf_auth_kwargs as _auth

    _ensure_iec_root()
    from WSU.models.neurofm_he20x_transformer import vit_large

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
    hf_repo = cfg.get("hf_repo", model_id)
    try:
        ckpt_path = hf_hub_download(hf_repo, "pytorch_model.bin", **_auth())
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and any(k.startswith("encoder.") for k in state):
            state = {
                k.replace("encoder.", "", 1): v
                for k, v in state.items()
                if k.startswith("encoder.")
            }
        model.load_state_dict(state, strict=True)
    except Exception:
        pass
    model.eval()
    n_params = _count_params(model)
    del model
    return InspectResult(
        model_id=model_id,
        loader="neurofm",
        n_params=n_params,
        input_h=224,
        input_w=224,
        embed_dim=cfg.get("embed_dim"),
    )


def _inspect_upath(model_id: str, cfg: dict[str, Any]) -> InspectResult:
    from utils.pathology_udfs import _ensure_iec_root, _load_checkpoint_state_dict

    _ensure_iec_root()
    from WSU.models.upath.UPath import UNetSwinTransformerEmbeddings

    path = cfg.get("checkpoint_path") or os.environ.get("UPATH_CHECKPOINT", "")
    model = UNetSwinTransformerEmbeddings()
    resolved = _resolve_checkpoint(path) if path else ""
    if resolved and os.path.isfile(resolved):
        model.load_state_dict(
            _load_checkpoint_state_dict(resolved), strict=False
        )
    model.eval()
    n_params = _count_params(model)
    del model
    return InspectResult(
        model_id=model_id,
        loader="upath",
        n_params=n_params,
        input_h=224,
        input_w=224,
        embed_dim=cfg.get("embed_dim"),
        notes="Resize in pathology UDF is 224×224",
    )


_LOADERS = {
    "clip": _inspect_clip,
    "hf": _inspect_hf,
    "timm": _inspect_timm,
    "conch": _inspect_conch,
    "torchscript": _inspect_torchscript,
    "neurofm": _inspect_neurofm,
    "upath": _inspect_upath,
}


def inspect_model(model_id: str, cfg: dict[str, Any]) -> InspectResult:
    loader = cfg.get("loader") or cfg.get("udf")
    inspect_fn = _LOADERS.get(loader)
    if inspect_fn is None:
        return InspectResult(
            model_id=model_id,
            loader=str(loader),
            n_params=None,
            input_h=None,
            input_w=None,
            embed_dim=cfg.get("embed_dim"),
            error=f"unsupported loader {loader!r}",
        )
    return inspect_fn(model_id, cfg)


def linear_probe_param_count(embed_dim: int, num_classes: int) -> int:
    """``nn.Linear(embed_dim, num_classes)`` with bias, as in the probe trainers."""
    return embed_dim * num_classes + num_classes


def _resolve_probe_num_classes(args: argparse.Namespace) -> tuple[int, str, list[str]] | None:
    if args.dataset is None and args.num_classes is None:
        return None
    if args.num_classes is not None:
        return args.num_classes, f"{args.num_classes} classes (--num-classes)", []

    from dataset_configs import resolve_dataset
    from utils.linear_probe.classes import (
        resolve_probe_classes,
        resolve_probe_label_mode,
        training_class_names,
    )

    dataset = resolve_dataset(args.dataset)
    probe_classes, _source = resolve_probe_classes(dataset, args.classes)
    label_mode = resolve_probe_label_mode(
        multi_label=args.multi_label,
        num_classes=len(probe_classes),
    )
    class_names = training_class_names(probe_classes, label_mode=label_mode)
    label = (
        f"{dataset} {label_mode} ({len(class_names)} classes: "
        f"{', '.join(class_names)})"
    )
    return len(class_names), label, class_names


def _print_row(result: InspectResult, *, probe_classes: int | None) -> None:
    print(f"{result.model_id}", flush=True)
    print(f"  loader:            {result.loader}", flush=True)
    if result.error:
        print(f"  error:             {result.error}", flush=True)
        return
    print(f"  FM params:         {_format_count(result.n_params)}", flush=True)
    print(f"  input tile size:   {result.input_size_label}", flush=True)
    embed = result.embed_dim
    print(f"  embed_dim:         {embed if embed is not None else '—'}", flush=True)
    if embed is not None and probe_classes is not None:
        n_probe = linear_probe_param_count(embed, probe_classes)
        print(
            f"  linear probe:      {_format_count(n_probe)}  "
            f"(nn.Linear({embed}, {probe_classes}) + bias)",
            flush=True,
        )
    elif embed is not None:
        print(
            f"  linear probe:      (embed_dim + 1) × num_classes = "
            f"{embed + 1} × C",
            flush=True,
        )
    if result.notes:
        print(f"  notes:             {result.notes}", flush=True)


def _print_summary(results: list[InspectResult], *, probe_classes: int | None) -> None:
    ok = [r for r in results if r.error is None]
    if not ok:
        return

    headers = ["Model", "FM params", "Input", "Dim"]
    if probe_classes is not None:
        headers.append("Probe params")

    rows: list[list[str]] = []
    for r in ok:
        row = [
            r.model_id,
            _format_count(r.n_params),
            r.input_size_label,
            str(r.embed_dim) if r.embed_dim is not None else "—",
        ]
        if probe_classes is not None and r.embed_dim is not None:
            row.append(_format_count(linear_probe_param_count(r.embed_dim, probe_classes)))
        elif probe_classes is not None:
            row.append("—")
        rows.append(row)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_row(headers), flush=True)
    print("  ".join("-" * w for w in widths), flush=True)
    for row in rows:
        print(fmt_row(row), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print parameter counts and native input tile sizes for each "
            "embedding model in utils/embedding_models.py (the frozen FMs "
            "used by the linear-probe pipeline)."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_ID",
        default=None,
        help="Subset of registered model ids (default: all).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "If set, also print linear-probe head params for this dataset's "
            "class count (same class resolution as dataset-linear-probe.py)."
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="CLASS",
        default=None,
        help="Probe class subset (only used with --dataset).",
    )
    parser.add_argument(
        "--multi-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match dataset-linear-probe.py (default multi-label).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        metavar="N",
        help="Override linear-probe class count instead of --dataset.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first model that fails to load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu). Inspection only needs CPU.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    from utils.torch_device import configure_torch_device

    configure_torch_device(args.device)
    try:
        model_ids = resolve_models(args.models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    probe_info = _resolve_probe_num_classes(args)
    probe_classes = probe_info[0] if probe_info else None

    print(
        f"Inspecting {len(model_ids)} embedding model(s) from "
        "utils/embedding_models.py",
        flush=True,
    )
    if probe_info:
        print(f"Linear probe head: {probe_info[1]}", flush=True)
    else:
        print(
            "Linear probe head: pass --dataset or --num-classes to print "
            "an exact parameter count (nn.Linear(embed_dim, C) + bias).",
            flush=True,
        )
    print(flush=True)

    results: list[InspectResult] = []
    for i, model_id in enumerate(model_ids, start=1):
        cfg = EMBEDDING_MODELS[model_id]
        print(f"[{i}/{len(model_ids)}] Loading {model_id} ...", flush=True)
        try:
            result = inspect_model(model_id, cfg)
        except Exception as exc:
            result = InspectResult(
                model_id=model_id,
                loader=str(cfg.get("loader") or cfg.get("udf")),
                n_params=None,
                input_h=None,
                input_w=None,
                embed_dim=cfg.get("embed_dim"),
                error=str(exc),
            )
            if args.fail_fast:
                _print_row(result, probe_classes=probe_classes)
                raise SystemExit(f"Failed on {model_id}: {exc}") from exc
        results.append(result)
        _print_row(result, probe_classes=probe_classes)
        print(flush=True)
        gc.collect()

    print("Summary", flush=True)
    print("=======", flush=True)
    _print_summary(results, probe_classes=probe_classes)

    failed = [r for r in results if r.error]
    if failed:
        print(flush=True)
        print(f"Failed ({len(failed)}):", flush=True)
        for r in failed:
            print(f"  {r.model_id}: {r.error}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
