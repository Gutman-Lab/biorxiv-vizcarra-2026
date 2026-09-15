"""t-SNE / UMAP representation projection helpers."""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    EMBEDDINGS_TABLE_NAME,
    EMBEDDINGS_TABLE_SCHEMA,
    TEST_SPLIT,
    TILES_TABLE_NAME,
    TILES_TABLE_SCHEMA,
)
from notebooks.utils.io import safe_filename
from notebooks.utils.paths import repo_root
from notebooks.utils.tables import get_result
from utils import get_table_or_create
from utils.classification.labels import labels_to_class_index
from utils.linear_probe.data import load_embeddings_for_model, load_image_samples, load_tile_metadata
from utils.linear_probe.single_label import filter_single_label_metadata

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import normalize
except ImportError as exc:
    raise ImportError("Install scikit-learn in the notebook kernel: uv pip install scikit-learn") from exc

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


@dataclass
class ProjectionContext:
    """Selected-run state needed to load embeddings / CNN features and title plots."""

    run_args: dict[str, Any]
    cnn_run_args: dict[str, Any]
    class_names: Sequence[str]
    all_results: list[dict[str, Any]]
    leaderboard: pd.DataFrame
    cnn_run_dir: Path
    run_name: str
    save_dir: Path | None = None
    projection_root: Path | None = None

    def __post_init__(self) -> None:
        if self.projection_root is None:
            base = self.save_dir if self.save_dir is not None else repo_root() / "reports" / "analysis" / self.run_name
            self.projection_root = base / "projections"

    @property
    def index_path(self) -> Path:
        assert self.projection_root is not None
        return self.projection_root / "projection_index.csv"


_projection_metadata_cache: dict[tuple[int, bool], dict[str, dict[str, Any]]] = {}


def load_projection_tile_metadata(
    ctx: ProjectionContext,
    *,
    include_file_path: bool = False,
) -> dict[str, dict[str, Any]]:
    cache_key = (id(ctx), include_file_path)
    if cache_key in _projection_metadata_cache:
        return _projection_metadata_cache[cache_key]

    tiles_table = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)
    tile_metadata = load_tile_metadata(
        tiles_table,
        dataset=ctx.run_args["resolved_dataset"],
        source_class_names=ctx.run_args["source_class_names"],
        probe_class_names=ctx.run_args["classes"],
        split=ctx.run_args.get("split"),
        train_val_split=ctx.run_args.get("train_val_split"),
        excluded_class_policy=ctx.run_args["excluded_class_policy"],
        seed=ctx.run_args["seed"],
        include_file_path=include_file_path,
    )
    tile_metadata, _ = filter_single_label_metadata(tile_metadata)
    _projection_metadata_cache[cache_key] = tile_metadata
    return tile_metadata


def stratified_samples(
    samples: list[dict[str, Any]],
    *,
    split: str = TEST_SPLIT,
    max_per_class: int = 400,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        if sample["split"] != split:
            continue
        by_class[labels_to_class_index(sample["labels"])].append(sample)

    chosen: list[dict[str, Any]] = []
    for _, pool in sorted(by_class.items()):
        if len(pool) <= max_per_class:
            picks = np.arange(len(pool))
        else:
            picks = rng.choice(len(pool), size=max_per_class, replace=False)
        chosen.extend(pool[i] for i in picks)
    return chosen


def samples_to_embedding_arrays(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([s["embedding"] for s in samples], dtype=np.float32)
    y = np.asarray([labels_to_class_index(s["labels"]) for s in samples], dtype=int)
    return x, y


def project_features_2d(
    x: np.ndarray,
    *,
    method: Literal["tsne", "umap"] = "tsne",
    seed: int = 0,
) -> np.ndarray:
    if x.shape[0] < 3:
        raise ValueError("Need at least 3 samples for 2D projection")

    x_norm = normalize(x)
    if method == "umap":
        if not HAS_UMAP:
            raise ImportError("umap-learn is not installed")
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1)
        return reducer.fit_transform(x_norm)

    perplexity = min(30, max(2, (x_norm.shape[0] - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(x_norm)


def plot_feature_projection(
    coords: np.ndarray,
    y: np.ndarray,
    *,
    class_names: Sequence[str],
    title: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    palette = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for class_idx, color in enumerate(palette):
        mask = y == class_idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.55,
            label=class_names[class_idx],
            color=color,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(title="Class", markerscale=2, frameon=True)
    fig.tight_layout()
    return fig, ax


def load_embedding_features_for_model(
    ctx: ProjectionContext,
    model_id: str,
    *,
    split: str = TEST_SPLIT,
    max_per_class: int = 400,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings_table = get_table_or_create(EMBEDDINGS_TABLE_NAME, EMBEDDINGS_TABLE_SCHEMA)
    tile_metadata = load_projection_tile_metadata(ctx, include_file_path=False)
    samples = load_embeddings_for_model(
        embeddings_table,
        model_id,
        ctx.run_args["resolved_dataset"],
        tile_metadata,
    )
    chosen = stratified_samples(samples, split=split, max_per_class=max_per_class, seed=seed)
    return samples_to_embedding_arrays(chosen)


def load_cnn_features_for_model(
    ctx: ProjectionContext,
    model_id: str,
    *,
    split: str = TEST_SPLIT,
    max_per_class: int = 400,
    seed: int = 0,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if "torch" not in sys.modules:
        from utils.torch_device import configure_torch_device

        configure_torch_device(str(ctx.cnn_run_args.get("device", "auto")))

    import torch
    import torch.nn as nn

    from utils.cnn_classifier.data import build_eval_dataloader
    from utils.cnn_classifier.models import build_classifier
    from utils.cnn_classifier.transforms import build_eval_transform

    result = get_result(ctx.all_results, model_id)
    arch = result.get("arch") or model_id.split("/", 1)[1]
    checkpoint_path = Path(
        result.get("final_training", {}).get("best_checkpoint_path")
        or ctx.cnn_run_dir / arch / "checkpoints" / "best.pt"
    )

    tile_metadata = load_projection_tile_metadata(ctx, include_file_path=True)
    image_samples = load_image_samples(tile_metadata)
    chosen = stratified_samples(image_samples, split=split, max_per_class=max_per_class, seed=seed)
    if not chosen:
        raise ValueError(f"No image samples available for split={split!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(arch, len(ctx.class_names), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    loader = build_eval_dataloader(
        chosen,
        transform=build_eval_transform(int(ctx.cnn_run_args.get("image_size", 224))),
        batch_size=batch_size,
        num_workers=int(ctx.cnn_run_args.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            feats = model(images).detach().cpu().numpy()
            features.append(feats)
            labels.append(y.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def load_features_for_representation(
    ctx: ProjectionContext,
    model_id: str,
    *,
    split: str = TEST_SPLIT,
    max_per_class: int = 400,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if model_id.startswith("cnn/"):
        return load_cnn_features_for_model(ctx, model_id, split=split, max_per_class=max_per_class, seed=seed)
    return load_embedding_features_for_model(ctx, model_id, split=split, max_per_class=max_per_class, seed=seed)


def projection_artifact_paths(ctx: ProjectionContext, model_id: str, *, split: str, method: str) -> dict[str, Path]:
    assert ctx.projection_root is not None
    safe_id = safe_filename(model_id)
    return {
        "png": ctx.projection_root / "images" / split / method / f"{safe_id}.png",
        "data": ctx.projection_root / "data" / split / method / f"{safe_id}.npz",
    }


def projection_title(ctx: ProjectionContext, model_id: str, *, split: str, method: str) -> str:
    row = ctx.leaderboard.set_index("model_id").loc[model_id]
    score_col = "holdout_macro_f1" if split == "hold-out" else "val_macro_f1"
    score = row.get(score_col, np.nan)
    score_text = f"macro-F1={score:.3f}" if pd.notna(score) else "macro-F1=n/a"
    return f"{row['model_label']}\n{method.upper()} | {split} | {score_text}"


def save_projection_artifacts(
    ctx: ProjectionContext,
    model_id: str,
    *,
    split: str,
    method: str,
    max_per_class: int,
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    paths = projection_artifact_paths(ctx, model_id, split=split, method=method)
    png_path = paths["png"]
    data_path = paths["data"]
    needs_compute = force or not png_path.exists() or not data_path.exists()

    status = "ready"
    error = ""
    n_points: int | None = None

    if needs_compute:
        try:
            x, y = load_features_for_representation(
                ctx,
                model_id,
                split=split,
                max_per_class=max_per_class,
                seed=seed,
            )
            coords = project_features_2d(x, method=method, seed=seed)
            n_points = int(coords.shape[0])

            data_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                data_path,
                coords=coords,
                labels=y,
                class_names=np.asarray(ctx.class_names, dtype=object),
                model_id=model_id,
                split=split,
                method=method,
                seed=seed,
                max_per_class=max_per_class,
            )

            fig, _ = plot_feature_projection(
                coords,
                y,
                class_names=ctx.class_names,
                title=projection_title(ctx, model_id, split=split, method=method),
            )
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        except Exception as exc:
            status = "error"
            error = repr(exc)
            print(f"ERROR {model_id} | {split} | {method}: {error}")
    elif data_path.exists():
        try:
            with np.load(data_path, allow_pickle=True) as data:
                n_points = int(data["coords"].shape[0])
        except Exception:
            n_points = None

    if status == "ready" and not png_path.exists():
        status = "missing"
        error = "PNG file was not created"

    return {
        "model_id": model_id,
        "model_label": ctx.leaderboard.set_index("model_id").loc[model_id, "model_label"],
        "method": method,
        "split": split,
        "status": status,
        "error": error,
        "n_points": n_points,
        "max_per_class": max_per_class,
        "seed": seed,
        "png_path": str(png_path),
        "data_path": str(data_path),
    }


def precompute_projection_plots(
    ctx: ProjectionContext,
    *,
    model_ids: list[str] | tuple[str, ...] | None = None,
    splits: tuple[str, ...] = ("hold-out",),
    methods: tuple[str, ...] = ("tsne", "umap"),
    max_per_class: int = 400,
    seed: int = 0,
    force: bool = False,
) -> pd.DataFrame:
    assert ctx.projection_root is not None
    model_ids = tuple(model_ids or ctx.leaderboard["model_id"].tolist())
    rows: list[dict[str, Any]] = []
    total = len(model_ids) * len(splits) * len(methods)
    done = 0

    ctx.projection_root.mkdir(parents=True, exist_ok=True)
    print(f"Projection root: {ctx.projection_root}")
    print(f"Saving grouped images under: {ctx.projection_root / 'images'}")
    print(f"Saving grouped arrays under: {ctx.projection_root / 'data'}")

    for split in splits:
        for method in methods:
            for model_id in model_ids:
                done += 1
                print(f"[{done}/{total}] {split} | {method} | {model_id}", flush=True)
                rows.append(
                    save_projection_artifacts(
                        ctx,
                        model_id,
                        split=split,
                        method=method,
                        max_per_class=max_per_class,
                        seed=seed,
                        force=force,
                    )
                )

    manifest = pd.DataFrame(rows)
    ctx.index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(ctx.index_path, index=False)
    print(f"Saved projection manifest -> {ctx.index_path}")
    return manifest


def load_projection_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Projection manifest not found: {path}. Run the precompute cell first."
        )
    manifest = pd.read_csv(path)
    manifest["png_exists"] = manifest["png_path"].map(lambda p: Path(p).exists())
    return manifest
