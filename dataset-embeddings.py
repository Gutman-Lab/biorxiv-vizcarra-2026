"""
Compute tile embeddings and store them in a separate Pixeltable.

Incremental per (dataset, tileName, model_id): re-running only embeds tiles
that are missing for the chosen dataset and model(s).
"""

# Configure the torch device seen before importing anything else.
from utils.torch_device import (
    configure_torch_device,
    parse_device_cli,
    resolve_runtime_device_label,
)

configure_torch_device(parse_device_cli())

import time
import pixeltable as pxt
from argparse import ArgumentParser
from config import (
    EMBEDDINGS_TABLE_NAME,
    EMBEDDINGS_TABLE_SCHEMA,
    TILES_TABLE_NAME,
    TILES_TABLE_SCHEMA,
)
from dataset_configs import (
    list_configured_datasets,
    list_datasets_in_table,
    resolve_dataset,
)

from utils import get_table_or_create
from utils.embedding_models import (
    EMBEDDING_MODELS,
    get_embed_fn,
    list_models,
    resolve_models,
)


def _existing_tile_names(
    embeddings_table: pxt.Table, dataset: str, model_id: str
) -> set[str]:
    rows = (
        embeddings_table.select(embeddings_table.tileName)
        .where(
            (embeddings_table.dataset == dataset)
            & (embeddings_table.model_id == model_id)
        )
        .collect()
    )
    return {r["tileName"] for r in rows if r.get("tileName")}


def get_table_tile_names(table: pxt.Table, dataset: str) -> list[str]:
    rows = (
        table.select(table.tileName)
        .where(table.dataset == dataset)
        .collect()
    )
    tile_names = [r["tileName"] for r in rows if r.get("tileName")]
    tile_names.sort()
    return tile_names


def run_model_embeddings(
    model_id: str,
    model_params: dict,
    tiles: pxt.Table,
    emb: pxt.Table,
    dataset: str,
    tile_names: list[str],
    model_idx: int,
    total_models: int,
    batch_size: int = 32,
):
    """Compute embeddings for tiles missing (dataset, tileName, model_id) rows."""
    existing = _existing_tile_names(emb, dataset, model_id)
    pending = [name for name in tile_names if name not in existing]
    total_pending = len(pending)

    if total_pending == 0:
        print(
            f"Model {model_idx}/{total_models}: no embeddings to compute for {model_id}"
        )
        return

    embed_fn = get_embed_fn(model_id, model_params)

    print(f"Model {model_idx}/{total_models}: {model_id} — warming up...")
    warmup_start = time.perf_counter()
    _ = (
        tiles.select(tiles.tileName, embedding=embed_fn(tiles.img))
        .where(
            (tiles.dataset == dataset)
            & (tiles.tileName == pending[0])
        )
        .collect()
    )
    print(
        f"Model {model_idx}/{total_models}: warmup done in "
        f"{time.perf_counter() - warmup_start:.1f}s"
    )

    run_start = time.perf_counter()
    infer_time = 0.0
    inserted = 0

    for i in range(0, total_pending, batch_size):
        batch = pending[i : i + batch_size]

        start_t = time.perf_counter()

        rows = (
            tiles.select(tiles.tileName, embedding=embed_fn(tiles.img))
            .where(
                (tiles.dataset == dataset)
                & (tiles.tileName.isin(batch))
            )
            .collect()
        )

        batch_infer = time.perf_counter() - start_t
        infer_time += batch_infer

        to_insert = [
            {
                "dataset": dataset,
                "tileName": row["tileName"],
                "model_id": model_id,
                "embedding": row["embedding"],
            }
            for row in rows
            if row.get("tileName") and row.get("embedding") is not None
        ]

        if not to_insert:
            continue

        inf_time_per_row = batch_infer / len(to_insert)
        for row in to_insert:
            row["inf_time"] = inf_time_per_row

        emb.insert(to_insert, print_stats=False)
        inserted += len(to_insert)

        pct = 100.0 * inserted / total_pending
        print(
            f"Model {model_idx}/{total_models}, "
            f"{inserted:,}/{total_pending:,} ({pct:.1f}%)",
            flush=True,
        )

    if inserted == 0:
        print(
            f"Model {model_idx}/{total_models}: no embeddings produced for {model_id}"
        )
        return

    elapsed_sec = time.perf_counter() - run_start
    tiles_per_sec = inserted / elapsed_sec if elapsed_sec > 0 else 0.0
    avg_inf_time = infer_time / inserted

    print(
        f"Model {model_idx}/{total_models}: {model_id} done — "
        f"{inserted:,} tiles in {elapsed_sec:.1f}s "
        f"({tiles_per_sec:.1f} tiles/s, infer {infer_time:.1f}s, "
        f"avg inf_time/row {avg_inf_time:.4f}s)"
    )


def run_dataset_embeddings(
    dataset: str,
    model_ids: list[str],
    tiles: pxt.Table,
    emb: pxt.Table,
    *,
    max_tiles: int | None = None,
    batch_size: int = 512,
) -> None:
    tile_names = get_table_tile_names(tiles, dataset)
    if not tile_names:
        print(f"Dataset {dataset!r}: no tiles in table — skipping.")
        return

    if max_tiles is not None:
        tile_names = tile_names[:max_tiles]

    print(f"Dataset {dataset!r}: {len(tile_names):,} tile(s) to consider.")

    total_models = len(model_ids)
    for model_idx, model_id in enumerate(model_ids, start=1):
        run_model_embeddings(
            model_id,
            EMBEDDING_MODELS[model_id],
            tiles,
            emb,
            dataset,
            tile_names,
            model_idx,
            total_models,
            batch_size=batch_size,
        )


def parse_args():
    configured = list_configured_datasets()
    known_datasets = ", ".join(sorted(configured.values()))

    parser = ArgumentParser(
        description=(
            "Compute tile embeddings into a separate Pixeltable (incremental, per dataset and model)."
        )
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help=(
            "Dataset slug or config module name "
            f"(e.g. abeta-dataset or abeta). Known: {known_datasets}"
        ),
    )
    dataset_group.add_argument(
        "--all-datasets",
        action="store_true",
        help="Embed all datasets present in the tiles table.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print configured datasets and slugs in the tiles table, then exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print registered embedding model ids, then exit.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_ID",
        default=None,
        help=(
            "One or more embedding model ids (space-separated). "
            "Default: all models in utils/embedding_models.py."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help=(
            "Tiles per inference batch (default: 512). "
            "CLIP ViT-B/32 on an L40S often fits 512–1024; large pathology models may need 32–64."
        ),
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Max tiles to embed per dataset (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Torch device: auto, cpu, cuda, or cuda:N "
            "(e.g. cuda:1 uses physical GPU 1 via CUDA_VISIBLE_DEVICES)"
        ),
    )
    return parser.parse_args()


def _resolve_cli_models(model_ids: list[str] | None) -> list[str]:
    try:
        return resolve_models(model_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_cli_datasets(args, tiles: pxt.Table) -> list[str]:
    if args.all_datasets:
        datasets = list_datasets_in_table(tiles)
        if not datasets:
            raise SystemExit("No datasets found in the tiles table.")
        return datasets

    if args.dataset is None:
        raise SystemExit(
            "Specify --dataset DATASET or --all-datasets. "
            "Use --list-datasets to see options."
        )

    try:
        return [resolve_dataset(args.dataset)]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _print_dataset_list(tiles: pxt.Table) -> None:
    configured = list_configured_datasets()
    in_table = list_datasets_in_table(tiles)

    print("Configured datasets (dataset_configs/):")
    if configured:
        for module_name, slug in sorted(configured.items(), key=lambda x: x[1]):
            print(f"  {slug}  (module: {module_name})")
    else:
        print("  (none)")

    print("\nDatasets in tiles table:")
    if in_table:
        for slug in in_table:
            print(f"  {slug}")
    else:
        print("  (none)")


def main():
    args = parse_args()

    tiles = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)

    if args.list_datasets:
        _print_dataset_list(tiles)
        return

    if args.list_models:
        for model_id in list_models():
            cfg = EMBEDDING_MODELS[model_id]
            desc = cfg.get("description", "")
            print(f"  {model_id}  — {desc}" if desc else f"  {model_id}")
        return

    model_ids = _resolve_cli_models(args.models)
    datasets = _resolve_cli_datasets(args, tiles)

    print(f"Using device: {resolve_runtime_device_label()}")
    print(f"Models ({len(model_ids)}): {', '.join(model_ids)}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")

    emb = get_table_or_create(EMBEDDINGS_TABLE_NAME, EMBEDDINGS_TABLE_SCHEMA)

    for dataset in datasets:
        run_dataset_embeddings(
            dataset,
            model_ids,
            tiles,
            emb,
            max_tiles=args.max_tiles,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
