"""
Train linear probes on a deterministic percentage of the training split.

Hyperparameters are loaded per embedding model from an existing completed
linear-probe report. Validation and hold-out data are always kept in full.
"""

from __future__ import annotations

from utils.torch_device import configure_torch_device, parse_device_cli

configure_torch_device(parse_device_cli())

import argparse
from argparse import ArgumentParser, ArgumentTypeError, Namespace
import json
from pathlib import Path
from typing import Any

from config import EMBEDDINGS_TABLE_NAME, EMBEDDINGS_TABLE_SCHEMA, REPORTS_DIR, TILES_TABLE_NAME, TILES_TABLE_SCHEMA
from dataset_configs import resolve_dataset
from utils import get_table_or_create
from utils.linear_probe.classes import (
    resolve_probe_classes,
    training_class_names,
)
from utils.linear_probe.common import (
    ensure_report_dir_available,
    result_path,
    save_model_result,
    save_run_args,
    validate_output_name,
)
from utils.linear_probe.data import (
    load_embeddings_for_model,
    load_tile_metadata,
    split_samples,
)
from utils.linear_probe.single_label import filter_single_label_metadata
from utils.linear_probe.single_label_train import evaluate_probe
from utils.training_subset import select_nested_training_subset, validate_train_percentage


def _percentage(value: str) -> float:
    try:
        return validate_train_percentage(float(value))
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description=(
            "Train all linear probes on a nested percentage of training tiles, "
            "using hyperparameters selected by an existing full-data run."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Existing linear-probe report directory containing args.json.",
    )
    parser.add_argument(
        "--train-percentage",
        type=_percentage,
        required=True,
        metavar="PERCENT",
        help="Percentage of the training split to use, in (0, 100].",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="New report folder name under reports/linear-probe/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_ID",
        help="Optional embedding-model subset; default is all models in Pixeltable.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="Training-tile selection seed; default is the source run seed.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, or cuda:N.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Required source report file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in source report file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in source report file: {path}")
    return payload


def _load_source_best_config(source_dir: Path, model_id: str) -> dict[str, Any]:
    source_result = _load_json_object(result_path(source_dir, model_id))
    best = source_result.get("best_config")
    if not isinstance(best, dict):
        raise SystemExit(f"Source result for {model_id!r} has no best_config object.")
    missing = [key for key in ("learning_rate", "class_weight") if key not in best]
    if missing:
        raise SystemExit(
            f"Source best_config for {model_id!r} is missing: {', '.join(missing)}"
        )
    return {
        "learning_rate": best["learning_rate"],
        "class_weight": best["class_weight"],
    }


def _resolve_source_models(
    source_dir: Path, requested_models: list[str] | None
) -> list[str]:
    source_models: list[str] = []
    for source_result_path in sorted(source_dir.glob("*/result.json")):
        result = _load_json_object(source_result_path)
        model_id = result.get("model_id")
        if isinstance(model_id, str) and model_id:
            source_models.append(model_id)
    if not source_models:
        raise SystemExit(f"No per-model result.json files found under {source_dir}.")
    if requested_models is None:
        return source_models

    missing = sorted(set(requested_models) - set(source_models))
    if missing:
        raise SystemExit(
            "Requested models are missing from the source run: " + ", ".join(missing)
        )
    return requested_models


def main() -> None:
    cli = parse_args()
    source_dir = cli.source_dir.expanduser().resolve()
    source_args = _load_json_object(source_dir / "args.json")
    if source_args.get("label_mode") != "single-label":
        raise SystemExit("Data-scaling linear probes require a single-label source run.")

    dataset_arg = source_args.get("resolved_dataset") or source_args.get("dataset")
    if not dataset_arg:
        raise SystemExit("Source args.json does not identify a dataset.")
    try:
        dataset = resolve_dataset(dataset_arg)
        probe_classes, source_classes = resolve_probe_classes(
            dataset, source_args.get("classes")
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    source_seed = int(source_args.get("seed", 0))
    subset_seed = source_seed if cli.subset_seed is None else cli.subset_seed
    merged = dict(source_args)
    merged.update(
        {
            "source_dir": str(source_dir),
            "train_percentage": cli.train_percentage,
            "subset_seed": subset_seed,
            "output_name": cli.output_name,
            "models": cli.models,
            "seed": source_seed,
            "device": cli.device,
            "classes": probe_classes,
            "source_class_names": source_classes,
            "class_names": training_class_names(
                probe_classes, label_mode="single-label"
            ),
            "label_mode": "single-label",
        }
    )
    args = Namespace(**merged)

    try:
        output_dir = (
            REPORTS_DIR / "linear-probe" / validate_output_name(args.output_name)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    ensure_report_dir_available(output_dir)
    output_dir.mkdir(parents=True)
    save_run_args(output_dir, args, dataset=dataset)

    print(f"Dataset: {dataset}", flush=True)
    print(f"Source run: {source_dir}", flush=True)
    print(
        f"Training subset: {args.train_percentage:g}% (seed={args.subset_seed})",
        flush=True,
    )
    print(f"Output dir: {output_dir}", flush=True)

    tiles = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)
    embeddings = get_table_or_create(EMBEDDINGS_TABLE_NAME, EMBEDDINGS_TABLE_SCHEMA)
    try:
        tile_metadata = load_tile_metadata(
            tiles,
            dataset=dataset,
            source_class_names=source_classes,
            probe_class_names=probe_classes,
            split=args.split,
            train_val_split=getattr(args, "train_val_split", None),
            excluded_class_policy=args.excluded_class_policy,
            seed=args.seed,
            max_tiles=args.max_tiles,
        )
        tile_metadata, removed = filter_single_label_metadata(tile_metadata)
        tile_metadata, subset_details = select_nested_training_subset(
            tile_metadata,
            percentage=args.train_percentage,
            seed=args.subset_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        "Single-label filter removed "
        f"{removed['removed_multi_label']:,} multi-label tiles.",
        flush=True,
    )
    print(
        f"Selected {subset_details['selected_train_count']:,}/"
        f"{subset_details['full_train_count']:,} training tiles.",
        flush=True,
    )

    model_ids = _resolve_source_models(source_dir, args.models)
    print(
        f"Models from source run ({len(model_ids)}): {', '.join(model_ids)}",
        flush=True,
    )
    for model_idx, model_id in enumerate(model_ids, start=1):
        print(f"Model {model_idx}/{len(model_ids)}: {model_id}", flush=True)
        best_config = _load_source_best_config(source_dir, model_id)
        samples = load_embeddings_for_model(
            embeddings, model_id, dataset, tile_metadata
        )
        train, validation, test = split_samples(samples)
        if not train:
            raise SystemExit(f"[{model_id}] no selected training samples available.")
        test_results = evaluate_probe(
            model_id=model_id,
            train_samples=train,
            val_samples=validation,
            test_samples=test,
            best_config=best_config,
            output_dir=output_dir,
            args=args,
        )
        result = {
            "model_id": model_id,
            "skipped": False,
            "label_mode": "single-label",
            "n_samples": len(samples),
            "training_subset": subset_details,
            "source_run": str(source_dir),
            "best_config": best_config,
            "test_results": test_results,
            "timing": {
                "tune_seconds": 0.0,
                "final_train_seconds": test_results.get("final_train_seconds"),
            },
        }
        save_model_result(output_dir, model_id, result)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
