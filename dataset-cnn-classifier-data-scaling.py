"""
Train one CNN on a deterministic percentage of the training split.

The architecture and selected hyperparameters are loaded from an existing
completed CNN report. Validation and hold-out data are always kept in full.
"""

from __future__ import annotations

from utils.torch_device import configure_torch_device, parse_device_cli

configure_torch_device(parse_device_cli())

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import json
from pathlib import Path
from typing import Any

from config import REPORTS_DIR, TEST_SPLIT, TRAIN_SPLIT, TILES_TABLE_NAME, TILES_TABLE_SCHEMA, TUNE_SPLIT
from dataset_configs import resolve_dataset
from utils import get_table_or_create
from utils.classification.labels import labels_to_class_index
from utils.cnn_classifier.common import (
    best_checkpoint_path,
    ensure_report_dir_available,
    final_checkpoint_dir,
    result_json_path,
    save_result,
    save_run_args,
    validate_output_name,
)
from utils.cnn_classifier.single_label import evaluate_on_test, train_with_best_config
from utils.linear_probe.classes import resolve_probe_classes, training_class_names
from utils.linear_probe.data import load_image_samples, load_tile_metadata, split_samples
from utils.linear_probe.single_label import filter_single_label_metadata
from utils.training_subset import select_nested_training_subset, validate_train_percentage


def _percentage(value: str) -> float:
    try:
        return validate_train_percentage(float(value))
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description=(
            "Train one CNN on a nested percentage of training tiles, using "
            "hyperparameters selected by an existing full-data run."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Existing CNN report directory containing args.json.",
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
        help="New report folder name under reports/cnn/.",
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


def _load_source_best_config(source_dir: Path, arch: str) -> dict[str, Any]:
    source_result = _load_json_object(result_json_path(source_dir, arch))
    best = source_result.get("best_config")
    if not isinstance(best, dict):
        raise SystemExit(f"Source CNN result for {arch!r} has no best_config object.")
    required = ("learning_rate", "weight_decay", "class_weight")
    missing = [key for key in required if key not in best]
    if missing:
        raise SystemExit(
            f"Source best_config for {arch!r} is missing: {', '.join(missing)}"
        )
    return {key: best[key] for key in required}


def _class_support(
    samples: list[dict[str, Any]], class_names: list[str]
) -> dict[str, int]:
    support = {name: 0 for name in class_names}
    for sample in samples:
        support[class_names[labels_to_class_index(sample["labels"])]] += 1
    return support


def main() -> None:
    cli = parse_args()
    source_dir = cli.source_dir.expanduser().resolve()
    source_args = _load_json_object(source_dir / "args.json")
    if source_args.get("label_mode") != "single-label":
        raise SystemExit("Data-scaling CNN training requires a single-label source run.")

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
            "device": cli.device,
            "classes": probe_classes,
            "source_class_names": source_classes,
            "class_names": training_class_names(
                probe_classes, label_mode="single-label"
            ),
            "label_mode": "single-label",
            "seed": source_seed,
        }
    )
    args = Namespace(**merged)

    try:
        output_dir = REPORTS_DIR / "cnn" / validate_output_name(args.output_name)
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
            include_file_path=True,
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

    samples = load_image_samples(tile_metadata)
    train, validation, test = split_samples(samples)
    if not train:
        raise SystemExit("No selected CNN training samples are available.")

    best_config = _load_source_best_config(source_dir, args.arch)
    final_training = train_with_best_config(
        train_samples=train,
        val_samples=validation,
        best_config=best_config,
        output_dir=output_dir,
        args=args,
    )
    test_results = evaluate_on_test(
        test_samples=test,
        checkpoint_path=best_checkpoint_path(
            final_checkpoint_dir(output_dir, args.arch)
        ),
        args=args,
    )
    result = {
        "dataset": dataset,
        "label_mode": "single-label",
        "n_samples": len(samples),
        "training_subset": subset_details,
        "source_run": str(source_dir),
        "split_counts": {
            TRAIN_SPLIT: len(train),
            TUNE_SPLIT: len(validation),
            TEST_SPLIT: len(test),
        },
        "split_class_support": {
            TRAIN_SPLIT: _class_support(train, args.class_names),
            TUNE_SPLIT: _class_support(validation, args.class_names),
            TEST_SPLIT: _class_support(test, args.class_names),
        },
        "best_config": best_config,
        "final_training": final_training,
        "test_results": test_results,
        "timing": {
            "tune_seconds": 0.0,
            "final_train_seconds": final_training.get("final_train_seconds"),
        },
    }
    save_result(output_dir, args.arch, result)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
