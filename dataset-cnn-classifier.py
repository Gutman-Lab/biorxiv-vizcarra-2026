"""
Train a single-label CNN classifier on tile images for one dataset.

Hyperparameters are tuned on validation; the best checkpoint is evaluated once
on hold-out. Splits match dataset-linear-probe.py when given the same dataset,
--split / --train-val-split, --seed, --classes, and --excluded-class-policy
(tiles without filePath are excluded from CNN training only).
"""

from __future__ import annotations

# Configure the torch device before anything imports torch.
from utils.torch_device import configure_torch_device, parse_device_cli

configure_torch_device(parse_device_cli())

import argparse
from argparse import ArgumentParser, ArgumentTypeError, Namespace

from config import REPORTS_DIR, TILES_TABLE_NAME, TILES_TABLE_SCHEMA
from dataset_configs import apply_config_train_val_split, resolve_dataset
from utils import get_table_or_create
from utils.cnn_classifier.class_weights import CLASS_WEIGHT_MODES
from utils.cnn_classifier.common import (
    ensure_report_dir_available,
    save_run_args,
    validate_output_name,
)
from utils.cnn_classifier.models import SUPPORTED_ARCHITECTURES
from utils.cnn_classifier.sampler import SAMPLER_MODES
from utils.cnn_classifier.single_label import run_cnn_classifier
from utils.cnn_classifier.transforms import DEFAULT_IMAGE_SIZE
from utils.cnn_classifier.tune import (
    DEFAULT_CLASS_WEIGHTS,
    DEFAULT_LEARNING_RATES,
    DEFAULT_WEIGHT_DECAYS,
    TUNE_METRICS,
)
from utils.linear_probe.classes import resolve_probe_classes, training_class_names
from utils.linear_probe.data import load_tile_metadata
from utils.linear_probe.single_label import filter_single_label_metadata
from utils.linear_probe.splits import add_split_cli_arguments


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return n


def _positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise ArgumentTypeError(f"expected a positive float, got {value!r}")
    return x


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return n


def _non_negative_float(value: str) -> float:
    x = float(value)
    if x < 0:
        raise ArgumentTypeError(f"expected a non-negative float, got {value!r}")
    return x


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description=(
            "Single-label CNN classifier on tile images for one dataset. "
            "Tune on validation, evaluate on hold-out."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="DATASET",
        help="Dataset slug or config module name (required).",
    )
    add_split_cli_arguments(parser)
    parser.add_argument(
        "--multi-label",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Not supported for CNN classifier v1 (single-label only).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="CLASS",
        default=None,
        help=(
            "Class names to keep, in output order (subset of dataset CLASSES). "
            "Default: all classes from the dataset config, in config order."
        ),
    )
    parser.add_argument(
        "--excluded-class-policy",
        choices=("remove", "negative"),
        default="remove",
        help=(
            "How to handle tiles with a positive label outside --classes. "
            "remove (default): drop those tiles. "
            "negative: keep them; excluded-only positives become negative."
        ),
    )
    parser.add_argument(
        "--arch",
        choices=SUPPORTED_ARCHITECTURES,
        default="resnet50",
        help="Torchvision ResNet architecture (default: resnet50).",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet pretrained weights (default: true).",
    )
    parser.add_argument(
        "--image-size",
        type=_positive_int,
        default=DEFAULT_IMAGE_SIZE,
        metavar="PX",
        help=f"Square input size in pixels (default: {DEFAULT_IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--sampler",
        choices=SAMPLER_MODES,
        default="none",
        help="Training sampler: natural (none) or weighted oversampling.",
    )
    parser.add_argument(
        "--tune-epochs",
        type=_positive_int,
        default=15,
        help="Max epochs per grid cell during hyperparameter tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=100,
        help="Max epochs for final training (after tuning).",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument(
        "--early-stop-patience",
        type=_non_negative_int,
        default=10,
        metavar="N",
        help=(
            "Final training only: stop after N epochs without validation metric "
            "improvement. Tuning uses a fixed --tune-epochs budget with no early "
            "stopping. 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=_non_negative_float,
        default=0.005,
        metavar="DELTA",
        help=(
            "Minimum --metric improvement required to reset early-stop patience or "
            "count a new best checkpoint."
        ),
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=_positive_float,
        metavar="LR",
        default=list(DEFAULT_LEARNING_RATES),
    )
    parser.add_argument(
        "--weight-decays",
        nargs="+",
        type=_positive_float,
        metavar="WD",
        default=list(DEFAULT_WEIGHT_DECAYS),
    )
    parser.add_argument(
        "--class-weights",
        nargs="+",
        choices=CLASS_WEIGHT_MODES,
        default=list(DEFAULT_CLASS_WEIGHTS),
        metavar="WEIGHT",
        help=(
            "Class-weight modes for the tuning grid. "
            f"Default sweep: {', '.join(DEFAULT_CLASS_WEIGHTS)}. "
            "inverse_freq is accepted but omitted from the default grid."
        ),
    )
    parser.add_argument(
        "--metric",
        default="macro_f1",
        choices=TUNE_METRICS,
        help=(
            "Validation metric for hyperparameter selection, best-checkpoint saving, "
            "and early stopping (default: macro_f1)."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        metavar="NAME",
        help=(
            "Report folder name under reports/cnn/ (required). "
            "The run aborts if that folder already exists."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed for dataset split assignment, --max-tiles subsampling, "
            "and training (saved in args.json for reproduction)."
        ),
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
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Cap tiles for smoke tests (stratified random sample across splits).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def _resolve_dataset_arg(dataset_arg: str) -> str:
    try:
        return resolve_dataset(dataset_arg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    args = parse_args()
    if args.multi_label:
        raise SystemExit(
            "CNN classifier v1 supports single-label only; omit --multi-label."
        )

    dataset = _resolve_dataset_arg(args.dataset)
    apply_config_train_val_split(args, dataset)

    try:
        probe_classes, source_classes = resolve_probe_classes(dataset, args.classes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.classes = probe_classes
    args.source_class_names = source_classes
    args.label_mode = "single-label"
    args.class_names = training_class_names(probe_classes, label_mode=args.label_mode)

    from utils.torch_device import resolve_runtime_device_label

    print(f"Using device: {resolve_runtime_device_label()}", flush=True)

    try:
        output_dir = REPORTS_DIR / "cnn" / validate_output_name(args.output_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ensure_report_dir_available(output_dir)
    output_dir.mkdir(parents=True)
    args_path = save_run_args(output_dir, args, dataset=dataset)

    print(f"Dataset: {dataset}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Saved run args: {args_path}", flush=True)
    print(f"Label mode: {args.label_mode}", flush=True)
    print(f"Classes ({len(args.class_names)}): {', '.join(args.class_names)}", flush=True)
    if probe_classes != source_classes:
        print(f"Dataset CLASSES ({len(source_classes)}): {', '.join(source_classes)}", flush=True)
    if args.split:
        print(f"Split ratios: {args.split}", flush=True)
        print(f"Split seed: {args.seed}", flush=True)
    elif args.train_val_split:
        print(
            f"Train/val split: {args.train_val_split} (hold-out kept from table)",
            flush=True,
        )
        print(f"Split seed: {args.seed}", flush=True)
    else:
        print("Split source: tiles.split column", flush=True)
    print(
        f"Training: arch={args.arch}, pretrained={args.pretrained}, "
        f"tune_epochs={args.tune_epochs}, final_epochs={args.epochs}, "
        f"batch_size={args.batch_size}, metric={args.metric}, "
        f"early_stop_patience={args.early_stop_patience}, "
        f"early_stop_min_delta={args.early_stop_min_delta}",
        flush=True,
    )

    tiles = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)

    try:
        tile_metadata = load_tile_metadata(
            tiles,
            dataset=dataset,
            source_class_names=source_classes,
            probe_class_names=probe_classes,
            split=args.split,
            train_val_split=args.train_val_split,
            excluded_class_policy=args.excluded_class_policy,
            seed=args.seed,
            max_tiles=args.max_tiles,
            include_file_path=True,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tile_metadata, removed = filter_single_label_metadata(tile_metadata)
    print(
        "Single-label filter: removed "
        f"{removed['removed_multi_label']:,} multi-label tiles; "
        f"{len(tile_metadata):,} remaining "
        f"({removed['negative_tiles']:,} negative, "
        f"{len(tile_metadata) - removed['negative_tiles']:,} positive).",
        flush=True,
    )
    if not tile_metadata:
        raise SystemExit("No tiles remain after single-label filtering.")

    run_cnn_classifier(
        dataset=dataset,
        tile_metadata=tile_metadata,
        output_dir=output_dir,
        args=args,
    )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
