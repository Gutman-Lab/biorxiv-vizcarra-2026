"""
Train a linear probe on frozen tile embeddings for one dataset.

Supports multi-label and single-label modes. Splits can be a full
wsi_name/tileName reassignment (``--split 80:20:20``), a train/val pack that
keeps a predefined hold-out (``--train-val-split 80:20`` or config
``TRAIN_VAL_SPLIT``), or the tiles split column.
"""

from __future__ import annotations

# Configure the torch device before anything imports torch.
from utils.torch_device import configure_torch_device, parse_device_cli

configure_torch_device(parse_device_cli())

import argparse
from argparse import ArgumentParser, ArgumentTypeError, Namespace

from config import EMBEDDINGS_TABLE_NAME, EMBEDDINGS_TABLE_SCHEMA, REPORTS_DIR, TILES_TABLE_NAME, TILES_TABLE_SCHEMA
from dataset_configs import apply_config_train_val_split, resolve_dataset
from utils import get_table_or_create
from utils.linear_probe.classes import (
    resolve_probe_classes,
    resolve_probe_label_mode,
    training_class_names,
)
from utils.linear_probe.common import (
    ensure_report_dir_available,
    save_run_args,
    validate_output_name,
)
from utils.linear_probe.data import load_tile_metadata, resolve_probe_models
from utils.linear_probe.splits import add_split_cli_arguments
from utils.linear_probe.multilabel import run_probe_for_model as run_multilabel_probe
from utils.linear_probe.single_label import (
    filter_single_label_metadata,
    run_probe_for_model as run_single_label_probe,
)

DEFAULT_LEARNING_RATES = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
PROBE_TYPES = ("linear", "logistic", "svm")
MULTILABEL_TUNE_METRICS = (
    "macro_f1",
    "macro_recall",
    "macro_precision",
    "balanced_accuracy",
    "micro_f1",
    "subset_accuracy",
)
SINGLE_LABEL_TUNE_METRICS = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "macro_recall",
    "macro_precision",
)
CLASS_WEIGHTS = ("balanced", "none")


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
            "Linear probe on frozen embeddings for one dataset. "
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
        default=True,
        help="Multi-label probe (default). Use --no-multi-label for single-label.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="CLASS",
        default=None,
        help=(
            "Class names to keep, in probe output order (subset of dataset CLASSES). "
            "Default: all classes from the dataset config, in config order. "
            "A single class becomes a single-label probe (negative vs that class)."
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
    parser.add_argument("--models", nargs="+", metavar="MODEL_ID", default=None)
    parser.add_argument("--probe-type", choices=PROBE_TYPES, default="linear")
    parser.add_argument(
        "--learning-rates", nargs="+", type=_positive_float, metavar="LR",
        default=list(DEFAULT_LEARNING_RATES),
    )
    parser.add_argument(
        "--c-values", nargs="+", type=_positive_float, metavar="C",
        default=list(DEFAULT_C_VALUES),
        help="For future logistic/svm probes (ignored for linear).",
    )
    parser.add_argument(
        "--class-weights", nargs="+", choices=CLASS_WEIGHTS, default=["balanced"], metavar="WEIGHT",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Validation metric for hyperparameter selection, best-checkpoint saving, "
            "and early stopping (default depends on label mode)."
        ),
    )
    parser.add_argument(
        "--tune-epochs",
        type=_positive_int,
        default=15,
        help="Max epochs per grid cell during single-label hyperparameter tuning.",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=100,
        help="Max epochs for single-label final training (after tuning).",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-patience",
        type=_non_negative_int,
        default=10,
        metavar="N",
        help=(
            "Final training only (single-label): stop after N epochs without "
            "validation metric improvement. Tuning uses a fixed --tune-epochs budget "
            "with no early stopping. 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=_non_negative_float,
        default=0.005,
        metavar="DELTA",
        help=(
            "Minimum --metric improvement required to reset early-stop patience or "
            "count a new best checkpoint (single-label)."
        ),
    )
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        metavar="NAME",
        help=(
            "Report folder name under reports/linear-probe/ (required). "
            "The run aborts if that folder already exists."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed for dataset split assignment, --max-tiles subsampling, "
            "and probe training (saved in args.json for reproduction)."
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
    args = parser.parse_args(argv)

    if args.metric is None:
        args.metric = "macro_f1"

    return args


def _validate_metric(metric: str, label_mode: str) -> None:
    if label_mode == "multi-label":
        allowed = MULTILABEL_TUNE_METRICS
    else:
        allowed = SINGLE_LABEL_TUNE_METRICS
    if metric not in allowed:
        raise SystemExit(
            f"--metric {metric!r} is not valid for {label_mode} mode. "
            f"Choose one of: {', '.join(allowed)}"
        )


def _resolve_dataset_arg(dataset_arg: str) -> str:
    try:
        return resolve_dataset(dataset_arg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    args = parse_args()
    dataset = _resolve_dataset_arg(args.dataset)
    apply_config_train_val_split(args, dataset)

    try:
        probe_classes, source_classes = resolve_probe_classes(dataset, args.classes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.classes = probe_classes
    args.source_class_names = source_classes
    args.label_mode = resolve_probe_label_mode(
        multi_label=args.multi_label,
        num_classes=len(probe_classes),
    )
    args.class_names = training_class_names(probe_classes, label_mode=args.label_mode)
    _validate_metric(args.metric, args.label_mode)

    from utils.torch_device import resolve_runtime_device_label

    print(f"Using device: {resolve_runtime_device_label()}", flush=True)

    try:
        output_dir = REPORTS_DIR / "linear-probe" / validate_output_name(args.output_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ensure_report_dir_available(output_dir)
    output_dir.mkdir(parents=True)
    args_path = save_run_args(output_dir, args, dataset=dataset)

    label_mode = args.label_mode
    print(f"Dataset: {dataset}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Saved run args: {args_path}", flush=True)
    print(f"Label mode: {label_mode}", flush=True)
    if label_mode == "single-label":
        print(
            f"Training: tune_epochs={args.tune_epochs}, final_epochs={args.epochs}, "
            f"metric={args.metric}, early_stop_patience={args.early_stop_patience}, "
            f"early_stop_min_delta={args.early_stop_min_delta}",
            flush=True,
        )
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

    tiles = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)
    embeddings = get_table_or_create(EMBEDDINGS_TABLE_NAME, EMBEDDINGS_TABLE_SCHEMA)

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
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if label_mode == "single-label":
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

    model_ids = resolve_probe_models(embeddings, dataset, args.models)
    print(f"Models to probe ({len(model_ids)}): {', '.join(model_ids)}", flush=True)

    run_probe = (
        run_multilabel_probe
        if label_mode == "multi-label"
        else run_single_label_probe
    )

    for i, model_id in enumerate(model_ids, start=1):
        run_probe(
            model_id, i, len(model_ids),
            dataset=dataset,
            embeddings_table=embeddings,
            tile_metadata=tile_metadata,
            output_dir=output_dir,
            args=args,
        )

    print(f"Done.", flush=True)


if __name__ == "__main__":
    main()
