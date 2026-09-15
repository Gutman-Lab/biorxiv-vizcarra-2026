"""
Print tile-table metrics for one dataset.

Example:

  uv run python dataset-stats.py --dataset tau
  uv run python dataset-stats.py --dataset abeta-dataset
  uv run python dataset-stats.py --dataset bach --split 80:20:20 --seed 0
  uv run python dataset-stats.py --dataset till
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from typing import Any

import pixeltable as pxt
from pixeltable.exceptions import NotFoundError

from config import (
    EMBEDDINGS_TABLE_NAME,
    SPLIT_NAMES,
    TILES_TABLE_NAME,
    TILES_TABLE_SCHEMA,
)
from dataset_configs import (
    apply_config_train_val_split,
    get_class_names,
    list_configured_datasets,
    resolve_dataset,
)
from utils import get_table_or_create
from utils.linear_probe.splits import (
    add_split_cli_arguments,
    compute_row_splits,
    compute_train_val_row_splits,
    format_ratio_label,
    split_name_breakdown,
)

MISSING = "(none)"
NEGATIVE_COMBO = "negative"
COMBO_SEP = "; "


def parse_args():
    known = ", ".join(sorted(list_configured_datasets().values()))
    parser = ArgumentParser(
        description=(
            "Print tile counts, WSIs, cases, class support, and embeddings "
            "for one dataset. Optional --split / --train-val-split uses the "
            "same packing as the linear probe and CNN classifier."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="DATASET",
        help=(
            "Dataset slug, config module, or hyphenated name "
            f"(e.g. tau-dataset, tau, or abeta). Known: {known}"
        ),
    )
    add_split_cli_arguments(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "RNG seed for --split / --train-val-split assignment "
            "(default: 0, same as probe/CNN)."
        ),
    )
    return parser.parse_args()


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _combo_name(class_names: list[str], positive_indices: list[int]) -> str:
    """Join positive class names in config order, e.g. ``pre_nft; inft``."""
    if not positive_indices:
        return NEGATIVE_COMBO
    return COMBO_SEP.join(class_names[i] for i in positive_indices)


def _label_list(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    return [float(v or 0) for v in list(raw)]


def _print_count_table(title: str, counts: Counter[str], *, total: int) -> None:
    print(f"{title}:")
    if not counts:
        print("  (none)")
        print()
        return
    width = max(len(key) for key in counts)
    for key, n in counts.most_common():
        pct = 100.0 * n / total if total else 0.0
        print(f"  {key:<{width}}  {n:>10,}  ({pct:5.1f}%)")
    print()


def fetch_tile_rows(tiles, dataset: str) -> tuple[list[dict[str, Any]], bool]:
    select_cols = [
        tiles.tileName,
        tiles.wsi_name,
        tiles.caseId,
        tiles.labels,
        tiles.split,
    ]
    has_brain_region = "brainRegion" in tiles.columns()
    if has_brain_region:
        select_cols.append(tiles.brainRegion)

    rows = (
        tiles.select(*select_cols)
        .where(tiles.dataset == dataset)
        .collect()
    )
    return list(rows), has_brain_region


def metrics_from_rows(
    rows: list[dict[str, Any]],
    class_names: list[str],
    *,
    has_brain_region: bool,
) -> dict[str, Any]:
    n_tiles = len(rows)
    wsis: set[str] = set()
    cases: set[str] = set()
    n_missing_wsi = 0
    n_missing_case = 0
    n_missing_labels = 0
    n_label_len_mismatch = 0
    split_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    class_positives = [0] * len(class_names)
    combo_counts: Counter[str] = Counter()
    n_all_zero = 0
    n_multi_positive = 0

    for row in rows:
        wsi = row.get("wsi_name")
        if _is_missing(wsi):
            n_missing_wsi += 1
        else:
            wsis.add(str(wsi))

        case_id = row.get("caseId")
        if _is_missing(case_id):
            n_missing_case += 1
        else:
            cases.add(str(case_id))

        split = row.get("split")
        split_counts[MISSING if _is_missing(split) else str(split)] += 1

        if has_brain_region:
            region = row.get("brainRegion")
            region_counts[MISSING if _is_missing(region) else str(region)] += 1

        labels = _label_list(row.get("labels"))
        if labels is None:
            n_missing_labels += 1
            continue
        if len(labels) != len(class_names):
            n_label_len_mismatch += 1
            continue

        positives = [i for i, v in enumerate(labels) if v > 0]
        combo_counts[_combo_name(class_names, positives)] += 1
        if not positives:
            n_all_zero += 1
        else:
            if len(positives) > 1:
                n_multi_positive += 1
            for i in positives:
                class_positives[i] += 1

    return {
        "n_tiles": n_tiles,
        "n_wsi": len(wsis),
        "n_missing_wsi": n_missing_wsi,
        "n_cases": len(cases),
        "n_missing_case": n_missing_case,
        "split_counts": split_counts,
        "has_brain_region": has_brain_region,
        "region_counts": region_counts,
        "n_missing_labels": n_missing_labels,
        "n_label_len_mismatch": n_label_len_mismatch,
        "class_positives": class_positives,
        "combo_counts": combo_counts,
        "n_all_zero": n_all_zero,
        "n_multi_positive": n_multi_positive,
        "n_labeled": n_tiles - n_missing_labels - n_label_len_mismatch,
    }


def collect_embedding_counts(dataset: str) -> list[tuple[str, int]]:
    try:
        emb = pxt.get_table(EMBEDDINGS_TABLE_NAME)
    except NotFoundError:
        return []

    rows = (
        emb.select(emb.model_id)
        .where(emb.dataset == dataset)
        .collect()
    )
    counts: Counter[str] = Counter()
    for row in rows:
        model_id = row.get("model_id")
        if model_id:
            counts[model_id] += 1
    return [(model_id, counts[model_id]) for model_id in sorted(counts)]


def print_metrics(
    dataset: str,
    class_names: list[str],
    metrics: dict[str, Any],
    *,
    show_header: bool = True,
    show_stored_splits: bool = True,
    section_title: str | None = None,
) -> None:
    n_tiles = metrics["n_tiles"]
    n_labeled = metrics["n_labeled"]

    if section_title:
        print(section_title)
        print()

    if show_header:
        print(f"Dataset: {dataset}")
        print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
        print()

    print(f"Tiles: {n_tiles:,}")
    print()

    print(f"WSIs (distinct wsi_name): {metrics['n_wsi']:,}")
    print(f"  tiles missing wsi_name: {metrics['n_missing_wsi']:,}")
    print(f"Cases (distinct caseId): {metrics['n_cases']:,}")
    print(f"  tiles missing caseId: {metrics['n_missing_case']:,}")
    print()

    if show_stored_splits:
        _print_count_table("Splits (stored column)", metrics["split_counts"], total=n_tiles)

    if metrics["has_brain_region"]:
        _print_count_table("Brain regions", metrics["region_counts"], total=n_tiles)

    print("Tiles positive for each class (a multi-label tile counts in every positive class):")
    if n_labeled == 0:
        print("  (no valid label vectors)")
    else:
        width = max(len(name) for name in class_names)
        for name, n in zip(class_names, metrics["class_positives"]):
            pct = 100.0 * n / n_labeled if n_labeled else 0.0
            print(f"  {name:<{width}}  {n:>10,}  ({pct:5.1f}%)")
        n_zero = metrics["n_all_zero"]
        pct_zero = 100.0 * n_zero / n_labeled if n_labeled else 0.0
        print(f"  {NEGATIVE_COMBO:<{width}}  {n_zero:>10,}  ({pct_zero:5.1f}%)")
    if metrics["n_missing_labels"]:
        print(f"  missing labels: {metrics['n_missing_labels']:,}")
    if metrics["n_label_len_mismatch"]:
        print(
            f"  label length != {len(class_names)}: "
            f"{metrics['n_label_len_mismatch']:,}"
        )
    print()

    if n_labeled:
        _print_count_table(
            "Label combinations (each tile in exactly one bucket)",
            metrics["combo_counts"],
            total=n_labeled,
        )


def print_embeddings(dataset: str, n_tiles: int) -> None:
    models = collect_embedding_counts(dataset)
    print("Embeddings:")
    if not models:
        print("  (none)")
        print()
        return
    width = max(len(model_id) for model_id, _ in models)
    for model_id, n in models:
        pct = 100.0 * n / n_tiles if n_tiles else 0.0
        print(f"  {model_id:<{width}}  {n:>10,} / {n_tiles:,} tiles ({pct:5.1f}%)")
    print()


def print_computed_split_metrics(
    rows: list[dict[str, Any]],
    class_names: list[str],
    *,
    has_brain_region: bool,
    group_key: str,
    group_splits: dict[str, str],
    ratios: tuple[float, ...],
    seed: int,
    heading: str,
) -> None:
    group_counts = Counter(group_splits.values())
    print(
        f"{heading} ({format_ratio_label(ratios)}, seed={seed}) "
        f"by {group_key}: {len(group_splits):,} groups — "
        f"{split_name_breakdown(dict(group_counts))}"
    )
    print()

    n_unassigned = 0
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        key = row.get(group_key)
        assigned = group_splits.get(key) if key else None
        if assigned in by_split:
            by_split[assigned].append(row)
        else:
            n_unassigned += 1

    if n_unassigned:
        print(f"Tiles not assigned to a split (missing {group_key}): {n_unassigned:,}")
        print()

    for split_name in SPLIT_NAMES:
        subset = by_split[split_name]
        metrics = metrics_from_rows(
            subset, class_names, has_brain_region=has_brain_region
        )
        print("=" * 60)
        print_metrics(
            "",
            class_names,
            metrics,
            show_header=False,
            show_stored_splits=False,
            section_title=f"Split: {split_name}",
        )


def main() -> None:
    args = parse_args()
    try:
        dataset = resolve_dataset(args.dataset)
        class_names = get_class_names(dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tiles = get_table_or_create(TILES_TABLE_NAME, TILES_TABLE_SCHEMA)
    rows, has_brain_region = fetch_tile_rows(tiles, dataset)
    metrics = metrics_from_rows(
        rows, class_names, has_brain_region=has_brain_region
    )
    print_metrics(dataset, class_names, metrics)
    print_embeddings(dataset, metrics["n_tiles"])

    apply_config_train_val_split(args, dataset)
    try:
        if args.split:
            group_key, group_splits, ratios = compute_row_splits(
                rows, split=args.split, seed=args.seed
            )
            heading = "Computed split"
        elif args.train_val_split:
            group_key, group_splits, ratios = compute_train_val_row_splits(
                rows, train_val_split=args.train_val_split, seed=args.seed
            )
            heading = "Kept hold-out from table; packed remaining into train/validation"
        else:
            return
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print_computed_split_metrics(
        rows,
        class_names,
        has_brain_region=has_brain_region,
        group_key=group_key,
        group_splits=group_splits,
        ratios=ratios,
        seed=args.seed,
        heading=heading,
    )


if __name__ == "__main__":
    main()
