"""Per-dataset configuration modules."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pixeltable as pxt


def list_configured_datasets() -> dict[str, str]:
    """Return {config_module_name: dataset_slug} for each dataset_configs/*.py."""
    package_dir = Path(__file__).parent
    result: dict[str, str] = {}
    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"dataset_configs.{info.name}")
        slug = getattr(mod, "DATASET", None)
        if slug is not None:
            result[info.name] = slug
    return result


def resolve_dataset(name: str) -> str:
    """
    Accept a dataset slug (e.g. tau-dataset), config module name (e.g.
    tau), or the hyphenated module name (e.g. abeta).

    Returns the canonical dataset slug used in Pixeltable rows.
    """
    configured = list_configured_datasets()
    if name in configured.values():
        return name
    if name in configured:
        return configured[name]
    as_module = name.replace("-", "_")
    if as_module in configured:
        return configured[as_module]
    as_slug = name.replace("_", "-")
    if as_slug in configured.values():
        return as_slug
    if f"{as_slug}-dataset" in configured.values():
        return f"{as_slug}-dataset"
    known = ", ".join(
        f"{slug} ({module})" for module, slug in sorted(configured.items())
    )
    raise ValueError(f"Unknown dataset {name!r}. Known datasets: {known}")


def list_datasets_in_table(tiles: pxt.Table) -> list[str]:
    """Distinct dataset slugs present in the tiles table."""
    rows = tiles.select(tiles.dataset).collect()
    datasets = {r["dataset"] for r in rows if r.get("dataset")}
    return sorted(datasets)


def get_dataset_module(dataset_slug: str):
    """Return the dataset_configs module for a resolved dataset slug."""
    configured = list_configured_datasets()
    for module_name, slug in configured.items():
        if slug == dataset_slug:
            return importlib.import_module(f"dataset_configs.{module_name}")
    known = ", ".join(sorted(configured.values()))
    raise ValueError(f"Unknown dataset {dataset_slug!r}. Known datasets: {known}")


def get_class_names(dataset_slug: str) -> list[str]:
    """Return label class names for a dataset (CLASSES or CLASS_NAMES in config)."""
    mod = get_dataset_module(dataset_slug)
    classes = getattr(mod, "CLASSES", None) or getattr(mod, "CLASS_NAMES", None)
    if not classes:
        raise ValueError(
            f"dataset {dataset_slug!r} config must define CLASSES or CLASS_NAMES"
        )
    return list(classes)


def get_train_val_split(dataset_slug: str) -> str | None:
    """
    Return ``TRAIN_VAL_SPLIT`` from the dataset config, if set.

    Used when hold-out is predefined in the split column but train/validation
    still need to be packed (e.g. ``80:20``).
    """
    mod = get_dataset_module(dataset_slug)
    value = getattr(mod, "TRAIN_VAL_SPLIT", None)
    if value is None or value == "":
        return None
    return str(value)


def apply_config_train_val_split(args, dataset_slug: str) -> None:
    """Fill ``args.train_val_split`` from dataset config when both CLI flags are unset."""
    if getattr(args, "split", None):
        return
    if getattr(args, "train_val_split", None):
        return
    args.train_val_split = get_train_val_split(dataset_slug)
