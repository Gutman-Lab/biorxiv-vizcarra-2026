"""Select and reorder probe classes relative to dataset config CLASSES."""

from __future__ import annotations

from dataset_configs import get_class_names

NEGATIVE_CLASS_NAME = "negative"
EXCLUDED_CLASS_POLICIES = ("remove", "negative")


def validate_excluded_class_policy(policy: str) -> str:
    if policy not in EXCLUDED_CLASS_POLICIES:
        raise ValueError(
            f"excluded_class_policy must be one of {', '.join(EXCLUDED_CLASS_POLICIES)}, "
            f"got {policy!r}"
        )
    return policy


def excluded_class_names(
    source_classes: list[str], probe_classes: list[str]
) -> list[str]:
    probe_set = set(probe_classes)
    return [name for name in source_classes if name not in probe_set]


def has_excluded_positive(
    raw_labels: list[float] | tuple[float, ...],
    source_classes: list[str],
    probe_classes: list[str],
) -> bool:
    """True when any class outside the probe set has a positive label."""
    values = list(raw_labels)
    if len(values) != len(source_classes):
        raise ValueError(
            f"expected {len(source_classes)} label values, got {len(values)}"
        )
    probe_set = set(probe_classes)
    for name, value in zip(source_classes, values):
        if name not in probe_set and (value or 0) > 0:
            return True
    return False


def training_class_names(probe_classes: list[str], *, label_mode: str) -> list[str]:
    """
    Class names used by the probe head and metrics.

    Single-label mode prepends an explicit negative class at index 0
    (including one-class probes such as til-positive vs negative).
    """
    if label_mode == "multi-label":
        return list(probe_classes)
    return [NEGATIVE_CLASS_NAME, *probe_classes]


def resolve_probe_classes(
    dataset_slug: str,
    classes_arg: list[str] | None,
) -> tuple[list[str], list[str]]:
    """
    Return (probe_classes, source_classes).

    ``source_classes`` comes from the dataset config (unmodified).
    ``probe_classes`` is ``classes_arg`` when provided, otherwise a copy of
    ``source_classes``.
    """
    source_classes = get_class_names(dataset_slug)
    if classes_arg is None:
        return list(source_classes), list(source_classes)

    source_set = set(source_classes)
    unknown = [name for name in classes_arg if name not in source_set]
    if unknown:
        raise ValueError(
            f"unknown class name(s): {', '.join(unknown)}. "
            f"Dataset CLASSES: {source_classes}"
        )

    if len(classes_arg) != len(set(classes_arg)):
        duplicates = sorted({name for name in classes_arg if classes_arg.count(name) > 1})
        raise ValueError(f"duplicate class name(s) in --classes: {', '.join(duplicates)}")

    return list(classes_arg), list(source_classes)


def remap_labels(
    raw_labels: list[float] | tuple[float, ...],
    source_classes: list[str],
    probe_classes: list[str],
) -> list[float]:
    """Select and reorder label values by class name."""
    values = list(raw_labels)
    if len(values) != len(source_classes):
        raise ValueError(
            f"expected {len(source_classes)} label values, got {len(values)}"
        )

    index_by_name = {name: index for index, name in enumerate(source_classes)}
    return [float(values[index_by_name[name]]) for name in probe_classes]


def resolve_probe_label_mode(*, multi_label: bool, num_classes: int) -> str:
    """
    Resolve training/eval mode from CLI flags and the number of probe classes.

    One probe class uses single-label mode (negative vs that class), matching
    the CNN classifier path. Multi-label requires at least two probe classes.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if num_classes == 1 or not multi_label:
        return "single-label"
    return "multi-label"
