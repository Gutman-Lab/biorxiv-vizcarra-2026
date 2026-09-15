"""Map multi-hot probe labels to a single class index."""

from __future__ import annotations


def labels_to_class_index(labels: list[int]) -> int:
    """
    Map probe-class binary labels to a class index.

    Index 0 is negative (all probe classes zero). A single positive at probe
    index ``i`` maps to class ``i + 1``.
    """
    positives = [index for index, value in enumerate(labels) if value > 0]
    if not positives:
        return 0
    if len(positives) > 1:
        raise ValueError(
            f"expected at most one positive probe class, got {len(positives)}"
        )
    return positives[0] + 1
