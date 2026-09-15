"""Single-label linear probe orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from config import TEST_SPLIT, TUNE_SPLIT
from utils.linear_probe.common import save_model_result
from utils.linear_probe.data import load_embeddings_for_model, split_samples
from utils.linear_probe.single_label_train import evaluate_probe, tune_hyperparameters


def filter_single_label_metadata(
    tile_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """
    Keep tiles with zero or one positive probe-class label.

    All-negative tiles are kept as the negative class (index 0). Only tiles
    with multiple positive probe classes are removed.
    """
    filtered: dict[str, dict[str, Any]] = {}
    removed_multi = 0
    negative_count = 0

    for tile_name, meta in tile_metadata.items():
        positive_count = sum(meta["labels"])
        if positive_count > 1:
            removed_multi += 1
            continue
        if positive_count == 0:
            negative_count += 1
        filtered[tile_name] = meta

    return filtered, {
        "removed_multi_label": removed_multi,
        "negative_tiles": negative_count,
    }


def run_probe_for_model(
    model_id: str,
    model_idx: int,
    total_models: int,
    *,
    dataset: str,
    embeddings_table,
    tile_metadata: dict[str, dict[str, Any]],
    output_dir: Path,
    args: Namespace,
) -> dict[str, Any]:
    print(f"Model {model_idx}/{total_models}: {model_id}", flush=True)

    samples = load_embeddings_for_model(
        embeddings_table, model_id, dataset, tile_metadata
    )
    if not samples:
        print(f"  [{model_id}] no samples; skipping.", flush=True)
        return {"model_id": model_id, "skipped": True}

    train, tune, test = split_samples(samples)
    print(
        f"  [{model_id}] samples: train={len(train):,}, "
        f"{TUNE_SPLIT}={len(tune):,}, {TEST_SPLIT}={len(test):,}",
        flush=True,
    )

    best_config = tune_hyperparameters(
        model_id=model_id,
        train_samples=train,
        tune_samples=tune,
        output_dir=output_dir,
        args=args,
    )
    test_results = evaluate_probe(
        model_id=model_id,
        train_samples=train,
        val_samples=tune,
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
        "best_config": best_config,
        "test_results": test_results,
        "timing": {
            "tune_seconds": best_config.get("tune_seconds"),
            "final_train_seconds": test_results.get("final_train_seconds"),
        },
    }
    save_model_result(output_dir, model_id, result)
    return result
