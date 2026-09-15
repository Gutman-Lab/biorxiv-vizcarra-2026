"""Multi-label linear probe orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from config import TEST_SPLIT, TUNE_SPLIT
from utils.linear_probe.common import save_model_result
from utils.linear_probe.data import load_embeddings_for_model, split_samples
from utils.linear_probe.multilabel_train import evaluate_probe, tune_hyperparameters


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
        model_id=model_id, train_samples=train, tune_samples=tune, args=args,
    )
    test_results = evaluate_probe(
        model_id=model_id, train_samples=train, test_samples=test,
        best_config=best_config, args=args,
    )

    result = {
        "model_id": model_id,
        "skipped": False,
        "label_mode": args.label_mode,
        "n_samples": len(samples),
        "best_config": best_config,
        "test_results": test_results,
    }
    save_model_result(output_dir, model_id, result)
    return result
