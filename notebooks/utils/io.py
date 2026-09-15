"""Load report JSONs and optionally save figures/tables."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from notebooks.utils.paths import cnn_run_dir, linear_probe_run_dir


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_run_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "args.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run args: {path}")
    return load_json(path)


def run_metadata(run_key: str, config: dict[str, Any], args: dict[str, Any] | None = None) -> dict[str, Any]:
    run_name = config["run_name"]
    return {
        "run_key": run_key,
        "run_name": run_name,
        "dataset_display": config.get("display_name", run_name),
        "modality": config.get("modality", "unknown"),
        "stain": config.get("stain", config.get("modality", "unknown")),
        "task_family": config.get("task_family", "unknown"),
        "resolved_dataset": (args or {}).get("resolved_dataset", (args or {}).get("dataset")),
    }


def annotate_result(result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    result.update(metadata)
    return result


def load_linear_probe_results(run_dir: Path, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not run_dir.exists():
        return results
    for result_path in sorted(run_dir.glob("*/result.json")):
        result = load_json(result_path)
        result["analysis_source"] = "linear_probe"
        result["result_path"] = str(result_path)
        if metadata:
            annotate_result(result, metadata)
        results.append(result)
    return results


def load_cnn_results(
    run_dir: Path,
    metadata: dict[str, Any] | None = None,
    args: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not run_dir.exists():
        return results
    for result_path in sorted(run_dir.glob("*/result.json")):
        result = load_json(result_path)
        arch = result_path.parent.name
        result["analysis_source"] = "cnn"
        result["arch"] = arch
        result["model_id"] = f"cnn/{arch}"
        result["model_label"] = f"CNN {arch}"
        result["result_path"] = str(result_path)
        if args:
            result["pretrained"] = args.get("pretrained")
            result["image_size"] = args.get("image_size")
        if metadata:
            annotate_result(result, metadata)
        results.append(result)
    return results


def load_registered_run(run_key: str, config: dict[str, Any]) -> dict[str, Any]:
    run_name = config["run_name"]
    linear_dir = linear_probe_run_dir(run_name)
    cnn_dir = cnn_run_dir(run_name)

    linear_args = load_run_args(linear_dir) if (linear_dir / "args.json").exists() else None
    cnn_args = load_run_args(cnn_dir) if (cnn_dir / "args.json").exists() else None
    run_args = linear_args or cnn_args
    if run_args is None:
        return {
            "run_key": run_key,
            "config": config,
            "available": False,
            "skip_reason": f"No args.json under {linear_dir} or {cnn_dir}",
            "all_results": [],
        }

    if linear_args and cnn_args and linear_args.get("class_names") != cnn_args.get("class_names"):
        raise ValueError(f"Linear-probe and CNN class_names differ for run {run_name!r}")

    metadata = run_metadata(run_key, config, run_args)
    probe_results = load_linear_probe_results(linear_dir, metadata=metadata)
    cnn_results = load_cnn_results(cnn_dir, metadata=metadata, args=cnn_args)
    all_results = probe_results + cnn_results
    if not all_results:
        return {
            "run_key": run_key,
            "config": config,
            "available": False,
            "skip_reason": f"No result.json files under {linear_dir} or {cnn_dir}",
            "run_args": run_args,
            "linear_args": linear_args,
            "cnn_args": cnn_args,
            "probe_results": [],
            "cnn_results": [],
            "all_results": [],
        }

    return {
        "run_key": run_key,
        "config": config,
        "available": True,
        "run_name": run_name,
        "metadata": metadata,
        "run_args": run_args,
        "linear_args": linear_args,
        "cnn_args": cnn_args,
        "linear_dir": linear_dir,
        "cnn_dir": cnn_dir,
        "probe_results": probe_results,
        "cnn_results": cnn_results,
        "all_results": all_results,
    }


def maybe_save_fig(fig, filename: str, *, save_dir: Path | None = None) -> Path | None:
    if save_dir is None:
        return None
    out = save_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved figure -> {out}")
    return out


def maybe_save_table(df: pd.DataFrame, filename: str, *, save_dir: Path | None = None) -> Path | None:
    if save_dir is None:
        return None
    out = save_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved table -> {out}")
    return out


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe or "result"
