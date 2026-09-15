"""Shared helpers for linear probe result I/O."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

def validate_output_name(output_name: str) -> str:
    if not output_name or output_name != output_name.strip():
        raise ValueError("output_name must be a non-empty string")
    if output_name in (".", ".."):
        raise ValueError(f"invalid output_name: {output_name!r}")
    if "/" in output_name or "\\" in output_name:
        raise ValueError(
            f"output_name must not contain path separators, got {output_name!r}"
        )
    return output_name


def ensure_report_dir_available(output_dir: Path) -> None:
    if output_dir.exists():
        raise SystemExit(
            f"Report directory already exists: {output_dir}\n"
            "Delete it manually before running again."
        )


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def model_output_dir(output_dir: Path, model_id: str) -> Path:
    """Per-embedding report directory under a linear-probe run."""
    return output_dir / safe_model_id(model_id)


def result_path(output_dir: Path, model_id: str) -> Path:
    return model_output_dir(output_dir, model_id) / "result.json"


def tune_checkpoint_root(output_dir: Path, model_id: str) -> Path:
    return model_output_dir(output_dir, model_id) / "tune_checkpoints"


def final_checkpoint_dir(output_dir: Path, model_id: str) -> Path:
    return model_output_dir(output_dir, model_id) / "checkpoints"


def best_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "best.pt"


def last_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "last.pt"


def args_path(output_dir: Path) -> Path:
    return output_dir / "args.json"


def namespace_to_json_dict(args: Namespace, *, dataset: str) -> dict[str, Any]:
    """Serialize CLI args for args.json (includes resolved dataset slug)."""
    payload = vars(args).copy()
    payload["resolved_dataset"] = dataset
    return payload


def save_run_args(output_dir: Path, args: Namespace, *, dataset: str) -> Path:
    path = args_path(output_dir)
    path.write_text(
        json.dumps(namespace_to_json_dict(args, dataset=dataset), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def save_model_result(output_dir: Path, model_id: str, result: dict[str, Any]) -> None:
    path = result_path(output_dir, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
