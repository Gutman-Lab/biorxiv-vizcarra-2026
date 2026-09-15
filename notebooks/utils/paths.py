"""Path helpers for analysis notebooks."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of ``notebooks/``)."""
    return Path(__file__).resolve().parents[2]


def linear_probe_run_dir(run_name: str, *, root: Path | None = None) -> Path:
    base = root if root is not None else repo_root()
    return base / "reports" / "linear-probe" / run_name


def cnn_run_dir(run_name: str, *, root: Path | None = None) -> Path:
    base = root if root is not None else repo_root()
    return base / "reports" / "cnn" / run_name


def embedding_timing_md(*, root: Path | None = None) -> Path:
    base = root if root is not None else repo_root()
    return base / "reports" / "embedding_models.md"
