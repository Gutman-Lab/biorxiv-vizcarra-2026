"""Dispatch: python -m load_tiles <dataset> [loader args...]

Examples:

  uv run python -m load_tiles tau --source /path/to/vizcarra-2023-tiles
  uv run python -m load_tiles tau-dataset --source /path/to/vizcarra-2023-tiles
  uv run python -m load_tiles tau --help
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path


def _loader_names() -> list[str]:
    package_dir = Path(__file__).parent
    return sorted(
        info.name
        for info in pkgutil.iter_modules([str(package_dir)])
        if not info.name.startswith("_")
    )


def _resolve_loader(name: str) -> str:
    loaders = _loader_names()
    if name in loaders:
        return name
    as_module = name.replace("-", "_")
    if as_module in loaders:
        return as_module

    from dataset_configs import list_configured_datasets

    for module_name, slug in list_configured_datasets().items():
        if name == slug and module_name in loaders:
            return module_name

    known = ", ".join(loaders)
    raise SystemExit(
        f"Unknown dataset loader {name!r}. Known loaders: {known}\n"
        "Usage: python -m load_tiles <dataset> [loader args...]"
    )


def _print_usage() -> None:
    loaders = ", ".join(_loader_names())
    print(
        "Load tiles into Pixeltable for a dataset.\n"
        "\n"
        "Usage: python -m load_tiles <dataset> [loader args...]\n"
        "\n"
        f"Datasets: {loaders}\n"
        "\n"
        "Example:\n"
        "  uv run python -m load_tiles tau --source /path/to/vizcarra-2023-tiles\n"
        "\n"
        "Pass --help after the dataset name for that loader's options.\n"
    )


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_usage()
        raise SystemExit(0 if argv else 2)
    if argv[0].startswith("-"):
        _print_usage()
        raise SystemExit(2)

    module_name = _resolve_loader(argv[0])
    sys.argv = [f"python -m load_tiles {module_name}", *argv[1:]]
    mod = importlib.import_module(f"load_tiles.{module_name}")
    mod.main()


if __name__ == "__main__":
    main()
