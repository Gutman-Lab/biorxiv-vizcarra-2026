# Utility functions.
from pathlib import Path
from typing import Iterator


def resolve_path(path: str | Path) -> Path:
    """Resolve to an absolute Path. Expands user and env if needed."""
    p = Path(path).expanduser().resolve()
    return p


def image_paths_from_dir(
    directory: str | Path,
    *,
    patterns: tuple[str, ...] = (
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tiff",
        "*.tif",
        "*.ndpi",
    ),
    recursive: bool = True,
) -> Iterator[Path]:
    """
    Yield image file paths under directory (each path once).

    patterns: glob patterns (e.g. ("*.png", "*.jpg")).
    recursive: if True, search subdirectories (rglob); else list only top-level (glob).
    """
    root = resolve_path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    seen: set[Path] = set()
    for pattern in patterns:
        gen = root.rglob(pattern) if recursive else root.glob(pattern)
        for p in gen:
            if p not in seen and p.is_file():
                seen.add(p)
                yield p
