"""
Load or reference image data for a given path.

Single responsibility: given a path you specify, resolve it and provide
the image (PIL) or a lightweight reference. No Pixeltable or app dependencies.
"""

from pathlib import Path
from typing import Iterator

from PIL import Image


# Common image extensions (PIL can open these)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp", ".ndpi"}


def resolve_path(path: str | Path) -> Path:
    """Resolve to an absolute Path. Expands user and env if needed."""
    p = Path(path).expanduser().resolve()
    return p


def load_image(path: str | Path) -> Image.Image:
    """
    Load image from path; return PIL Image.

    Raises:
        FileNotFoundError: path does not exist
        OSError: path exists but is not a file or cannot be opened as image
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    if not p.is_file():
        raise OSError(f"Not a file: {p}")
    return Image.open(p).copy()


def image_paths_from_dir(
    directory: str | Path,
    *,
    patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.ndpi"),
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


class ImageRef:
    """
    Lightweight reference to an image path. Loads on first access.

    Use when you want to pass around a path and only load when needed.
    """

    __slots__ = ("_path", "_image")

    def __init__(self, path: str | Path) -> None:
        self._path = resolve_path(path)
        self._image: Image.Image | None = None

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def load(self) -> Image.Image:
        """Load and cache the image; return PIL Image."""
        if self._image is None:
            self._image = load_image(self._path)
        return self._image

    @property
    def image(self) -> Image.Image:
        """Same as load(); for property-style access."""
        return self.load()

    def __repr__(self) -> str:
        return f"ImageRef({self._path!r})"
