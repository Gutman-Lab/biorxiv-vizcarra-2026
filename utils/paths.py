"""Default paths that can be overridden with environment variables."""

from pathlib import Path

# Sibling repo for custom WSU models (UPath, etc.). Override with IMAGE_EMBEDDING_COMPARISON_ROOT.
DEFAULT_IMAGE_EMBEDDING_COMPARISON_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "ImageEmbeddingModelComparison"
)
