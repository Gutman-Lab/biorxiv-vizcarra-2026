"""PyTorch dataset and DataLoader helpers for tile images loaded from on-disk paths."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from core.image_loader import load_image
from utils.classification.labels import labels_to_class_index

# Sample dicts come from ``utils.linear_probe.data.load_image_samples()`` after
# ``load_tile_metadata(..., include_file_path=True)``, single-label filtering, and
# optional ``max_tiles`` stratified subsampling (see ``load_tile_metadata``).
ImageSample = dict[str, Any]
Transform = Callable[[Image.Image], Any]


class TileImageDataset(Dataset[tuple[Any, int]]):
    """
    Load square RGB tile images from ``filePath`` and map labels to a class index.

    Each sample dict must include ``filePath`` and ``labels`` (probe-class binary
    vector). Class index 0 is all-negative; a single positive at probe index ``i``
    maps to ``i + 1`` via ``labels_to_class_index``.
    """

    __slots__ = ("_class_indices", "_samples", "_transform")

    def __init__(
        self,
        samples: Sequence[ImageSample],
        *,
        transform: Transform | None = None,
    ) -> None:
        if not samples:
            raise ValueError("cannot build TileImageDataset from an empty sample list")

        self._samples = list(samples)
        self._transform = transform
        self._class_indices = [
            labels_to_class_index(sample["labels"]) for sample in self._samples
        ]

    @property
    def class_indices(self) -> list[int]:
        """Class index for each sample, aligned with ``samples`` order."""
        return list(self._class_indices)

    @property
    def samples(self) -> list[ImageSample]:
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        sample = self._samples[index]
        file_path = sample.get("filePath")
        if not file_path:
            tile_name = sample.get("tileName", index)
            raise ValueError(f"sample {tile_name!r} is missing filePath")

        image = load_image(file_path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image, self._class_indices[index]


def class_indices_from_samples(samples: Sequence[ImageSample]) -> torch.Tensor:
    """Return class indices for a sample list as a long tensor."""
    if not samples:
        raise ValueError("cannot build class indices from an empty sample list")
    return torch.tensor(
        [labels_to_class_index(sample["labels"]) for sample in samples],
        dtype=torch.long,
    )


def build_dataloader(
    samples: Sequence[ImageSample],
    *,
    transform: Transform | None = None,
    batch_size: int,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader[tuple[Any, int]]:
    """
    Build a DataLoader over ``TileImageDataset``.

    ``sampler`` and ``shuffle`` are mutually exclusive (PyTorch requirement).
    Set ``pin_memory=True`` when training on CUDA, matching the linear probe.
    """
    if not samples:
        raise ValueError("cannot build DataLoader from an empty sample list")
    if sampler is not None and shuffle:
        raise ValueError("sampler and shuffle are mutually exclusive")

    dataset = TileImageDataset(samples, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_train_dataloader(
    samples: Sequence[ImageSample],
    *,
    transform: Transform,
    batch_size: int,
    sampler: WeightedRandomSampler | None = None,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader[tuple[Any, int]]:
    """Train loader: shuffled unless a weighted sampler is provided."""
    return build_dataloader(
        samples,
        transform=transform,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_eval_dataloader(
    samples: Sequence[ImageSample],
    *,
    transform: Transform,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader[tuple[Any, int]]:
    """Eval/validation loader: no shuffling or sampler."""
    return build_dataloader(
        samples,
        transform=transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
