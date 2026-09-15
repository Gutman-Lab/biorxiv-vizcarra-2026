"""Conservative train/eval transforms for tile CNN classifiers."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224

# Fixed mild jitter for the initial hyperparameter sweep (not CLI-tunable in v1).
COLOR_JITTER_BRIGHTNESS = 0.1
COLOR_JITTER_CONTRAST = 0.1
COLOR_JITTER_SATURATION = 0.1
COLOR_JITTER_HUE = 0.02
COLOR_JITTER_APPLY_P = 0.8

FLIP_P = 0.5


def build_train_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Train transforms: resize, flips, optional mild color jitter, ImageNet normalize.

    Avoids crop, rotation, translation, and affine transforms so labeled objects
    at tile edges are not moved or removed.
    """
    if image_size < 1:
        raise ValueError(f"image_size must be positive, got {image_size}")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=FLIP_P),
            transforms.RandomVerticalFlip(p=FLIP_P),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=COLOR_JITTER_BRIGHTNESS,
                        contrast=COLOR_JITTER_CONTRAST,
                        saturation=COLOR_JITTER_SATURATION,
                        hue=COLOR_JITTER_HUE,
                    )
                ],
                p=COLOR_JITTER_APPLY_P,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Eval transforms: resize and ImageNet normalize only."""
    if image_size < 1:
        raise ValueError(f"image_size must be positive, got {image_size}")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
