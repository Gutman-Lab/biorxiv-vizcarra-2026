"""Torchvision model factory for tile CNN classifiers."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

SUPPORTED_ARCHITECTURES = ("resnet18", "resnet34", "resnet50")

_ResNetBuilder = Callable[[bool], nn.Module]


def _build_resnet18(*, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return models.resnet18(weights=weights)


def _build_resnet34(*, pretrained: bool) -> nn.Module:
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    return models.resnet34(weights=weights)


def _build_resnet50(*, pretrained: bool) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return models.resnet50(weights=weights)


_BUILDERS: dict[str, _ResNetBuilder] = {
    "resnet18": _build_resnet18,
    "resnet34": _build_resnet34,
    "resnet50": _build_resnet50,
}


def build_classifier(
    arch: str,
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a torchvision ResNet classifier with a replaced final linear head.

    Supported architectures: ``resnet18``, ``resnet34``, ``resnet50``.
    """
    normalized_arch = arch.lower()
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if normalized_arch not in _BUILDERS:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(
            f"unsupported architecture {arch!r}; expected one of: {supported}"
        )

    model = _BUILDERS[normalized_arch](pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
