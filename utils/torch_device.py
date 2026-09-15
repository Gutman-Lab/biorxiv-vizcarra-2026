"""Configure PyTorch/CUDA device before torch is imported."""

from __future__ import annotations

import os
import sys

_ENV_DEVICE = "PIXELEMBED_TORCH_DEVICE"


def parse_device_cli(argv: list[str] | None = None) -> str:
    """Read --device from argv without importing torch."""
    argv = argv if argv is not None else sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return "auto"


def configure_torch_device(device: str) -> str:
    """
    Apply device selection via environment variables before ``import torch``.

    For ``cuda:N``, sets ``CUDA_VISIBLE_DEVICES=N`` so Pixeltable's built-in
    UDFs (which use ``resolve_torch_device('auto')``) see that GPU as ``cuda:0``.

    Returns a human-readable label for logging.
    """
    device = device.strip().lower()

    if device == "auto":
        os.environ.pop(_ENV_DEVICE, None)
        return "auto"

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ[_ENV_DEVICE] = "cpu"
        return "cpu"

    if device == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ[_ENV_DEVICE] = "cuda"
        return "cuda:0"

    if device.startswith("cuda:"):
        index = device.split(":", 1)[1].strip()
        if not index.isdigit():
            raise ValueError(
                f"Invalid device {device!r}. Expected cuda:N with integer N."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = index
        os.environ[_ENV_DEVICE] = "cuda"
        return f"cuda:0 (physical GPU {index})"

    raise ValueError(
        f"Unsupported device {device!r}. Use auto, cpu, cuda, or cuda:N."
    )


def resolve_runtime_device_label() -> str:
    """Return the device PyTorch will use (call after ``import torch``)."""
    import torch

    configured = os.environ.get(_ENV_DEVICE)
    if configured == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != "":
            return (
                f"physical GPU {visible} "
                f"(PyTorch sees cuda:0 — {name})"
            )
        return f"cuda:0 ({name})"
    return "cpu"


def get_torch_device():
    """Return the active ``torch.device`` (call after ``configure_torch_device``)."""
    import torch

    configured = os.environ.get(_ENV_DEVICE)
    if configured == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
