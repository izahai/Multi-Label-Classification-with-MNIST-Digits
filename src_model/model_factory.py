"""Build detector architectures and restore the correct checkpoint model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from .mnist_detector import MNISTDetector
from .small_mnist_detector import SmallMNISTDetector


MODEL_NAMES = ("small", "large")
DEFAULT_MODEL_NAME = "small"
LARGE_MODEL_PARAMETERS = 4_585_519


def build_detector(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    num_classes: int = 10,
    num_slots: int = 20,
    dropout: float = 0.20,
) -> nn.Module:
    """Create a small or large detector with the shared output contract."""
    config = {
        "num_classes": num_classes,
        "num_slots": num_slots,
        "dropout": dropout,
    }
    if model_name == "small":
        return SmallMNISTDetector(**config)
    if model_name == "large":
        return MNISTDetector(**config)
    raise ValueError(f"Unknown model name {model_name!r}; expected one of {MODEL_NAMES}.")


def checkpoint_model_name(checkpoint: dict[str, Any]) -> str:
    """Read architecture metadata, treating metadata-free checkpoints as large."""
    model_name = str(checkpoint.get("model_name", "large"))
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Checkpoint contains unknown model name: {model_name!r}.")
    return model_name


def build_detector_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[nn.Module, str]:
    """Construct the architecture described by a new or legacy checkpoint."""
    model_name = checkpoint_model_name(checkpoint)
    model_config = checkpoint.get(
        "model_config",
        {"num_classes": 10, "num_slots": 20, "dropout": 0.20},
    )
    return build_detector(model_name, **model_config), model_name


def default_output_dir(model_name: str) -> Path:
    if model_name == "small":
        return Path("outputs/mnist_detector_small")
    if model_name == "large":
        return Path("outputs/mnist_detector")
    raise ValueError(f"Unknown model name {model_name!r}; expected one of {MODEL_NAMES}.")
