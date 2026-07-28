"""Dataset utilities for composite-MNIST multi-label classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset


class CompositeMNISTClassification(Dataset):
    """Load images and 10-class presence labels from one ``.pt`` split.

    Extra fields such as bounding boxes, positions, and duplicate digit labels
    are deliberately ignored.
    """

    def __init__(
        self,
        path: str | Path,
        max_samples: int | None = None,
        source: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.source = source or self.path.parent.name
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {self.path}")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive when provided.")

        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        missing = {"images", "labels"} - payload.keys()
        if missing:
            raise ValueError(f"{self.path} is missing keys: {sorted(missing)}")

        images = payload["images"]
        labels = payload["labels"]
        if images.ndim != 3 or tuple(images.shape[1:]) != (64, 64):
            raise ValueError(
                f"{self.path}: expected images shaped (N, 64, 64), "
                f"got {tuple(images.shape)}."
            )
        if labels.ndim != 2 or labels.shape[1] != 10:
            raise ValueError(
                f"{self.path}: expected labels shaped (N, 10), "
                f"got {tuple(labels.shape)}."
            )
        if len(images) != len(labels):
            raise ValueError(
                f"{self.path}: images and labels have different lengths "
                f"({len(images)} != {len(labels)})."
            )

        length = len(images) if max_samples is None else min(max_samples, len(images))
        self.images = images[:length]
        self.labels = labels[:length]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].to(torch.float32).unsqueeze(0).div_(255.0)
        target = self.labels[index].to(torch.float32)
        return image, target


def load_split(
    data_dir: str | Path,
    split: str,
    *,
    max_samples: int | None = None,
    source: str | None = None,
) -> CompositeMNISTClassification:
    """Load one named split from a dataset directory."""
    return CompositeMNISTClassification(
        Path(data_dir) / f"{split}.pt",
        max_samples=max_samples,
        source=source,
    )


def load_combined_train(
    original_data_dir: str | Path,
    bbox_data_dir: str | Path,
    *,
    max_samples_per_source: int | None = None,
) -> ConcatDataset:
    """Concatenate only the two training splits."""
    return ConcatDataset(
        [
            load_split(
                original_data_dir,
                "train",
                max_samples=max_samples_per_source,
                source="original",
            ),
            load_split(
                bbox_data_dir,
                "train",
                max_samples=max_samples_per_source,
                source="bbox",
            ),
        ]
    )
