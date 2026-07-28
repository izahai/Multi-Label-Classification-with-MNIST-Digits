"""Dataset loader for generated composite MNIST detection files."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CompositeMNISTDetection(Dataset):
    """Read one `.pt` split and expose variable-length detection targets."""

    def __init__(self, path: str | Path, max_samples: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {self.path}")

        self.data = torch.load(self.path, map_location="cpu", weights_only=True)
        required = {"images", "num_digits", "all_bboxes", "bbox_labels"}
        missing = required - self.data.keys()
        if missing:
            raise ValueError(f"{self.path} is missing keys: {sorted(missing)}")

        self.image_size = int(self.data.get("image_size", self.data["images"].shape[-1]))
        available = len(self.data["images"])
        self.length = available if max_samples is None else min(max_samples, available)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        count = int(self.data["num_digits"][index].item())
        image = self.data["images"][index].float().unsqueeze(0) / 255.0
        boxes = self.data["all_bboxes"][index, :count].float() / self.image_size
        labels = self.data["bbox_labels"][index, :count].long()

        target = {
            "boxes": boxes.clamp(0.0, 1.0),
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
        }
        return image, target


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Stack equal-sized images while retaining variable-length targets."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
