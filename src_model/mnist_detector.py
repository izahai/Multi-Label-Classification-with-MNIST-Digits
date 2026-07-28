"""Large residual CNN detector for composite 64×64 MNIST images.

The detector produces 20 unordered object slots. Each slot predicts:

- one objectness logit/confidence;
- logits for digit classes 0–9;
- one normalized bounding box in xyxy-exclusive format.

During training, the slots should be matched to ground-truth objects with a
set-based assignment method such as Hungarian matching.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Residual two-convolution block followed by 2× spatial downsampling.

    The shortcut uses parameter-free channel padding or slicing, so the
    original convolution parameter names remain stable for old checkpoints.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _shortcut(self, images: torch.Tensor) -> torch.Tensor:
        if self.in_channels == self.out_channels:
            return images
        if self.in_channels > self.out_channels:
            return images[:, : self.out_channels]
        padding = images.new_zeros(
            images.shape[0],
            self.out_channels - self.in_channels,
            images.shape[2],
            images.shape[3],
        )
        return torch.cat((images, padding), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = self._shortcut(images)
        features = self[0](images)
        features = self[1](features)
        features = self[2](features)
        features = self[3](features)
        features = self[4](features)
        features = self[5](features + residual)
        return self[6](features)


class MNISTDetector(nn.Module):
    """Predict digit classes, confidence scores, and boxes in 20 slots."""

    def __init__(
        self,
        num_classes: int = 10,
        num_slots: int = 20,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots

        # Input: (batch, 1, 64, 64)
        self.backbone = nn.Sequential(
            ConvBlock(1, 32),      # -> (batch, 32, 32, 32)
            ConvBlock(32, 64),     # -> (batch, 64, 16, 16)
            ConvBlock(64, 128),    # -> (batch, 128, 8, 8)
            ConvBlock(128, 256),   # -> (batch, 256, 4, 4)
        )

        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_slots * 128),
            nn.SiLU(inplace=True),
        )

        self.objectness_head = nn.Linear(128, 1)
        self.class_head = nn.Linear(128, num_classes)
        self.box_head = nn.Linear(128, 4)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Begin with low object confidence so empty slots do not dominate early.
        nn.init.constant_(self.objectness_head.bias, -2.0)

    @staticmethod
    def _to_xyxy(raw_boxes: torch.Tensor) -> torch.Tensor:
        """Convert unconstrained predictions to normalized, valid xyxy boxes."""
        center = raw_boxes[..., :2].sigmoid()
        size = raw_boxes[..., 2:].sigmoid()
        top_left = (center - size / 2).clamp(0.0, 1.0)
        bottom_right = (center + size / 2).clamp(0.0, 1.0)
        return torch.cat((top_left, bottom_right), dim=-1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run detection on normalized grayscale images.

        Args:
            images: Float tensor shaped ``(batch, 1, 64, 64)``, normally scaled
                to the range [0, 1].

        Returns:
            A dictionary containing:

            - ``objectness_logits``: ``(batch, 20)`` for training with BCE.
            - ``confidence_scores``: ``(batch, 20)`` in the range [0, 1].
            - ``class_logits``: ``(batch, 20, 10)`` for cross-entropy.
            - ``boxes``: ``(batch, 20, 4)`` normalized xyxy boxes.
        """
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                "Expected images with shape (batch, 1, height, width), "
                f"but received {tuple(images.shape)}."
            )

        features = self.backbone(images)
        slots = self.shared_head(features).reshape(
            images.shape[0],
            self.num_slots,
            128,
        )

        objectness_logits = self.objectness_head(slots).squeeze(-1)
        class_logits = self.class_head(slots)
        boxes = self._to_xyxy(self.box_head(slots))

        return {
            "objectness_logits": objectness_logits,
            "confidence_scores": objectness_logits.sigmoid(),
            "class_logits": class_logits,
            "boxes": boxes,
        }


if __name__ == "__main__":
    model = MNISTDetector()
    sample = torch.rand(2, 1, 64, 64)
    predictions = model(sample)
    for name, tensor in predictions.items():
        print(f"{name}: {tuple(tensor.shape)}")
