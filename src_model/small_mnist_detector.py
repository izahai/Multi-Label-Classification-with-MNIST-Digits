"""Compact residual 20-slot detector with about 10% of large parameters."""

from __future__ import annotations

import torch
from torch import nn

from .mnist_detector import ConvBlock


class SmallMNISTDetector(nn.Module):
    """Predict the same outputs as MNISTDetector with substantially less capacity."""

    def __init__(
        self,
        num_classes: int = 10,
        num_slots: int = 20,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.slot_features = 64

        self.backbone = nn.Sequential(
            ConvBlock(1, 16),     # -> (batch, 16, 32, 32)
            ConvBlock(16, 32),   # -> (batch, 32, 16, 16)
            ConvBlock(32, 64),   # -> (batch, 64, 8, 8)
            ConvBlock(64, 96),   # -> (batch, 96, 4, 4)
        )
        self.shared_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(96 * 2 * 2, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_slots * self.slot_features),
            nn.SiLU(inplace=True),
        )
        self.objectness_head = nn.Linear(self.slot_features, 1)
        self.class_head = nn.Linear(self.slot_features, num_classes)
        self.box_head = nn.Linear(self.slot_features, 4)
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
        nn.init.constant_(self.objectness_head.bias, -2.0)

    @staticmethod
    def _to_xyxy(raw_boxes: torch.Tensor) -> torch.Tensor:
        center = raw_boxes[..., :2].sigmoid()
        size = raw_boxes[..., 2:].sigmoid()
        top_left = (center - size / 2).clamp(0.0, 1.0)
        bottom_right = (center + size / 2).clamp(0.0, 1.0)
        return torch.cat((top_left, bottom_right), dim=-1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                "Expected images with shape (batch, 1, height, width), "
                f"but received {tuple(images.shape)}."
            )

        features = self.backbone(images)
        slots = self.shared_head(features).reshape(
            images.shape[0],
            self.num_slots,
            self.slot_features,
        )
        objectness_logits = self.objectness_head(slots).squeeze(-1)
        return {
            "objectness_logits": objectness_logits,
            "confidence_scores": objectness_logits.sigmoid(),
            "class_logits": self.class_head(slots),
            "boxes": self._to_xyxy(self.box_head(slots)),
        }


if __name__ == "__main__":
    model = SmallMNISTDetector()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameters:,}")
    for name, tensor in model(torch.rand(2, 1, 64, 64)).items():
        print(f"{name}: {tuple(tensor.shape)}")
