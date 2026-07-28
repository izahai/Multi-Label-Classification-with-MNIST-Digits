"""YOLO-style grid detectors for 64×64 composite MNIST images."""

from __future__ import annotations

import torch
from torch import nn

from .mnist_detector import ConvBlock


class YOLOMNISTDetector(nn.Module):
    """Anchor-free 8×8 grid detector with three prediction slots per cell."""

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int = 10,
        anchors_per_cell: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = 8
        self.anchors_per_cell = anchors_per_cell
        self.num_slots = self.grid_size * self.grid_size * anchors_per_cell
        self.values_per_slot = 1 + num_classes + 4

        self.backbone = nn.Sequential(
            ConvBlock(1, channels[0]),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[2]),
        )
        self.prediction_head = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                channels[2],
                anchors_per_cell * self.values_per_slot,
                kernel_size=1,
            ),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Initial low objectness reduces false positives before the first update.
        final_layer = self.prediction_head[-1]
        final_layer.bias.data.view(self.anchors_per_cell, self.values_per_slot)[:, 0] = -2.0

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"Expected (batch, 1, height, width), got {tuple(images.shape)}.")
        raw = self.prediction_head(self.backbone(images))
        batch_size = len(images)
        raw = raw.view(
            batch_size,
            self.anchors_per_cell,
            self.values_per_slot,
            self.grid_size,
            self.grid_size,
        ).permute(0, 3, 4, 1, 2).contiguous()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_size, device=images.device),
            torch.arange(self.grid_size, device=images.device),
            indexing="ij",
        )
        offsets = raw[..., 11:13].sigmoid()
        centers = torch.stack(
            ((grid_x.unsqueeze(-1) + offsets[..., 0]) / self.grid_size,
             (grid_y.unsqueeze(-1) + offsets[..., 1]) / self.grid_size),
            dim=-1,
        )
        sizes = raw[..., 13:15].sigmoid()
        boxes = torch.cat(
            ((centers - sizes / 2).clamp(0, 1), (centers + sizes / 2).clamp(0, 1)),
            dim=-1,
        )
        return {
            "objectness_logits": raw[..., 0].reshape(batch_size, -1),
            "confidence_scores": raw[..., 0].sigmoid().reshape(batch_size, -1),
            "class_logits": raw[..., 1:11].reshape(batch_size, -1, self.num_classes),
            "boxes": boxes.reshape(batch_size, -1, 4),
            "yolo_raw": raw,
        }


class YOLOSmallMNISTDetector(YOLOMNISTDetector):
    def __init__(self, num_classes: int = 10, num_slots: int = 20, dropout: float = 0.20) -> None:
        if num_slots != 20:
            raise ValueError("YOLO detector uses a fixed 8×8×3 = 192 prediction slots.")
        super().__init__((16, 32, 64), num_classes=num_classes, dropout=dropout)


class YOLOLargeMNISTDetector(YOLOMNISTDetector):
    def __init__(self, num_classes: int = 10, num_slots: int = 20, dropout: float = 0.20) -> None:
        if num_slots != 20:
            raise ValueError("YOLO detector uses a fixed 8×8×3 = 192 prediction slots.")
        super().__init__((32, 64, 128), num_classes=num_classes, dropout=dropout)
