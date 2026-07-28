"""GPU-native grid assignment and losses for the YOLO-style detector."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class YOLODetectionCriterion(nn.Module):
    """Assign each box to its centre cell and a free slot without Hungarian matching."""

    def __init__(self, objectness_weight: float = 1.0, class_weight: float = 1.0, box_weight: float = 5.0) -> None:
        super().__init__()
        self.objectness_weight = objectness_weight
        self.class_weight = class_weight
        self.box_weight = box_weight

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        raw = outputs["yolo_raw"]
        batch_size, grid_size, _, anchors, _ = raw.shape
        object_targets = torch.zeros_like(raw[..., 0])
        class_targets = torch.full_like(raw[..., 0], -1, dtype=torch.long)
        box_targets = torch.zeros_like(raw[..., 11:15])

        for batch_index, target in enumerate(targets):
            boxes, labels = target["boxes"], target["labels"]
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
            cell_x = (centers[:, 0] * grid_size).long().clamp(0, grid_size - 1)
            cell_y = (centers[:, 1] * grid_size).long().clamp(0, grid_size - 1)
            used = torch.zeros((grid_size, grid_size), dtype=torch.long, device=raw.device)
            for box_index in range(len(boxes)):
                y, x = int(cell_y[box_index]), int(cell_x[box_index])
                slot = int(used[y, x].item())
                if slot >= anchors:
                    raise RuntimeError("More objects than YOLO slots in one grid cell.")
                used[y, x] += 1
                object_targets[batch_index, y, x, slot] = 1
                class_targets[batch_index, y, x, slot] = labels[box_index]
                box_targets[batch_index, y, x, slot, :2] = centers[box_index] * grid_size - torch.tensor((x, y), device=raw.device)
                box_targets[batch_index, y, x, slot, 2:] = sizes[box_index]

        positive = object_targets.bool()
        objectness_loss = F.binary_cross_entropy_with_logits(raw[..., 0], object_targets, pos_weight=torch.tensor(10.0, device=raw.device))
        if positive.any():
            class_loss = F.cross_entropy(raw[..., 1:11][positive], class_targets[positive])
            predicted_box = raw[..., 11:15].sigmoid()
            box_loss = F.mse_loss(predicted_box[positive], box_targets[positive])
        else:
            class_loss = raw.sum() * 0
            box_loss = raw.sum() * 0
        loss = self.objectness_weight * objectness_loss + self.class_weight * class_loss + self.box_weight * box_loss
        return {"loss": loss, "objectness_loss": objectness_loss, "class_loss": class_loss, "bbox_loss": box_loss, "giou_loss": raw.sum() * 0}
