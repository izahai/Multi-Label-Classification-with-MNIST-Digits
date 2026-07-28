"""Hungarian matching and set-prediction losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .detection_utils import generalized_box_iou, hungarian_assignment


class HungarianDetectionCriterion(nn.Module):
    """Match 20 slots to targets and compute detection losses."""

    def __init__(
        self,
        class_cost: float = 1.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
        objectness_loss_weight: float = 1.0,
        class_loss_weight: float = 1.0,
        bbox_loss_weight: float = 5.0,
        giou_loss_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.objectness_loss_weight = objectness_loss_weight
        self.class_loss_weight = class_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.giou_loss_weight = giou_loss_weight

    @torch.no_grad()
    def match(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        probabilities = outputs["class_logits"].softmax(dim=-1)
        assignments: list[tuple[torch.Tensor, torch.Tensor]] = []
        for sample, target in enumerate(targets):
            target_labels = target["labels"]
            target_boxes = target["boxes"]
            classification_cost = -probabilities[sample][:, target_labels]
            box_cost = torch.cdist(outputs["boxes"][sample], target_boxes, p=1)
            giou_cost = -generalized_box_iou(outputs["boxes"][sample], target_boxes)
            total_cost = (
                self.class_cost * classification_cost
                + self.bbox_cost * box_cost
                + self.giou_cost * giou_cost
            )
            assignments.append(hungarian_assignment(total_cost))
        return assignments

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        assignments = self.match(outputs, targets)
        objectness_targets = torch.zeros_like(outputs["objectness_logits"])
        class_loss = outputs["class_logits"].sum() * 0.0
        bbox_loss = outputs["boxes"].sum() * 0.0
        giou_loss = outputs["boxes"].sum() * 0.0
        target_count = 0

        for sample, (prediction_indices, target_indices) in enumerate(assignments):
            if len(prediction_indices) == 0:
                continue
            objectness_targets[sample, prediction_indices] = 1.0
            matched_labels = targets[sample]["labels"][target_indices]
            matched_boxes = targets[sample]["boxes"][target_indices]
            predicted_classes = outputs["class_logits"][sample, prediction_indices]
            predicted_boxes = outputs["boxes"][sample, prediction_indices]

            class_loss = class_loss + F.cross_entropy(
                predicted_classes,
                matched_labels,
                reduction="sum",
            )
            bbox_loss = bbox_loss + F.l1_loss(
                predicted_boxes,
                matched_boxes,
                reduction="sum",
            )
            paired_giou = generalized_box_iou(predicted_boxes, matched_boxes).diagonal()
            giou_loss = giou_loss + (1.0 - paired_giou).sum()
            target_count += len(prediction_indices)

        normalizer = max(target_count, 1)
        objectness_loss = F.binary_cross_entropy_with_logits(
            outputs["objectness_logits"],
            objectness_targets,
        )
        class_loss = class_loss / normalizer
        bbox_loss = bbox_loss / normalizer
        giou_loss = giou_loss / normalizer
        total_loss = (
            self.objectness_loss_weight * objectness_loss
            + self.class_loss_weight * class_loss
            + self.bbox_loss_weight * bbox_loss
            + self.giou_loss_weight * giou_loss
        )
        return {
            "loss": total_loss,
            "objectness_loss": objectness_loss,
            "class_loss": class_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss,
        }
