"""Bounding-box, assignment, and post-processing utilities."""

from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return sizes[..., 0] * sizes[..., 1]


def box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized xyxy boxes."""
    intersection_top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    intersection_bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = (intersection_bottom_right - intersection_top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = box_area(first)[:, None] + box_area(second)[None, :] - intersection
    return intersection / union.clamp(min=1e-8)


def generalized_box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Pairwise generalized IoU for normalized xyxy boxes."""
    iou = box_iou(first, second)
    enclosing_top_left = torch.minimum(first[:, None, :2], second[None, :, :2])
    enclosing_bottom_right = torch.maximum(first[:, None, 2:], second[None, :, 2:])
    enclosing_size = (enclosing_bottom_right - enclosing_top_left).clamp(min=0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]

    intersection_top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    intersection_bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = (intersection_bottom_right - intersection_top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = box_area(first)[:, None] + box_area(second)[None, :] - intersection
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-8)


def hungarian_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize a rectangular `(predictions, targets)` cost matrix.

    This is the shortest-augmenting-path Hungarian algorithm. The generated
    dataset has at most eight targets and the model has twenty predictions, so
    every target receives one unique prediction.
    """
    num_predictions, num_targets = cost.shape
    device = cost.device
    if num_targets == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    if num_targets > num_predictions:
        raise ValueError("Hungarian matching requires at least as many slots as targets.")

    matrix = cost.detach().float().cpu().t().tolist()  # targets × predictions
    rows, columns = num_targets, num_predictions
    row_potential = [0.0] * (rows + 1)
    column_potential = [0.0] * (columns + 1)
    matching = [0] * (columns + 1)
    previous = [0] * (columns + 1)

    for row in range(1, rows + 1):
        matching[0] = row
        current_column = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[current_column] = True
            current_row = matching[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced_cost = (
                    matrix[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum[column]:
                    minimum[column] = reduced_cost
                    previous[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column

            for column in range(columns + 1):
                if used[column]:
                    row_potential[matching[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matching[current_column] == 0:
                break

        while True:
            next_column = previous[current_column]
            matching[current_column] = matching[next_column]
            current_column = next_column
            if current_column == 0:
                break

    target_to_prediction = [-1] * rows
    for column in range(1, columns + 1):
        if matching[column] != 0:
            target_to_prediction[matching[column] - 1] = column - 1

    target_indices = torch.arange(rows, dtype=torch.long, device=device)
    prediction_indices = torch.tensor(
        target_to_prediction,
        dtype=torch.long,
        device=device,
    )
    return prediction_indices, target_indices


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Pure-PyTorch non-maximum suppression."""
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        overlaps = box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        order = remaining[overlaps <= threshold]
    return torch.stack(kept)


def postprocess_detections(
    outputs: dict[str, torch.Tensor],
    confidence_threshold: float,
    nms_threshold: float,
    max_detections: int = 20,
) -> list[dict[str, torch.Tensor]]:
    """Convert raw slots into filtered, class-aware detections."""
    class_probabilities = outputs["class_logits"].softmax(dim=-1)
    class_scores, labels = class_probabilities.max(dim=-1)
    scores = outputs["confidence_scores"] * class_scores
    results: list[dict[str, torch.Tensor]] = []

    for sample in range(len(scores)):
        selected = scores[sample] >= confidence_threshold
        sample_boxes = outputs["boxes"][sample][selected]
        sample_scores = scores[sample][selected]
        sample_labels = labels[sample][selected]
        kept_by_class: list[torch.Tensor] = []

        for label in sample_labels.unique():
            class_indices = torch.nonzero(sample_labels == label, as_tuple=False).flatten()
            class_keep = nms(
                sample_boxes[class_indices],
                sample_scores[class_indices],
                nms_threshold,
            )
            kept_by_class.append(class_indices[class_keep])

        if kept_by_class:
            keep = torch.cat(kept_by_class)
            keep = keep[sample_scores[keep].argsort(descending=True)[:max_detections]]
        else:
            keep = torch.empty(0, dtype=torch.long, device=sample_boxes.device)

        results.append(
            {
                "boxes": sample_boxes[keep],
                "scores": sample_scores[keep],
                "labels": sample_labels[keep],
            }
        )
    return results
