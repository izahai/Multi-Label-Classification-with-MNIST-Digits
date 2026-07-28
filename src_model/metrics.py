"""Detection metrics for composite MNIST validation and testing."""

from __future__ import annotations

from collections import defaultdict

import torch

from .detection_utils import box_iou


def _integrated_average_precision(
    recall: torch.Tensor,
    precision: torch.Tensor,
) -> float:
    """Area under the monotonically interpolated precision-recall curve."""
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = torch.maximum(precision[index], precision[index + 1])
    changing = torch.nonzero(recall[1:] != recall[:-1], as_tuple=False).flatten()
    return float(
        ((recall[changing + 1] - recall[changing]) * precision[changing + 1])
        .sum()
        .item()
    )


class DetectionMetrics:
    """Accumulate detection metrics and class-presence exact matches."""

    def __init__(self, iou_threshold: float = 0.50, num_classes: int = 10) -> None:
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.predictions: list[dict[str, torch.Tensor]] = []
        self.targets: list[dict[str, torch.Tensor]] = []
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.correct_classifications = 0
        self.localized_pairs = 0
        self.iou_sum = 0.0
        self.exact_matches = 0
        self.group_match_sum = 0.0
        self.correct_per_sample_sum = 0
        self.sample_count = 0

    @staticmethod
    def _match_sample(
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        iou_threshold: float,
        require_same_class: bool,
    ) -> list[tuple[int, int, float]]:
        """Greedily match confidence-sorted detections to unused targets."""
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        target_boxes = target["boxes"]
        target_labels = target["labels"]
        if len(boxes) == 0 or len(target_boxes) == 0:
            return []

        overlaps = box_iou(boxes, target_boxes)
        used_targets: set[int] = set()
        matches: list[tuple[int, int, float]] = []
        for prediction_index in scores.argsort(descending=True).tolist():
            candidates = overlaps[prediction_index].argsort(descending=True).tolist()
            for target_index in candidates:
                overlap = float(overlaps[prediction_index, target_index].item())
                if overlap < iou_threshold:
                    break
                if target_index in used_targets:
                    continue
                if require_same_class and labels[prediction_index] != target_labels[target_index]:
                    continue
                used_targets.add(target_index)
                matches.append((prediction_index, target_index, overlap))
                break
        return matches

    def update(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        for prediction, target in zip(predictions, targets):
            prediction_cpu = {key: value.detach().cpu() for key, value in prediction.items()}
            target_cpu = {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
                "image_id": target["image_id"].detach().cpu(),
            }
            self.predictions.append(prediction_cpu)
            self.targets.append(target_cpu)

            detection_matches = self._match_sample(
                prediction_cpu,
                target_cpu,
                self.iou_threshold,
                require_same_class=True,
            )
            localized_matches = self._match_sample(
                prediction_cpu,
                target_cpu,
                self.iou_threshold,
                require_same_class=False,
            )
            correct = len(detection_matches)
            prediction_count = len(prediction_cpu["boxes"])
            target_count = len(target_cpu["boxes"])

            self.true_positives += correct
            self.false_positives += prediction_count - correct
            self.false_negatives += target_count - correct
            self.iou_sum += sum(match[2] for match in detection_matches)
            self.localized_pairs += len(localized_matches)
            self.correct_classifications += sum(
                int(prediction_cpu["labels"][prediction_index] == target_cpu["labels"][target_index])
                for prediction_index, target_index, _ in localized_matches
            )
            # Exact match is a pure multi-label classification metric. It uses
            # a 10-class presence vector and deliberately ignores boxes and
            # duplicate instances of the same digit.
            predicted_presence = torch.zeros(self.num_classes, dtype=torch.bool)
            target_presence = torch.zeros(self.num_classes, dtype=torch.bool)
            predicted_presence[prediction_cpu["labels"].unique()] = True
            target_presence[target_cpu["labels"].unique()] = True
            self.exact_matches += int(torch.equal(predicted_presence, target_presence))
            self.group_match_sum += correct / max(target_count, 1)
            self.correct_per_sample_sum += correct
            self.sample_count += 1

    def _average_precision(self, class_index: int, threshold: float) -> float | None:
        ground_truths: dict[int, torch.Tensor] = {}
        total_ground_truths = 0
        detections: list[tuple[float, int, torch.Tensor]] = []

        for image_index, (prediction, target) in enumerate(
            zip(self.predictions, self.targets)
        ):
            target_boxes = target["boxes"][target["labels"] == class_index]
            ground_truths[image_index] = target_boxes
            total_ground_truths += len(target_boxes)
            selected = prediction["labels"] == class_index
            for score, box in zip(
                prediction["scores"][selected],
                prediction["boxes"][selected],
            ):
                detections.append((float(score.item()), image_index, box))

        if total_ground_truths == 0:
            return None
        detections.sort(key=lambda item: item[0], reverse=True)
        matched: dict[int, set[int]] = defaultdict(set)
        true_positive = torch.zeros(len(detections))
        false_positive = torch.zeros(len(detections))

        for detection_index, (_, image_index, box) in enumerate(detections):
            target_boxes = ground_truths[image_index]
            if len(target_boxes) == 0:
                false_positive[detection_index] = 1
                continue
            overlaps = box_iou(box.unsqueeze(0), target_boxes).squeeze(0)
            best_overlap, best_target = overlaps.max(dim=0)
            target_index = int(best_target.item())
            if (
                float(best_overlap.item()) >= threshold
                and target_index not in matched[image_index]
            ):
                true_positive[detection_index] = 1
                matched[image_index].add(target_index)
            else:
                false_positive[detection_index] = 1

        if len(detections) == 0:
            return 0.0
        cumulative_tp = true_positive.cumsum(0)
        cumulative_fp = false_positive.cumsum(0)
        recall = cumulative_tp / total_ground_truths
        precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp(min=1e-8)
        return _integrated_average_precision(recall, precision)

    def _mean_ap(self, threshold: float) -> float:
        values = [
            value
            for class_index in range(self.num_classes)
            if (value := self._average_precision(class_index, threshold)) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    def compute(self) -> dict[str, float]:
        precision = self.true_positives / max(
            self.true_positives + self.false_positives,
            1,
        )
        recall = self.true_positives / max(
            self.true_positives + self.false_negatives,
            1,
        )
        thresholds = [0.50 + 0.05 * index for index in range(10)]
        map_values = [self._mean_ap(threshold) for threshold in thresholds]
        return {
            "precision": precision,
            "recall": recall,
            "mean_iou": self.iou_sum / max(self.true_positives, 1),
            "classification_accuracy": self.correct_classifications
            / max(self.localized_pairs, 1),
            "exact_match": self.exact_matches / max(self.sample_count, 1),
            "binary_group_match": self.group_match_sum / max(self.sample_count, 1),
            "correct_digits_per_sample": self.correct_per_sample_sum
            / max(self.sample_count, 1),
            "map50": map_values[0],
            "map50_95": sum(map_values) / len(map_values),
        }
