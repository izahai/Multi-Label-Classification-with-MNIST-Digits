"""Evaluate presence-vector classification on data/uni train, val, and test.

This evaluator ignores bounding boxes completely. It converts retained model
slots into a 10-element binary digit-presence vector, then compares it with the
dataset's `labels` multi-hot vector.

Exact match: all 10 binary entries are correct for one sample.
Binary match: percentage of individual binary entries that are correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model_factory import build_detector_from_checkpoint
from .train import select_device


class UniClassificationDataset(Dataset):
    """Read images and multi-hot digit-presence labels from one original split."""

    def __init__(self, path: Path, max_samples: int | None = None) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {path}")
        self.data = torch.load(path, map_location="cpu", weights_only=True)
        required = {"images", "labels"}
        missing = required - self.data.keys()
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        available = len(self.data["images"])
        self.length = available if max_samples is None else min(max_samples, available)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data["images"][index].float().unsqueeze(0) / 255.0
        presence = self.data["labels"][index].bool()
        return image, presence


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    confidence_threshold: float,
) -> dict[str, float | int]:
    """Compute exact and per-bit presence-vector matches for one split."""
    exact_matches = 0
    correct_bits = 0
    total_bits = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    predicted_digits = 0
    sample_count = 0

    model.eval()
    for batch_index, (images, true_presence) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        class_probabilities = outputs["class_logits"].softmax(dim=-1)
        class_confidence, predicted_labels = class_probabilities.max(dim=-1)
        slot_scores = outputs["confidence_scores"] * class_confidence
        retained_slots = slot_scores >= confidence_threshold

        predicted_presence_counts = torch.zeros(
            (len(images), model.num_classes),
            device=device,
            dtype=torch.int64,
        )
        predicted_presence_counts.scatter_add_(
            1,
            predicted_labels,
            retained_slots.to(torch.int64),
        )
        predicted_presence = predicted_presence_counts > 0
        true_presence = true_presence.to(device, non_blocking=True)

        exact_matches += int((predicted_presence == true_presence).all(dim=1).sum().item())
        correct_bits += int((predicted_presence == true_presence).sum().item())
        total_bits += true_presence.numel()
        true_positives += int((predicted_presence & true_presence).sum().item())
        false_positives += int((predicted_presence & ~true_presence).sum().item())
        false_negatives += int((~predicted_presence & true_presence).sum().item())
        predicted_digits += int(retained_slots.sum().item())
        sample_count += len(images)

        if batch_index == len(loader) or batch_index % 50 == 0:
            print(f"\r  {batch_index:,}/{len(loader):,} batches", end="", flush=True)
    print()

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return {
        "samples": sample_count,
        "exact_match": exact_matches / max(sample_count, 1),
        "binary_match": correct_bits / max(total_bits, 1),
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "average_retained_slots": predicted_digits / max(sample_count, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/mnist_detector_small/best.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("--confidence-threshold must be between 0 and 1.")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, model_name = build_detector_from_checkpoint(checkpoint)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Confidence threshold: {args.confidence_threshold:.2f}")

    all_metrics: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "test"):
        print(f"\nEvaluating {split}.pt")
        dataset = UniClassificationDataset(
            args.data_dir / f"{split}.pt",
            args.max_samples,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate_split(model, loader, device, args.confidence_threshold)
        all_metrics[split] = metrics
        print(
            f"{split.upper()} ({metrics['samples']:,} samples): "
            f"exact={metrics['exact_match']:.2%} | "
            f"binary={metrics['binary_match']:.2%} | "
            f"precision={metrics['presence_precision']:.2%} | "
            f"recall={metrics['presence_recall']:.2%} | "
            f"retained slots/sample={metrics['average_retained_slots']:.2f}"
        )

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        print(f"\nSaved metrics to {args.json_output.resolve()}")


if __name__ == "__main__":
    main()
