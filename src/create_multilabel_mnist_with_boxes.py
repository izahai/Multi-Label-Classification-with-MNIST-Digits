"""Create composite MNIST data for multi-label classification and detection.

Each 64×64 image contains 6, 7, or 8 MNIST digits. The number of examples for
each count is balanced (it differs by at most one sample), and every digit has
a ground-truth axis-aligned bounding box and class label.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from tqdm import tqdm
except ImportError:  # Keep generation usable when tqdm is not installed.
    def tqdm(
        iterable: object,
        *,
        total: int,
        desc: str,
        unit: str,
        **_: object,
    ) -> object:
        """Small dependency-free progress bar used when tqdm is unavailable."""
        bar_width = 30
        update_every = max(1, total // 200)
        for completed, item in enumerate(iterable, start=1):
            if completed == 1 or completed == total or completed % update_every == 0:
                filled = round(bar_width * completed / total)
                bar = "#" * filled + "-" * (bar_width - filled)
                print(
                    f"\r{desc}: [{bar}] {completed:,}/{total:,} {unit}s",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            yield item
        print(file=sys.stderr)


IMAGE_SIZE = 64
DIGIT_SIZE = 28
MAX_DIGITS = 8
BOX_FORMAT = "xyxy_exclusive"  # (x_min, y_min, x_max, y_max)


def balanced_digit_counts(num_samples: int) -> torch.Tensor:
    """Return a shuffled sequence of 6, 7, and 8 with nearly equal frequency."""
    base, remainder = divmod(num_samples, 3)
    counts = torch.tensor([6] * base + [7] * base + [8] * base)
    if remainder:
        counts = torch.cat((counts, torch.tensor([6, 7][:remainder])))
    return counts[torch.randperm(num_samples)]


def make_digit_patch(
    source_images: torch.Tensor,
    index: int,
    min_rotation_degrees: float,
    max_rotation_degrees: float,
    foreground_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return a rotated uint8 patch, its foreground mask, and rotation angle."""
    image = source_images[index].unsqueeze(0).float() / 255.0
    angle = float(torch.empty(1).uniform_(min_rotation_degrees, max_rotation_degrees).item())
    rotated = TF.rotate(
        image,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    ).squeeze(0).clamp(0, 1)
    mask = rotated >= foreground_threshold
    patch = (rotated * 255).round().to(torch.uint8)
    patch[~mask] = 0
    return patch, mask, angle


def random_position() -> tuple[int, int]:
    """Choose a fully in-bounds top-left position as (row, column)."""
    limit = IMAGE_SIZE - DIGIT_SIZE
    return (
        int(torch.randint(0, limit + 1, ()).item()),
        int(torch.randint(0, limit + 1, ()).item()),
    )


def pixel_overlap_ratio(
    first_mask: torch.Tensor,
    first_position: tuple[int, int],
    second_mask: torch.Tensor,
    second_position: tuple[int, int],
) -> float:
    """Calculate foreground intersection / smaller foreground pixel count."""
    first_row, first_column = first_position
    second_row, second_column = second_position
    top = max(first_row, second_row)
    left = max(first_column, second_column)
    bottom = min(first_row + DIGIT_SIZE, second_row + DIGIT_SIZE)
    right = min(first_column + DIGIT_SIZE, second_column + DIGIT_SIZE)
    if top >= bottom or left >= right:
        return 0.0

    first_region = first_mask[
        top - first_row : bottom - first_row,
        left - first_column : right - first_column,
    ]
    second_region = second_mask[
        top - second_row : bottom - second_row,
        left - second_column : right - second_column,
    ]
    intersection = int(torch.logical_and(first_region, second_region).sum().item())
    smaller_area = min(int(first_mask.sum().item()), int(second_mask.sum().item()))
    return intersection / smaller_area if smaller_area else 0.0


def low_overlap_position(
    digit_mask: torch.Tensor,
    placed_digits: list[tuple[torch.Tensor, tuple[int, int]]],
    max_overlap_ratio: float,
    attempts: int,
) -> tuple[int, int] | None:
    """Randomly search for a position satisfying the foreground overlap limit."""
    for _ in range(attempts):
        candidate = random_position()
        ratios = [
            pixel_overlap_ratio(digit_mask, candidate, placed_mask, placed_position)
            for placed_mask, placed_position in placed_digits
        ]
        if all(ratio <= max_overlap_ratio for ratio in ratios):
            return candidate
    return None


def tight_bbox(mask: torch.Tensor, position: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return the global xyxy-exclusive box around actual foreground pixels."""
    foreground = mask.nonzero(as_tuple=False)
    if len(foreground) == 0:
        raise ValueError("A digit has no foreground pixels at the selected threshold.")
    row, column = position
    y_min, x_min = foreground.min(dim=0).values.tolist()
    y_max, x_max = foreground.max(dim=0).values.tolist()
    return column + x_min, row + y_min, column + x_max + 1, row + y_max + 1


def compose_split(
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    num_samples: int,
    max_overlap_ratio: float,
    placement_attempts: int,
    digit_attempts: int,
    min_rotation_degrees: float,
    max_rotation_degrees: float,
    foreground_threshold: float,
    split_name: str,
) -> dict[str, object]:
    """Generate one split while retaining labels, positions, and boxes per digit."""
    images = torch.zeros((num_samples, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    labels = torch.zeros((num_samples, 10), dtype=torch.float32)
    center_labels = torch.empty(num_samples, dtype=torch.int64)
    count_labels = torch.zeros((num_samples, 10), dtype=torch.int64)
    num_digits = balanced_digit_counts(num_samples)
    all_digit_labels = torch.full((num_samples, MAX_DIGITS), -1, dtype=torch.int64)
    all_positions = torch.full((num_samples, MAX_DIGITS, 2), -1, dtype=torch.int64)
    all_bboxes = torch.full((num_samples, MAX_DIGITS, 4), -1, dtype=torch.int64)
    bbox_labels = torch.full((num_samples, MAX_DIGITS), -1, dtype=torch.int64)
    all_rotation_degrees = torch.full((num_samples, MAX_DIGITS), float("nan"), dtype=torch.float32)

    progress = tqdm(
        enumerate(num_digits.tolist()),
        total=num_samples,
        desc=f"Generating {split_name}",
        unit="sample",
        dynamic_ncols=True,
    )
    for sample_index, digit_count in progress:
        placed_digits: list[tuple[torch.Tensor, tuple[int, int]]] = []

        for slot in range(digit_count):
            for _ in range(digit_attempts):
                source_index = int(torch.randint(len(source_images), ()).item())
                digit_label = int(source_labels[source_index].item())
                patch, mask, angle = make_digit_patch(
                    source_images,
                    source_index,
                    min_rotation_degrees,
                    max_rotation_degrees,
                    foreground_threshold,
                )
                position = low_overlap_position(
                    mask,
                    placed_digits,
                    max_overlap_ratio=max_overlap_ratio,
                    attempts=placement_attempts,
                )
                if position is not None:
                    break
            else:
                raise RuntimeError(
                    f"Could not place digit {slot + 1}/{digit_count} in sample "
                    f"{sample_index} below overlap ratio {max_overlap_ratio}. "
                    "Increase --placement-attempts/--digit-attempts or relax "
                    "--max-overlap-ratio."
                )

            row, column = position
            target = images[sample_index, row : row + DIGIT_SIZE, column : column + DIGIT_SIZE]
            # Only foreground pixels are drawn; later digits overwrite earlier
            # digit pixels wherever their foreground masks intersect.
            target[mask] = patch[mask]

            all_digit_labels[sample_index, slot] = digit_label
            bbox_labels[sample_index, slot] = digit_label
            all_positions[sample_index, slot] = torch.tensor((row, column))
            all_bboxes[sample_index, slot] = torch.tensor(tight_bbox(mask, position))
            all_rotation_degrees[sample_index, slot] = angle
            placed_digits.append((mask, position))
            labels[sample_index, digit_label] = 1.0
            count_labels[sample_index, digit_label] += 1

        center_labels[sample_index] = all_digit_labels[sample_index, 0]

    return {
        "images": images,
        "labels": labels,
        "center_labels": center_labels,
        "count_labels": count_labels,
        "num_digits": num_digits,
        "all_digit_labels": all_digit_labels,
        "all_positions": all_positions,
        "all_bboxes": all_bboxes,
        "bbox_labels": bbox_labels,
        "all_rotation_degrees": all_rotation_degrees,
        "image_size": IMAGE_SIZE,
        "digit_size": DIGIT_SIZE,
        "center_position": None,
        "num_classes": 10,
        "min_digits": 6,
        "max_digits": 8,
        "max_pairwise_overlap_ratio": max_overlap_ratio,
        "overlap_measure": "foreground_intersection_over_smaller_foreground_area",
        "placement_attempts": placement_attempts,
        "digit_attempts": digit_attempts,
        "foreground_threshold": foreground_threshold,
        "use_random_rotation": True,
        "rotation_degrees": (min_rotation_degrees, max_rotation_degrees),
        "compositing": "later_foreground_pixels_overwrite_earlier_pixels",
        "label_type": "multi_hot_digit_presence",
        "bbox_format": BOX_FORMAT,
        "bbox_label_key": "bbox_labels",
        "unused_slot_value": -1,
        "difficulty": "pixel_controlled_overlap_rotated_digits_with_detection_targets",
    }


def write_train_summary(output_dir: Path, train_size: int) -> None:
    """Write a concise schema document alongside the generated train split."""
    summary = f"""# `train.pt` structure with bounding boxes

`train.pt` contains **{train_size:,}** synthetic 64×64 MNIST composite images. Every sample contains 6, 7, or 8 digits; the generator balances these three values as evenly as possible.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `images` | `uint8`, `({train_size}, 64, 64)` | Grayscale composite images. |
| `labels` | `float32`, `({train_size}, 10)` | Multi-hot digit-presence target for classes 0–9. |
| `center_labels` | `int64`, `({train_size},)` | Legacy compatibility key containing the label in slot 0. |
| `count_labels` | `int64`, `({train_size}, 10)` | Number of appearances of each class. |
| `num_digits` | `int64`, `({train_size},)` | Number of digit instances: 6, 7, or 8. |
| `all_digit_labels` | `int64`, `({train_size}, 8)` | Class label for each instance slot. |
| `all_positions` | `int64`, `({train_size}, 8, 2)` | Each patch's top-left `(row, column)`. |
| `all_bboxes` | `int64`, `({train_size}, 8, 4)` | Tight foreground boxes `(x_min, y_min, x_max, y_max)` after rotation, with exclusive maxima. |
| `bbox_labels` | `int64`, `({train_size}, 8)` | Class label paired with the box at the same slot in `all_bboxes`. |
| `all_rotation_degrees` | `float32`, `({train_size}, 8)` | Random rotation angle used for each digit. |

Slots from `num_digits` through slot 7 are padding and use `-1` in the label, position, and box tensors and `NaN` in `all_rotation_degrees`. Every digit is randomly placed. Pairwise overlap is measured using actual rotated foreground masks as `intersection / smaller foreground area`. Later digit foreground pixels overwrite earlier digit pixels.
"""
    (output_dir / "train_structure.md").write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/uni_with_bboxes"))
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--val-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-overlap-ratio",
        type=float,
        default=0.20,
        help="Maximum pairwise foreground intersection/smaller-area ratio (default: 0.20).",
    )
    parser.add_argument(
        "--placement-attempts",
        type=int,
        default=200,
        help="Random candidate positions evaluated per digit attempt (default: 200).",
    )
    parser.add_argument(
        "--digit-attempts",
        type=int,
        default=20,
        help="New source-digit/rotation attempts when placement fails (default: 20).",
    )
    parser.add_argument(
        "--min-rotation-degrees",
        type=float,
        default=-30.0,
        help="Minimum random rotation angle (default: -30).",
    )
    parser.add_argument(
        "--max-rotation-degrees",
        type=float,
        default=30.0,
        help="Maximum random rotation angle (default: 30).",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.05,
        help="Normalized pixel threshold defining digit foreground (default: 0.05).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.train_size,
        args.val_size,
        args.test_size,
        args.placement_attempts,
        args.digit_attempts,
    ) < 1:
        raise ValueError("Split sizes and attempt counts must be positive.")
    if not 0 <= args.max_overlap_ratio <= 1:
        raise ValueError("--max-overlap-ratio must be between 0 and 1.")
    if not 0 <= args.foreground_threshold <= 1:
        raise ValueError("--foreground-threshold must be between 0 and 1.")
    if args.min_rotation_degrees > args.max_rotation_degrees:
        raise ValueError("--min-rotation-degrees cannot exceed --max-rotation-degrees.")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mnist_train = MNIST(args.mnist_dir, train=True, download=True)
    mnist_test = MNIST(args.mnist_dir, train=False, download=True)
    train_images, val_images = mnist_train.data[:50_000], mnist_train.data[50_000:]
    train_labels, val_labels = mnist_train.targets[:50_000], mnist_train.targets[50_000:]

    for split, source_images, source_labels, size in (
        ("train", train_images, train_labels, args.train_size),
        ("val", val_images, val_labels, args.val_size),
        ("test", mnist_test.data, mnist_test.targets, args.test_size),
    ):
        print(f"Creating {split}.pt ({size:,} samples)...")
        dataset = compose_split(
            source_images,
            source_labels,
            size,
            max_overlap_ratio=args.max_overlap_ratio,
            placement_attempts=args.placement_attempts,
            digit_attempts=args.digit_attempts,
            min_rotation_degrees=args.min_rotation_degrees,
            max_rotation_degrees=args.max_rotation_degrees,
            foreground_threshold=args.foreground_threshold,
            split_name=split,
        )
        torch.save(dataset, args.output_dir / f"{split}.pt")

    write_train_summary(args.output_dir, args.train_size)
    print(f"Saved dataset and schema to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
