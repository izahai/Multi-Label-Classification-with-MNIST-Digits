"""Create composite MNIST data for multi-label classification and detection.

Each 64×64 image contains 6, 7, or 8 MNIST digits. The number of examples for
each count is balanced (it differs by at most one sample). One rotated digit is
kept at the image centre and rendered on top; every digit has a ground-truth
axis-aligned bounding box and class label.
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
CENTER_POSITION = ((IMAGE_SIZE - DIGIT_SIZE) // 2, (IMAGE_SIZE - DIGIT_SIZE) // 2)
BOX_FORMAT = "xyxy_exclusive"  # (x_min, y_min, x_max, y_max)


def balanced_digit_counts(num_samples: int) -> torch.Tensor:
    """Return a shuffled sequence of 6, 7, and 8 with nearly equal frequency."""
    base, remainder = divmod(num_samples, 3)
    counts = torch.tensor(
        [6] * base + [7] * base + [8] * base,
        dtype=torch.int64,
    )
    if remainder:
        counts = torch.cat(
            (
                counts,
                torch.tensor([6, 7][:remainder], dtype=torch.int64),
            )
        )
    return counts[torch.randperm(num_samples)]


def make_digit_patch(
    source_images: torch.Tensor,
    index: int,
    use_random_rotation: bool,
    min_rotation_degrees: float,
    max_rotation_degrees: float,
    foreground_threshold: float,
    use_intensity_jitter: bool,
    min_intensity_scale: float,
    max_intensity_scale: float,
    min_intensity_shift: float,
    max_intensity_shift: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Return an augmented patch, its actual mask, angle, scale, and shift."""
    image = source_images[index].unsqueeze(0).float() / 255.0
    angle = (
        float(
            torch.empty(1)
            .uniform_(min_rotation_degrees, max_rotation_degrees)
            .item()
        )
        if use_random_rotation
        else 0.0
    )
    rotated = TF.rotate(
        image,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    ).squeeze(0).clamp(0, 1)
    rotated_mask = rotated >= foreground_threshold
    if use_intensity_jitter:
        intensity_scale = float(
            torch.empty(1)
            .uniform_(min_intensity_scale, max_intensity_scale)
            .item()
        )
        intensity_shift = float(
            torch.empty(1)
            .uniform_(min_intensity_shift, max_intensity_shift)
            .item()
        )
    else:
        intensity_scale = 1.0
        intensity_shift = 0.0

    augmented = (rotated * intensity_scale + intensity_shift).clamp(0, 1)
    augmented[~rotated_mask] = 0.0
    mask = augmented >= foreground_threshold
    patch = (augmented * 255).round().to(torch.uint8)
    patch[~mask] = 0
    return patch, mask, angle, intensity_scale, intensity_shift


def apply_gaussian_noise(image: torch.Tensor, standard_deviation: float) -> None:
    """Apply zero-mean Gaussian noise to an entire uint8 image in place."""
    if standard_deviation <= 0:
        return
    noisy = image.float().div(255.0)
    noisy.add_(torch.randn_like(noisy) * standard_deviation)
    image.copy_(noisy.clamp_(0, 1).mul_(255).round_().to(torch.uint8))


def apply_salt_pepper_noise(
    image: torch.Tensor,
    probability: float,
    salt_ratio: float,
    min_pepper_intensity: float,
    max_pepper_intensity: float,
    min_salt_intensity: float,
    max_salt_intensity: float,
) -> None:
    """Apply variable-intensity salt and pepper noise to an entire image."""
    selected = torch.rand(image.shape) < probability
    salt = selected & (torch.rand(image.shape) < salt_ratio)
    pepper = selected & ~salt

    if salt.any():
        salt_values = torch.empty(int(salt.sum().item())).uniform_(
            min_salt_intensity,
            max_salt_intensity,
        )
        image[salt] = (salt_values * 255).round().to(torch.uint8)
    if pepper.any():
        pepper_values = torch.empty(int(pepper.sum().item())).uniform_(
            min_pepper_intensity,
            max_pepper_intensity,
        )
        image[pepper] = (pepper_values * 255).round().to(torch.uint8)


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
    """Find a random position below the maximum actual-foreground overlap."""
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
    use_random_rotation: bool,
    min_center_rotation_degrees: float,
    max_center_rotation_degrees: float,
    min_extra_rotation_degrees: float,
    max_extra_rotation_degrees: float,
    foreground_threshold: float,
    use_intensity_jitter: bool,
    min_digit_intensity_scale: float,
    max_digit_intensity_scale: float,
    min_digit_intensity_shift: float,
    max_digit_intensity_shift: float,
    use_gaussian_noise: bool,
    min_gaussian_noise_std: float,
    max_gaussian_noise_std: float,
    use_salt_pepper_noise: bool,
    min_salt_pepper_probability: float,
    max_salt_pepper_probability: float,
    salt_ratio: float,
    min_pepper_intensity: float,
    max_pepper_intensity: float,
    min_salt_intensity: float,
    max_salt_intensity: float,
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
    all_intensity_scales = torch.full((num_samples, MAX_DIGITS), float("nan"), dtype=torch.float32)
    all_intensity_shifts = torch.full(
        (num_samples, MAX_DIGITS),
        float("nan"),
        dtype=torch.float32,
    )
    gaussian_noise_stds = torch.empty(num_samples, dtype=torch.float32)
    salt_pepper_probabilities = torch.empty(num_samples, dtype=torch.float32)

    progress = tqdm(
        enumerate(num_digits.tolist()),
        total=num_samples,
        desc=f"Generating {split_name}",
        unit="sample",
        dynamic_ncols=True,
    )
    for sample_index, digit_count in progress:
        placed_digits: list[tuple[torch.Tensor, tuple[int, int]]] = []
        pending_draws: list[tuple[torch.Tensor, torch.Tensor, tuple[int, int]]] = []

        for slot in range(digit_count):
            for _ in range(digit_attempts):
                source_index = int(torch.randint(len(source_images), ()).item())
                digit_label = int(source_labels[source_index].item())
                if slot == 0:
                    min_rotation = min_center_rotation_degrees
                    max_rotation = max_center_rotation_degrees
                else:
                    min_rotation = min_extra_rotation_degrees
                    max_rotation = max_extra_rotation_degrees
                patch, mask, angle, intensity_scale, intensity_shift = make_digit_patch(
                    source_images,
                    source_index,
                    use_random_rotation,
                    min_rotation,
                    max_rotation,
                    foreground_threshold,
                    use_intensity_jitter,
                    min_digit_intensity_scale,
                    max_digit_intensity_scale,
                    min_digit_intensity_shift,
                    max_digit_intensity_shift,
                )
                position = (
                    CENTER_POSITION
                    if slot == 0
                    else low_overlap_position(
                        mask,
                        placed_digits,
                        max_overlap_ratio=max_overlap_ratio,
                        attempts=placement_attempts,
                    )
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
            all_digit_labels[sample_index, slot] = digit_label
            bbox_labels[sample_index, slot] = digit_label
            all_positions[sample_index, slot] = torch.tensor((row, column))
            all_bboxes[sample_index, slot] = torch.tensor(tight_bbox(mask, position))
            all_rotation_degrees[sample_index, slot] = angle
            all_intensity_scales[sample_index, slot] = intensity_scale
            all_intensity_shifts[sample_index, slot] = intensity_shift
            placed_digits.append((mask, position))
            pending_draws.append((patch, mask, position))
            labels[sample_index, digit_label] = 1.0
            count_labels[sample_index, digit_label] += 1

        # Draw the centred digit (slot 0) last so it remains visible wherever
        # it overlaps the randomly placed distractor digits.
        for patch, mask, (row, column) in pending_draws[1:] + pending_draws[:1]:
            target = images[sample_index, row : row + DIGIT_SIZE, column : column + DIGIT_SIZE]
            target[mask] = patch[mask]

        gaussian_std = (
            float(
                torch.empty(1)
                .uniform_(min_gaussian_noise_std, max_gaussian_noise_std)
                .item()
            )
            if use_gaussian_noise
            else 0.0
        )
        apply_gaussian_noise(images[sample_index], gaussian_std)
        gaussian_noise_stds[sample_index] = gaussian_std

        noise_probability = (
            float(
                torch.empty(1)
                .uniform_(
                    min_salt_pepper_probability,
                    max_salt_pepper_probability,
                )
                .item()
            )
            if use_salt_pepper_noise
            else 0.0
        )
        if use_salt_pepper_noise:
            apply_salt_pepper_noise(
                images[sample_index],
                probability=noise_probability,
                salt_ratio=salt_ratio,
                min_pepper_intensity=min_pepper_intensity,
                max_pepper_intensity=max_pepper_intensity,
                min_salt_intensity=min_salt_intensity,
                max_salt_intensity=max_salt_intensity,
            )
        salt_pepper_probabilities[sample_index] = noise_probability
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
        "all_intensity_scales": all_intensity_scales,
        "all_intensity_shifts": all_intensity_shifts,
        "gaussian_noise_stds": gaussian_noise_stds,
        "salt_pepper_probabilities": salt_pepper_probabilities,
        "image_size": IMAGE_SIZE,
        "digit_size": DIGIT_SIZE,
        "center_position": CENTER_POSITION,
        "num_classes": 10,
        "min_digits": 6,
        "max_digits": 8,
        "max_overlap_ratio": max_overlap_ratio,
        "max_pairwise_overlap_ratio": max_overlap_ratio,
        "overlap_measure": "foreground_intersection_over_smaller_foreground_area",
        "placement_attempts": placement_attempts,
        "digit_attempts": digit_attempts,
        "foreground_threshold": foreground_threshold,
        "use_random_rotation": use_random_rotation,
        "center_rotation_degrees": (
            min_center_rotation_degrees,
            max_center_rotation_degrees,
        ),
        "extra_rotation_degrees": (
            min_extra_rotation_degrees,
            max_extra_rotation_degrees,
        ),
        "use_intensity_jitter": use_intensity_jitter,
        "intensity_scale_range": (
            min_digit_intensity_scale,
            max_digit_intensity_scale,
        ),
        "digit_intensity_scale_range": (
            min_digit_intensity_scale,
            max_digit_intensity_scale,
        ),
        "intensity_shift_range": (
            min_digit_intensity_shift,
            max_digit_intensity_shift,
        ),
        "use_gaussian_noise": use_gaussian_noise,
        "gaussian_noise_std_range": (
            min_gaussian_noise_std,
            max_gaussian_noise_std,
        ),
        "use_salt_pepper_noise": use_salt_pepper_noise,
        "salt_pepper_prob_range": (
            min_salt_pepper_probability,
            max_salt_pepper_probability,
        ),
        "salt_pepper_probability_range": (
            min_salt_pepper_probability,
            max_salt_pepper_probability,
        ),
        "salt_ratio": salt_ratio,
        "pepper_intensity_range": (
            min_pepper_intensity,
            max_pepper_intensity,
        ),
        "salt_intensity_range": (min_salt_intensity, max_salt_intensity),
        "compositing": "center_digit_foreground_overwrites_all_other_digits",
        "label_type": "multi_hot_digit_presence",
        "label_priority": "center_digit_foreground_pixels_overwrite_distractors",
        "bbox_format": BOX_FORMAT,
        "bbox_label_key": "bbox_labels",
        "unused_slot_value": -1,
        "difficulty": "hard_overlap_rotation_noise",
    }


def write_train_summary(output_dir: Path, train_size: int) -> None:
    """Write a concise schema document alongside the generated train split."""
    summary = f"""# `train.pt` structure with bounding boxes

`train.pt` contains **{train_size:,}** synthetic 64×64 MNIST composite images. Every sample contains 6, 7, or 8 digits; the generator balances these three values as evenly as possible.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `images` | `uint8`, `({train_size}, 64, 64)` | Grayscale composite images. |
| `labels` | `float32`, `({train_size}, 10)` | Multi-hot digit-presence target for classes 0–9. |
| `center_labels` | `int64`, `({train_size},)` | Label of slot 0, the centred digit. |
| `count_labels` | `int64`, `({train_size}, 10)` | Number of appearances of each class. |
| `num_digits` | `int64`, `({train_size},)` | Number of digit instances: 6, 7, or 8. |
| `all_digit_labels` | `int64`, `({train_size}, 8)` | Class label for each instance slot. |
| `all_positions` | `int64`, `({train_size}, 8, 2)` | Each patch's top-left `(row, column)`. |
| `all_bboxes` | `int64`, `({train_size}, 8, 4)` | Tight foreground boxes `(x_min, y_min, x_max, y_max)` after digit augmentation, with exclusive maxima. |
| `bbox_labels` | `int64`, `({train_size}, 8)` | Class label paired with the box at the same slot in `all_bboxes`. |
| `all_rotation_degrees` | `float32`, `({train_size}, 8)` | Random rotation angle used for each digit. |
| `all_intensity_scales` | `float32`, `({train_size}, 8)` | Random bold/dim scale applied to each digit. |
| `all_intensity_shifts` | `float32`, `({train_size}, 8)` | Normalized intensity offset applied to each digit. |
| `gaussian_noise_stds` | `float32`, `({train_size},)` | Gaussian noise standard deviation sampled for each full image. |
| `salt_pepper_probabilities` | `float32`, `({train_size},)` | Salt-and-pepper probability sampled for each full image. |

Slots from `num_digits` through slot 7 are padding and use `-1` in the label, position, and box tensors and `NaN` in the rotation/intensity tensors. Slot 0 is fixed at the image centre and uses its own rotation range; the other digits are randomly placed and use the distractor rotation range. Pairwise overlap is measured from augmented foreground pixels as `intersection / smaller foreground area`. Placement only enforces the configured maximum, so digits may have little or no overlap. The centred digit is drawn last, followed by full-image Gaussian and salt-and-pepper noise.
"""
    (output_dir / "train_structure.md").write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gen_data"))
    parser.add_argument(
        "--mode",
        choices=("all", "train"),
        default="all",
        help=(
            "Splits to generate: 'all' creates train/val/test; "
            "'train' creates only train.pt (default: all)."
        ),
    )
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--val-size", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-overlap-ratio",
        type=float,
        default=0.20,
        help="Maximum actual-foreground overlap for every digit pair (default: 0.20).",
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
        help="Legacy override for both minimum rotation angles.",
    )
    parser.add_argument(
        "--max-rotation-degrees",
        type=float,
        help="Legacy override for both maximum rotation angles.",
    )
    parser.add_argument(
        "--min-center-rotation-degrees",
        type=float,
        help="Minimum center-digit rotation (default: -15).",
    )
    parser.add_argument(
        "--max-center-rotation-degrees",
        type=float,
        help="Maximum center-digit rotation (default: 15).",
    )
    parser.add_argument(
        "--min-extra-rotation-degrees",
        type=float,
        help="Minimum distractor rotation (default: -35).",
    )
    parser.add_argument(
        "--max-extra-rotation-degrees",
        type=float,
        help="Maximum distractor rotation (default: 35).",
    )
    parser.add_argument(
        "--random-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply independent random digit rotations (default: true).",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.05,
        help="Normalized pixel threshold defining digit foreground (default: 0.05).",
    )
    parser.add_argument(
        "--min-digit-intensity-scale",
        type=float,
        default=0.75,
        help="Minimum per-digit intensity multiplier (default: 0.75).",
    )
    parser.add_argument(
        "--max-digit-intensity-scale",
        type=float,
        default=1.20,
        help="Maximum per-digit intensity multiplier (default: 1.20).",
    )
    parser.add_argument(
        "--min-digit-intensity-shift",
        type=float,
        default=-0.05,
        help="Minimum normalized per-digit intensity shift (default: -0.05).",
    )
    parser.add_argument(
        "--max-digit-intensity-shift",
        type=float,
        default=0.05,
        help="Maximum normalized per-digit intensity shift (default: 0.05).",
    )
    parser.add_argument(
        "--intensity-jitter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-digit intensity scale and shift (default: true).",
    )
    parser.add_argument(
        "--min-gaussian-noise-std",
        type=float,
        default=0.02,
        help="Minimum full-image Gaussian noise std (default: 0.02).",
    )
    parser.add_argument(
        "--max-gaussian-noise-std",
        type=float,
        default=0.12,
        help="Maximum full-image Gaussian noise std (default: 0.12).",
    )
    parser.add_argument(
        "--gaussian-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply full-image Gaussian noise (default: true).",
    )
    parser.add_argument(
        "--min-salt-pepper-probability",
        type=float,
        default=0.00,
        help="Minimum per-image noisy-pixel probability (default: 0.0).",
    )
    parser.add_argument(
        "--max-salt-pepper-probability",
        type=float,
        default=0.025,
        help="Maximum per-image noisy-pixel probability (default: 0.025).",
    )
    parser.add_argument(
        "--salt-pepper-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply full-image salt-and-pepper noise (default: true).",
    )
    parser.add_argument(
        "--salt-ratio",
        type=float,
        default=0.50,
        help="Fraction of selected noise pixels assigned to salt (default: 0.50).",
    )
    parser.add_argument(
        "--min-pepper-intensity",
        type=float,
        default=0.0,
        help="Minimum normalized pepper-pixel intensity (default: 0.0).",
    )
    parser.add_argument(
        "--max-pepper-intensity",
        type=float,
        default=0.25,
        help="Maximum normalized pepper-pixel intensity (default: 0.25).",
    )
    parser.add_argument(
        "--min-salt-intensity",
        type=float,
        default=0.45,
        help="Minimum normalized salt-pixel intensity (default: 0.45).",
    )
    parser.add_argument(
        "--max-salt-intensity",
        type=float,
        default=1.0,
        help="Maximum normalized salt-pixel intensity (default: 1.0).",
    )
    args = parser.parse_args()
    minimum_rotation_supplied = args.min_rotation_degrees is not None
    maximum_rotation_supplied = args.max_rotation_degrees is not None
    if minimum_rotation_supplied and (
        args.min_center_rotation_degrees is not None
        or args.min_extra_rotation_degrees is not None
    ):
        parser.error(
            "--min-rotation-degrees cannot be combined with the new minimum "
            "center/extra rotation arguments."
        )
    if maximum_rotation_supplied and (
        args.max_center_rotation_degrees is not None
        or args.max_extra_rotation_degrees is not None
    ):
        parser.error(
            "--max-rotation-degrees cannot be combined with the new maximum "
            "center/extra rotation arguments."
        )

    minimum_rotation = args.min_rotation_degrees
    maximum_rotation = args.max_rotation_degrees
    args.min_center_rotation_degrees = (
        minimum_rotation
        if minimum_rotation_supplied
        else (
            -15.0
            if args.min_center_rotation_degrees is None
            else args.min_center_rotation_degrees
        )
    )
    args.min_extra_rotation_degrees = (
        minimum_rotation
        if minimum_rotation_supplied
        else (
            -35.0
            if args.min_extra_rotation_degrees is None
            else args.min_extra_rotation_degrees
        )
    )
    args.max_center_rotation_degrees = (
        maximum_rotation
        if maximum_rotation_supplied
        else (
            15.0
            if args.max_center_rotation_degrees is None
            else args.max_center_rotation_degrees
        )
    )
    args.max_extra_rotation_degrees = (
        maximum_rotation
        if maximum_rotation_supplied
        else (
            35.0
            if args.max_extra_rotation_degrees is None
            else args.max_extra_rotation_degrees
        )
    )
    return args


def main() -> None:
    args = parse_args()
    if min(args.train_size, args.placement_attempts, args.digit_attempts) < 1:
        raise ValueError("Train size and attempt counts must be positive.")
    if args.mode == "all" and min(args.val_size, args.test_size) < 1:
        raise ValueError("Validation and test sizes must be positive in --mode all.")
    if not 0 <= args.max_overlap_ratio <= 1:
        raise ValueError("--max-overlap-ratio must be between 0 and 1.")
    if not 0 <= args.foreground_threshold <= 1:
        raise ValueError("--foreground-threshold must be between 0 and 1.")
    ranges = {
        "center rotation": (
            args.min_center_rotation_degrees,
            args.max_center_rotation_degrees,
        ),
        "extra rotation": (
            args.min_extra_rotation_degrees,
            args.max_extra_rotation_degrees,
        ),
        "digit intensity scale": (
            args.min_digit_intensity_scale,
            args.max_digit_intensity_scale,
        ),
        "digit intensity shift": (
            args.min_digit_intensity_shift,
            args.max_digit_intensity_shift,
        ),
        "Gaussian noise std": (
            args.min_gaussian_noise_std,
            args.max_gaussian_noise_std,
        ),
        "salt-and-pepper probability": (
            args.min_salt_pepper_probability,
            args.max_salt_pepper_probability,
        ),
        "pepper intensity": (
            args.min_pepper_intensity,
            args.max_pepper_intensity,
        ),
        "salt intensity": (
            args.min_salt_intensity,
            args.max_salt_intensity,
        ),
    }
    for name, (minimum, maximum) in ranges.items():
        if minimum > maximum:
            raise ValueError(f"Minimum {name} cannot exceed maximum {name}.")
    if args.min_digit_intensity_scale < 0:
        raise ValueError("Digit intensity scales must be non-negative.")
    if args.min_gaussian_noise_std < 0:
        raise ValueError("Gaussian noise standard deviations must be non-negative.")
    for name in (
        "min_salt_pepper_probability",
        "max_salt_pepper_probability",
        "salt_ratio",
        "min_pepper_intensity",
        "max_pepper_intensity",
        "min_salt_intensity",
        "max_salt_intensity",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mnist_train = MNIST(args.mnist_dir, train=True, download=True)
    train_images, val_images = mnist_train.data[:50_000], mnist_train.data[50_000:]
    train_labels, val_labels = mnist_train.targets[:50_000], mnist_train.targets[50_000:]

    splits = [("train", train_images, train_labels, args.train_size)]
    if args.mode == "all":
        mnist_test = MNIST(args.mnist_dir, train=False, download=True)
        splits.extend(
            [
                ("val", val_images, val_labels, args.val_size),
                ("test", mnist_test.data, mnist_test.targets, args.test_size),
            ]
        )

    for split, source_images, source_labels, size in splits:
        print(f"Creating {split}.pt ({size:,} samples)...")
        dataset = compose_split(
            source_images,
            source_labels,
            size,
            max_overlap_ratio=args.max_overlap_ratio,
            placement_attempts=args.placement_attempts,
            digit_attempts=args.digit_attempts,
            use_random_rotation=args.random_rotation,
            min_center_rotation_degrees=args.min_center_rotation_degrees,
            max_center_rotation_degrees=args.max_center_rotation_degrees,
            min_extra_rotation_degrees=args.min_extra_rotation_degrees,
            max_extra_rotation_degrees=args.max_extra_rotation_degrees,
            foreground_threshold=args.foreground_threshold,
            use_intensity_jitter=args.intensity_jitter,
            min_digit_intensity_scale=args.min_digit_intensity_scale,
            max_digit_intensity_scale=args.max_digit_intensity_scale,
            min_digit_intensity_shift=args.min_digit_intensity_shift,
            max_digit_intensity_shift=args.max_digit_intensity_shift,
            use_gaussian_noise=args.gaussian_noise,
            min_gaussian_noise_std=args.min_gaussian_noise_std,
            max_gaussian_noise_std=args.max_gaussian_noise_std,
            use_salt_pepper_noise=args.salt_pepper_noise,
            min_salt_pepper_probability=args.min_salt_pepper_probability,
            max_salt_pepper_probability=args.max_salt_pepper_probability,
            salt_ratio=args.salt_ratio,
            min_pepper_intensity=args.min_pepper_intensity,
            max_pepper_intensity=args.max_pepper_intensity,
            min_salt_intensity=args.min_salt_intensity,
            max_salt_intensity=args.max_salt_intensity,
            split_name=split,
        )
        torch.save(dataset, args.output_dir / f"{split}.pt")

    if args.mode == "all":
        write_train_summary(args.output_dir, args.train_size)
        print(f"Saved dataset and schema to: {args.output_dir.resolve()}")
    else:
        print(f"Saved train.pt to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
