"""Inspect MNIST-union splits and report the distribution of `num_digits`."""

from collections import Counter
from pathlib import Path

import torch


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "uni"
SPLITS = ("train", "val", "test")


def load_num_digits(path: Path) -> torch.Tensor:
    """Load and validate the one-dimensional `num_digits` tensor from a split."""
    try:
        dataset = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Supports PyTorch files that contain non-tensor metadata objects.
        dataset = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(dataset, dict) or "num_digits" not in dataset:
        raise ValueError(f"{path} must contain a dictionary with a 'num_digits' key.")

    num_digits = dataset["num_digits"]
    if not isinstance(num_digits, torch.Tensor) or num_digits.ndim != 1:
        raise ValueError(f"{path}: 'num_digits' must be a one-dimensional tensor.")

    return num_digits.to(dtype=torch.int64)


def print_distribution(name: str, values: torch.Tensor) -> Counter[int]:
    """Print counts and percentages for one split, returning its distribution."""
    distribution = Counter(values.tolist())
    total = len(values)

    print(f"\n{name.upper()} ({total:,} samples)")
    print("num_digits  samples  percentage")
    for count in sorted(distribution):
        samples = distribution[count]
        print(f"{count:>10}  {samples:>7,}  {samples / total:>9.2%}")

    return distribution


def main() -> None:
    combined = Counter()
    total_samples = 0

    for split in SPLITS:
        path = DATA_DIR / f"{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {path}")

        num_digits = load_num_digits(path)
        combined.update(print_distribution(split, num_digits))
        total_samples += len(num_digits)

    print(f"\nALL SPLITS ({total_samples:,} samples)")
    print("num_digits  samples  percentage")
    for count in sorted(combined):
        samples = combined[count]
        print(f"{count:>10}  {samples:>7,}  {samples / total_samples:>9.2%}")


if __name__ == "__main__":
    main()
