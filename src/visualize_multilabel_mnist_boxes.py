"""Interactively inspect composite MNIST images and their ground-truth boxes.

Controls: Left/Right arrows or Previous/Next buttons navigate sequentially;
press R or use Random to jump to a randomly selected sample; Q closes the
viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import torch


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "uni_with_bboxes"


class DatasetViewer:
    """Matplotlib viewer with navigation controls for one generated data split."""

    def __init__(self, dataset: dict[str, object], split_name: str) -> None:
        self.images = dataset["images"]
        self.num_digits = dataset["num_digits"]
        self.boxes = dataset["all_bboxes"]
        self.box_labels = dataset["bbox_labels"]
        self.split_name = split_name
        self.index = 0

        self.figure, self.axis = plt.subplots(figsize=(8, 8))
        self.figure.subplots_adjust(bottom=0.15)
        self._add_buttons()
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def _add_buttons(self) -> None:
        self.previous_button = Button(self.figure.add_axes((0.20, 0.04, 0.16, 0.06)), "Previous")
        self.random_button = Button(self.figure.add_axes((0.42, 0.04, 0.16, 0.06)), "Random")
        self.next_button = Button(self.figure.add_axes((0.64, 0.04, 0.16, 0.06)), "Next")
        self.previous_button.on_clicked(lambda _: self.move(-1))
        self.random_button.on_clicked(lambda _: self.random_sample())
        self.next_button.on_clicked(lambda _: self.move(1))

    def draw(self) -> None:
        self.axis.clear()
        image = self.images[self.index]
        count = int(self.num_digits[self.index].item())

        self.axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        colors = plt.cm.tab10.colors
        for slot in range(count):
            x_min, y_min, x_max, y_max = self.boxes[self.index, slot].tolist()
            label = int(self.box_labels[self.index, slot].item())
            color = colors[label % len(colors)]
            self.axis.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor="none",
                )
            )
            self.axis.text(
                x_min,
                max(0, y_min - 2),
                str(label),
                color="white",
                fontsize=10,
                fontweight="bold",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1},
            )

        self.axis.set_title(
            f"{self.split_name}: sample {self.index + 1:,}/{len(self.images):,} "
            f"— {count} digits\nLeft/Right: navigate | R: random | Q: close"
        )
        self.axis.axis("off")
        self.figure.canvas.draw_idle()

    def move(self, offset: int) -> None:
        self.index = (self.index + offset) % len(self.images)
        self.draw()

    def random_sample(self) -> None:
        self.index = int(torch.randint(len(self.images), ()).item())
        self.draw()

    def on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key in ("right", "n"):
            self.move(1)
        elif key in ("left", "p"):
            self.move(-1)
        elif key == "r":
            self.random_sample()
        elif key in ("q", "escape"):
            plt.close(self.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0, help="Initial zero-based sample index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.data_dir / f"{args.split}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    dataset = torch.load(path, map_location="cpu", weights_only=True)
    required_keys = {"images", "num_digits", "all_bboxes", "bbox_labels"}
    missing = required_keys - dataset.keys()
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")

    viewer = DatasetViewer(dataset, args.split)
    viewer.index = args.index % len(viewer.images)
    viewer.draw()
    plt.show()


if __name__ == "__main__":
    main()
