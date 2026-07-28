"""Run checkpoint inference and compare detections with ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from .dataset import CompositeMNISTDetection
from .detection_utils import postprocess_detections
from .model_factory import build_detector_from_checkpoint
from .train import select_device


def draw_boxes(
    axis: plt.Axes,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    image_size: int,
    scores: torch.Tensor | None = None,
    linestyle: str = "-",
) -> None:
    colors = plt.cm.tab10.colors
    for index, (box, label_tensor) in enumerate(zip(boxes, labels)):
        x_min, y_min, x_max, y_max = (box * image_size).tolist()
        label = int(label_tensor.item())
        color = colors[label % len(colors)]
        axis.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1.7,
                edgecolor=color,
                facecolor="none",
                linestyle=linestyle,
            )
        )
        text = str(label)
        if scores is not None:
            text += f" {float(scores[index]):.2f}"
        axis.text(
            x_min,
            max(0, y_min - 1),
            text,
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/mnist_detector_small/best.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/uni_with_bboxes"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, model_name = build_detector_from_checkpoint(checkpoint)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model: {model_name}")

    dataset = CompositeMNISTDetection(args.data_dir / f"{args.split}.pt")
    index = (
        int(torch.randint(len(dataset), ()).item())
        if args.random
        else args.index % len(dataset)
    )
    image, target = dataset[index]
    with torch.no_grad():
        outputs = model(image.unsqueeze(0).to(device))
        prediction = postprocess_detections(
            outputs,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
        )[0]
    prediction = {key: value.cpu() for key, value in prediction.items()}

    print(f"{args.split} sample {index}: {len(target['boxes'])} ground-truth digits")
    for label, score, box in zip(
        prediction["labels"],
        prediction["scores"],
        prediction["boxes"],
    ):
        pixel_box = (box * dataset.image_size).round().to(torch.int64).tolist()
        print(f"prediction label={int(label)} score={float(score):.3f} box={pixel_box}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis in axes:
        axis.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    axes[0].set_title("Ground truth")
    draw_boxes(
        axes[0],
        target["boxes"],
        target["labels"],
        dataset.image_size,
    )
    axes[1].set_title("Predictions after confidence + NMS")
    draw_boxes(
        axes[1],
        prediction["boxes"],
        prediction["labels"],
        dataset.image_size,
        prediction["scores"],
        linestyle="--",
    )
    figure.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print(f"Saved visualization to {args.output.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
