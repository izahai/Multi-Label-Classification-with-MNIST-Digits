"""Evaluate a trained classifier on separate original and bbox data sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .dataset import load_split
from .models import build_classifier_from_checkpoint
from .train import evaluate, make_loader, print_metrics, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--bbox-data-dir",
        type=Path,
        default=Path("data/uni_with_bboxes"),
    )
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers cannot be negative.")
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, model_name = build_classifier_from_checkpoint(checkpoint)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    threshold = (
        float(checkpoint.get("threshold", 0.5))
        if args.threshold is None
        else args.threshold
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")

    splits = args.splits or ["train", "val", "test"]
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Presence threshold: {threshold:.2f}")

    results: dict[str, dict[str, float]] = {}
    for split in splits:
        for source, directory in (
            ("original", args.original_data_dir),
            ("bbox", args.bbox_data_dir),
        ):
            name = f"{source}_{split}"
            dataset = load_split(
                directory,
                split,
                max_samples=args.max_samples,
                source=source,
            )
            loader = make_loader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                device=device,
            )
            metrics = evaluate(
                model,
                loader,
                criterion,
                device,
                threshold,
                description=name.replace("_", " ").title(),
                amp_enabled=amp_enabled,
            )
            results[name] = metrics
            print_metrics(name.replace("_", " ").title(), metrics)

    output_path = args.json_output
    if output_path is None:
        output_path = args.checkpoint.parent / "evaluation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics to {output_path.resolve()}")


if __name__ == "__main__":
    main()
