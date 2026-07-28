"""Train the 20-slot composite MNIST detector.

Example:
    python -m src_model.train --data-dir data/uni_with_bboxes
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .criterion import HungarianDetectionCriterion
from .dataset import CompositeMNISTDetection, detection_collate
from .detection_utils import postprocess_detections
from .metrics import DetectionMetrics
from .mnist_detector import MNISTDetector


LOSS_NAMES = (
    "loss",
    "objectness_loss",
    "class_loss",
    "bbox_loss",
    "giou_loss",
)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_targets(
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in targets
    ]


def make_loader(
    dataset: CompositeMNISTDetection,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: MNISTDetector,
    loader: DataLoader,
    criterion: HungarianDetectionCriterion,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    gradient_clip: float,
    log_interval: int,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            losses = criterion(outputs, targets)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = len(images)
        sample_count += batch_size
        for name in LOSS_NAMES:
            totals[name] += float(losses[name].detach().item()) * batch_size

        if batch_index % log_interval == 0 or batch_index == len(loader):
            print(
                f"\r  train {batch_index:,}/{len(loader):,} batches "
                f"| loss {totals['loss'] / sample_count:.4f}",
                end="",
                flush=True,
            )
    print()
    return {name: value / max(sample_count, 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: MNISTDetector,
    loader: DataLoader,
    criterion: HungarianDetectionCriterion,
    device: torch.device,
    confidence_threshold: float,
    nms_threshold: float,
    iou_threshold: float,
) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in LOSS_NAMES}
    metrics = DetectionMetrics(iou_threshold=iou_threshold)
    sample_count = 0

    for batch_index, (images, targets_cpu) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = move_targets(targets_cpu, device)
        outputs = model(images)
        losses = criterion(outputs, targets)
        predictions = postprocess_detections(
            outputs,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )
        metrics.update(predictions, targets)

        batch_size = len(images)
        sample_count += batch_size
        for name in LOSS_NAMES:
            totals[name] += float(losses[name].item()) * batch_size
        if batch_index == len(loader) or batch_index % 50 == 0:
            print(
                f"\r  eval  {batch_index:,}/{len(loader):,} batches",
                end="",
                flush=True,
            )
    print()

    results = {name: value / max(sample_count, 1) for name, value in totals.items()}
    results.update(metrics.compute())
    return results


def save_checkpoint(
    path: Path,
    model: MNISTDetector,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_metric: float,
    stale_epochs: int,
    history: list[dict[str, float]],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_metric": best_metric,
            "stale_epochs": stale_epochs,
            "history": history,
            "model_config": {
                "num_classes": 10,
                "num_slots": args.num_slots,
                "dropout": args.dropout,
            },
            "training_args": vars(args),
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(title="Detection loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for key, label in (
        ("val_map50", "mAP@0.5"),
        ("val_map50_95", "mAP@0.5:0.95"),
        ("val_exact_match", "Exact match"),
        ("val_binary_group_match", "Group match"),
    ):
        axes[1].plot(epochs, [row[key] for row in history], label=label)
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: loss={metrics['loss']:.4f} "
        f"mAP50={metrics['map50']:.4f} "
        f"mAP50-95={metrics['map50_95']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"exact={metrics['exact_match']:.4f} "
        f"group={metrics['binary_group_match']:.4f} "
        f"correct/sample={metrics['correct_digits_per_sample']:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/uni_with_bboxes"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mnist_detector"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-slots", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_slots < 8:
        raise ValueError("--num-slots must be at least 8 for this dataset.")
    if min(args.epochs, args.batch_size, args.patience, args.log_interval) < 1:
        raise ValueError("Epochs, batch size, patience, and log interval must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_dataset = CompositeMNISTDetection(
        args.data_dir / "train.pt",
        args.max_train_samples,
    )
    val_dataset = CompositeMNISTDetection(
        args.data_dir / "val.pt",
        args.max_val_samples,
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        device,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
    )

    model = MNISTDetector(num_slots=args.num_slots, dropout=args.dropout).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"Model parameters: {total_parameters:,} total | "
        f"{trainable_parameters:,} trainable"
    )
    criterion = HungarianDetectionCriterion()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_metric = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        history = checkpoint.get("history", [])
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.gradient_clip,
            args.log_interval,
            amp_enabled,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.confidence_threshold,
            args.nms_threshold,
            args.iou_threshold,
        )
        scheduler.step(val_metrics["loss"])

        row: dict[str, float] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        print_metrics("Validation", val_metrics)

        current_metric = val_metrics["map50_95"]
        improved = current_metric > best_metric + 1e-6
        if improved:
            best_metric = current_metric
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            stale_epochs,
            history,
            args,
        )
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                stale_epochs,
                history,
                args,
            )
        write_history(args.output_dir / "history.csv", history)
        plot_history(args.output_dir / "training_curves.png", history)

        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without improvement.")
            break

    if not args.skip_test:
        best_path = args.output_dir / "best.pt"
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_dataset = CompositeMNISTDetection(
            args.data_dir / "test.pt",
            args.max_test_samples,
        )
        test_loader = make_loader(
            test_dataset,
            args.batch_size,
            False,
            args.num_workers,
            device,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            args.confidence_threshold,
            args.nms_threshold,
            args.iou_threshold,
        )
        print_metrics("Test", test_metrics)
        torch.save(test_metrics, args.output_dir / "test_metrics.pt")


if __name__ == "__main__":
    main()
