"""Train a CNN for composite-MNIST multi-label classification.

The original and bbox training splits are concatenated. Validation and test
metrics are calculated only on the original data source.

Example:
    python -m src_model_cls.train --model-name small
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .dataset import load_combined_train, load_split
from .metrics import PresenceMetrics
from .models import (
    DEFAULT_MODEL_NAME,
    MODEL_DEFAULT_DROPOUTS,
    MODEL_NAMES,
    build_classifier,
    build_classifier_from_checkpoint,
    count_parameters,
)


def seed_everything(seed: int) -> None:
    """Seed all RNGs and require deterministic PyTorch operations."""
    # Required by deterministic CUDA matrix multiplications. This is set before
    # the first CUDA operation in this process.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a deterministic Python/NumPy seed."""
    del worker_id  # The worker-specific value is already in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ExponentialMovingAverage:
    """Maintain an EMA copy of both model parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            shadow_value = self.shadow[name]
            if torch.is_floating_point(value):
                shadow_value.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow_value.copy_(value.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA state does not match the current model architecture.")
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def select_device(requested: str) -> torch.device:
    """Select CUDA, MPS, or CPU, while validating an explicit request."""
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


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int = 42,
) -> DataLoader:
    """Create a classification DataLoader with accelerator-friendly options."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    threshold: float,
    gradient_clip: float,
    amp_enabled: bool,
    ema: ExponentialMovingAverage,
) -> dict[str, float]:
    """Train for one epoch and return BCE plus presence metrics."""
    model.train()
    metrics = PresenceMetrics(threshold)
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(loader, desc="Train", unit="batch", dynamic_ncols=True)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        batch_size = len(images)
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        metrics.update(logits.detach(), targets)
        running = metrics.compute()
        progress.set_postfix(
            loss=f"{loss_sum / sample_count:.4f}",
            exact=f"{running['exact_match']:.3f}",
            binary=f"{running['binary_match']:.3f}",
        )

    return {"loss": loss_sum / max(sample_count, 1), **metrics.compute()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    *,
    description: str,
    amp_enabled: bool = False,
) -> dict[str, float]:
    """Evaluate one source without mixing its metrics with another source."""
    model.eval()
    metrics = PresenceMetrics(threshold)
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = len(images)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        metrics.update(logits, targets)
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")

    return {"loss": loss_sum / max(sample_count, 1), **metrics.compute()}


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: loss={metrics['loss']:.4f} "
        f"exact={metrics['exact_match']:.4f} "
        f"binary={metrics['binary_match']:.4f}"
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
    axes[0].plot(
        epochs,
        [row["original_val_loss"] for row in history],
        label="validation",
    )
    axes[0].set(title="BCE loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for key, label in (
        ("original_val_exact_match", "Exact match"),
        ("original_val_binary_match", "Binary match"),
    ):
        axes[1].plot(epochs, [row[key] for row in history], label=label)
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI arguments to checkpoint-safe primitive values."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    model_name: str,
    dropout: float,
    threshold: float,
    best_exact_match: float,
    best_val_loss: float,
    stale_epochs: int,
    history: list[dict[str, float]],
    top_checkpoints: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "model_config": {"num_classes": 10, "dropout": dropout},
            "threshold": threshold,
            # Inference checkpoints use EMA weights; raw weights are retained for resume.
            "model_state": ema.state_dict(),
            "training_model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_exact_match": best_exact_match,
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "history": history,
            "top_checkpoints": top_checkpoints,
            "training_args": serializable_args(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--bbox-data-dir",
        type=Path,
        default=Path("data/gen_data"),
    )
    parser.add_argument("--model-name", choices=MODEL_NAMES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--checkpoint-average-count",
        type=int,
        choices=(3, 4, 5),
        default=5,
        help="Number of best validation checkpoints to average after training.",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--learning-rate": args.learning_rate,
        "--gradient-clip": args.gradient_clip,
        "--patience": args.patience,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1).")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = select_device(args.device)

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        saved_name = str(resume_checkpoint["model_name"])
        if args.model_name is not None and args.model_name != saved_name:
            raise ValueError(
                f"--model-name {args.model_name!r} conflicts with checkpoint "
                f"model {saved_name!r}."
            )
        args.model_name = saved_name
        saved_config = resume_checkpoint.get("model_config", {})
        saved_dropout = float(
            saved_config.get(
                "dropout",
                MODEL_DEFAULT_DROPOUTS.get(saved_name, 0.2),
            )
        )
        saved_threshold = float(resume_checkpoint.get("threshold", 0.5))
        if args.dropout is not None and args.dropout != saved_dropout:
            raise ValueError("--dropout conflicts with the resume checkpoint.")
        if args.threshold is not None and args.threshold != saved_threshold:
            raise ValueError("--threshold conflicts with the resume checkpoint.")
        args.dropout = saved_dropout
        args.threshold = saved_threshold
    else:
        args.model_name = args.model_name or DEFAULT_MODEL_NAME
        args.dropout = (
            MODEL_DEFAULT_DROPOUTS[args.model_name]
            if args.dropout is None
            else args.dropout
        )
        args.threshold = 0.5 if args.threshold is None else args.threshold

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")

    args.output_dir = args.output_dir or Path(
        f"outputs/mnist_classifier_{args.model_name}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Reproducibility: seed={args.seed}, deterministic_algorithms=True")
    print("Loading and concatenating the two training sources...")
    train_dataset = load_combined_train(
        args.original_data_dir,
        args.bbox_data_dir,
        max_samples_per_source=args.max_train_samples,
    )
    original_val = load_split(
        args.original_data_dir,
        "val",
        max_samples=args.max_val_samples,
        source="original",
    )
    print(
        f"Samples: train={len(train_dataset):,} "
        f"(original={len(train_dataset.datasets[0]):,}, "
        f"bbox={len(train_dataset.datasets[1]):,}) | "
        f"validation={len(original_val):,}"
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
    }
    train_loader = make_loader(
        train_dataset,
        shuffle=True,
        seed=args.seed,
        **loader_options,
    )
    original_val_loader = make_loader(
        original_val,
        shuffle=False,
        **loader_options,
    )

    model = build_classifier(
        args.model_name,
        dropout=args.dropout,
    ).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    print(
        f"Model: {args.model_name} | {total_parameters:,} total parameters | "
        f"{trainable_parameters:,} trainable"
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ExponentialMovingAverage(model, args.ema_decay)

    start_epoch = 1
    best_exact_match = -1.0
    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    top_checkpoints: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        model.load_state_dict(
            resume_checkpoint.get("training_model_state", resume_checkpoint["model_state"])
        )
        ema.load_state_dict(resume_checkpoint.get("ema_state", resume_checkpoint["model_state"]))
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_exact_match = float(
            resume_checkpoint.get("best_exact_match", -1.0)
        )
        best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
        stale_epochs = int(resume_checkpoint.get("stale_epochs", 0))
        history = list(resume_checkpoint.get("history", []))
        top_checkpoints = [
            record
            for record in resume_checkpoint.get("top_checkpoints", [])
            if (args.output_dir / record["path"]).is_file()
        ]
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
            args.threshold,
            args.gradient_clip,
            amp_enabled,
            ema,
        )
        # Validation and model selection use the smoother EMA weights.
        raw_model_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        ema.copy_to(model)
        original_metrics = evaluate(
            model,
            original_val_loader,
            criterion,
            device,
            args.threshold,
            description="Original val",
            amp_enabled=amp_enabled,
        )
        model.load_state_dict(raw_model_state)
        scheduler.step(original_metrics["loss"])

        row: dict[str, float] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"original_val_{key}": value for key, value in original_metrics.items()}
        )
        history.append(row)

        print_metrics("Train", train_metrics)
        print_metrics("Validation", original_metrics)

        exact_improved = original_metrics["exact_match"] > best_exact_match + 1e-12
        exact_tied = (
            abs(original_metrics["exact_match"] - best_exact_match) <= 1e-12
        )
        loss_improved = original_metrics["loss"] < best_val_loss - 1e-12
        improved = exact_improved or (exact_tied and loss_improved)
        if improved:
            best_exact_match = original_metrics["exact_match"]
            best_val_loss = original_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        checkpoint_options = {
            "model": model,
            "ema": ema,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "model_name": args.model_name,
            "dropout": args.dropout,
            "threshold": args.threshold,
            "best_exact_match": best_exact_match,
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "history": history,
            "top_checkpoints": top_checkpoints,
            "args": args,
        }
        save_checkpoint(args.output_dir / "last.pt", **checkpoint_options)

        candidate_path = (
            args.output_dir / "top_checkpoints" / f"epoch_{epoch:03d}.pt"
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(candidate_path, **checkpoint_options)
        candidate_record = {
            "path": str(candidate_path.relative_to(args.output_dir)),
            "epoch": epoch,
            "exact_match": original_metrics["exact_match"],
            "loss": original_metrics["loss"],
        }
        top_checkpoints.append(candidate_record)
        top_checkpoints.sort(key=lambda item: (-item["exact_match"], item["loss"], item["epoch"]))
        discarded = top_checkpoints[args.checkpoint_average_count :]
        top_checkpoints = top_checkpoints[: args.checkpoint_average_count]
        for record in discarded:
            discarded_path = args.output_dir / record["path"]
            if discarded_path.is_file():
                discarded_path.unlink()

        checkpoint_options["top_checkpoints"] = top_checkpoints
        save_checkpoint(args.output_dir / "last.pt", **checkpoint_options)

        if improved:
            save_checkpoint(args.output_dir / "best.pt", **checkpoint_options)
            print(
                f"Saved new best checkpoint: exact={best_exact_match:.4f}, "
                f"loss={best_val_loss:.4f}"
            )
        write_history(args.output_dir / "history.csv", history)
        plot_history(args.output_dir / "training_curves.png", history)

        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} stale epochs.")
            break

    final_top_checkpoints = sorted(
        top_checkpoints,
        key=lambda item: (-item["exact_match"], item["loss"], item["epoch"]),
    )
    average_path: Path | None = None
    if len(final_top_checkpoints) >= 3:
        checkpoints = [
            torch.load(
                args.output_dir / record["path"],
                map_location="cpu",
                weights_only=False,
            )
            for record in final_top_checkpoints
        ]
        averaged_state = {
            name: (
                torch.stack(
                    [checkpoint["model_state"][name].float() for checkpoint in checkpoints]
                )
                .mean(0)
                .to(value.dtype)
            )
            if torch.is_floating_point(value)
            else value.clone()
            for name, value in checkpoints[0]["model_state"].items()
        }
        averaged_checkpoint = checkpoints[0]
        averaged_checkpoint["model_state"] = averaged_state
        averaged_checkpoint["averaged_checkpoints"] = [
            record["path"] for record in final_top_checkpoints
        ]
        average_path = args.output_dir / f"averaged_top_{len(final_top_checkpoints)}.pt"
        torch.save(averaged_checkpoint, average_path)
        print(f"Saved averaged checkpoint to {average_path.resolve()}")

    if args.skip_test:
        return

    best_path = args.output_dir / "best.pt"
    if not best_path.is_file() and args.resume is not None:
        resumed_best = args.resume.parent / "best.pt"
        if resumed_best.is_file():
            best_path = resumed_best
            print(f"Using existing best checkpoint from {best_path}.")
    if not best_path.is_file():
        best_path = args.output_dir / "last.pt"
        print("No best checkpoint was written in this run; using last.pt for test.")
    original_test = load_split(
        args.original_data_dir,
        "test",
        max_samples=args.max_test_samples,
        source="original",
    )
    test_loader = make_loader(original_test, shuffle=False, **loader_options)

    def test_checkpoint(label: str, path: Path) -> dict[str, float]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        checkpoint_model, _ = build_classifier_from_checkpoint(checkpoint)
        checkpoint_model = checkpoint_model.to(device)
        checkpoint_model.load_state_dict(checkpoint["model_state"])
        result = evaluate(
            checkpoint_model,
            test_loader,
            criterion,
            device,
            args.threshold,
            description=label.replace("_", " ").title(),
            amp_enabled=amp_enabled,
        )
        print_metrics(label.replace("_", " ").title(), result)
        return result

    test_results = {"best": test_checkpoint("best", best_path)}
    for rank, record in enumerate(final_top_checkpoints, start=1):
        label = f"top_{rank:02d}_epoch_{record['epoch']:03d}"
        test_results[label] = test_checkpoint(label, args.output_dir / record["path"])

    if average_path is not None:
        test_results[f"averaged_top_{len(final_top_checkpoints)}"] = test_checkpoint(
            f"averaged_top_{len(final_top_checkpoints)}", average_path
        )

    metrics_path = args.output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_results, indent=2), encoding="utf-8")
    print(f"Saved test metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
