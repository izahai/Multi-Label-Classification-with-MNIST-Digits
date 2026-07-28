"""Metrics for 10-class digit-presence prediction."""

from __future__ import annotations

import torch


class PresenceMetrics:
    """Accumulate exact-sample and per-bit accuracy."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        self.threshold = threshold
        self.exact_matches = 0
        self.correct_bits = 0
        self.samples = 0
        self.total_bits = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.shape != targets.shape or logits.ndim != 2 or logits.shape[1] != 10:
            raise ValueError(
                "logits and targets must both have shape (batch, 10); "
                f"got {tuple(logits.shape)} and {tuple(targets.shape)}."
            )
        predictions = logits.sigmoid() >= self.threshold
        truth = targets >= 0.5
        matches = predictions == truth
        self.exact_matches += int(matches.all(dim=1).sum().item())
        self.correct_bits += int(matches.sum().item())
        self.samples += len(targets)
        self.total_bits += targets.numel()

    def compute(self) -> dict[str, float]:
        return {
            "exact_match": self.exact_matches / max(self.samples, 1),
            "binary_match": self.correct_bits / max(self.total_bits, 1),
        }
