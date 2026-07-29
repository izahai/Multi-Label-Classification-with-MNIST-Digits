"""Regression tests for classifier construction and checkpoint restoration."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .models import (
    MODEL_NAMES,
    build_classifier,
    build_classifier_from_checkpoint,
    count_parameters,
)
from .train import parse_args


class ClassifierModelTests(unittest.TestCase):
    def test_all_registered_models_build_with_expected_default_dropout(self) -> None:
        expected_dropouts = {
            "small": 0.2,
            "large": 0.2,
            "dense_net": 0.3,
            "ConvNeXt": 0.0,
        }
        self.assertEqual(MODEL_NAMES, tuple(expected_dropouts))
        for model_name, expected_dropout in expected_dropouts.items():
            model = build_classifier(model_name)
            self.assertEqual(model.dropout_probability, expected_dropout)

    def test_dense_net_forward_backward_and_parameter_count(self) -> None:
        model = build_classifier("dense_net")
        model.eval()
        images = torch.randn(1, 1, 64, 64)
        targets = torch.zeros(1, 10)

        logits = model(images)
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        loss.backward()

        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(count_parameters(model), (6_955_146, 6_955_146))

    def test_dense_net_checkpoint_restores_architecture_and_dropout(self) -> None:
        original = build_classifier("dense_net", dropout=0.15)
        checkpoint = {
            "model_name": "dense_net",
            "model_config": {"num_classes": 10, "dropout": 0.15},
            "model_state": original.state_dict(),
        }

        restored, model_name = build_classifier_from_checkpoint(checkpoint)
        restored.load_state_dict(checkpoint["model_state"])

        self.assertEqual(model_name, "dense_net")
        self.assertEqual(restored.dropout_probability, 0.15)
        self.assertEqual(count_parameters(restored), count_parameters(original))

    def test_convnext_forward_backward_and_parameter_count(self) -> None:
        model = build_classifier("ConvNeXt")
        model.eval()
        images = torch.randn(1, 1, 64, 64)
        targets = torch.zeros(1, 10)

        logits = model(images)
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        loss.backward()

        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(count_parameters(model), (4_850_554, 4_850_554))

    def test_convnext_checkpoint_restores_architecture_and_dropout(self) -> None:
        original = build_classifier("ConvNeXt", dropout=0.1)
        checkpoint = {
            "model_name": "ConvNeXt",
            "model_config": {"num_classes": 10, "dropout": 0.1},
            "model_state": original.state_dict(),
        }

        restored, model_name = build_classifier_from_checkpoint(checkpoint)
        restored.load_state_dict(checkpoint["model_state"])

        self.assertEqual(model_name, "ConvNeXt")
        self.assertEqual(restored.dropout_probability, 0.1)
        self.assertEqual(count_parameters(restored), count_parameters(original))

    def test_training_cli_uses_model_name(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["src_model_cls.train", "--model-name", "ConvNeXt"],
        ):
            args = parse_args()
        self.assertEqual(args.model_name, "ConvNeXt")

        with patch.object(
            sys,
            "argv",
            ["src_model_cls.train", "--model-size", "small"],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()


if __name__ == "__main__":
    unittest.main()
