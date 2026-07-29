"""Regression tests for classifier construction and checkpoint restoration."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from .models import (
    AdaptiveGeMPool2d,
    ClassAttentionHead,
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
            "densenet_atn_head": 0.3,
            "densenet_atn_head_v2": 0.3,
            "denset_net_atn_head": 0.3,
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

    def test_densenet_attention_head_produces_ten_logits(self) -> None:
        model = build_classifier("densenet_atn_head")
        model.eval()
        logits = model(torch.randn(1, 1, 64, 64))
        nn.BCEWithLogitsLoss()(logits, torch.zeros(1, 10)).backward()

        self.assertEqual(logits.shape, (1, 10))
        self.assertIsInstance(model.dropout, nn.Dropout2d)

    def test_adaptive_gem_matches_average_pooling_at_p_one(self) -> None:
        pool = AdaptiveGeMPool2d(p=1.0)
        features = torch.rand(2, 4, 7, 9, requires_grad=True)

        pooled = pool(features, (2, 3))
        expected = F.adaptive_avg_pool2d(features, (2, 3))
        pooled.sum().backward()

        torch.testing.assert_close(pooled, expected)
        self.assertEqual(pooled.shape, (2, 4, 2, 3))
        self.assertIsNotNone(pool.p.grad)

    def test_densenet_attention_v2_fuses_three_gem_scales(self) -> None:
        model = build_classifier("densenet_atn_head_v2")
        model.eval()
        pooled_shapes = []
        handles = [
            pool.register_forward_hook(
                lambda _module, _inputs, output: pooled_shapes.append(output.shape)
            )
            for pool in model.scale_pools
        ]
        try:
            logits = model(torch.randn(1, 1, 64, 64))
        finally:
            for handle in handles:
                handle.remove()
        nn.BCEWithLogitsLoss()(logits, torch.zeros(1, 10)).backward()

        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(pooled_shapes, [torch.Size((1, 256, 8, 8))] * 3)
        self.assertIsInstance(model.dropout, nn.Dropout2d)
        self.assertIsInstance(model.classifier, ClassAttentionHead)
        self.assertEqual(model.classifier.classifier.shape, (10, 256))

    def test_densenet_attention_v2_checkpoint_round_trip(self) -> None:
        original = build_classifier("densenet_atn_head_v2", dropout=0.15)
        checkpoint = {
            "model_name": "densenet_atn_head_v2",
            "model_config": {"num_classes": 10, "dropout": 0.15},
            "model_state": original.state_dict(),
        }

        restored, model_name = build_classifier_from_checkpoint(checkpoint)
        restored.load_state_dict(checkpoint["model_state"])

        self.assertEqual(model_name, "densenet_atn_head_v2")
        self.assertEqual(restored.dropout_probability, 0.15)
        self.assertEqual(count_parameters(restored), count_parameters(original))

    def test_training_cli_uses_model_name(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["src_model_cls.train", "--model-name", "small"],
        ):
            args = parse_args()
        self.assertEqual(args.model_name, "small")

        with patch.object(
            sys,
            "argv",
            ["src_model_cls.train", "--model-size", "small"],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()


if __name__ == "__main__":
    unittest.main()
