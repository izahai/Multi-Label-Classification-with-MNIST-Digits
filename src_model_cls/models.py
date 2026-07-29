"""CNN architectures for multi-label MNIST classification."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAMES = (
    "small",
    "large",
    "dense_net",
    "densenet_atn_head",
    "densenet_atn_head_v2",
    # Keep the spelling from the original experiment request usable in the CLI.
    "denset_net_atn_head",
)
DEFAULT_MODEL_NAME = "small"
MODEL_DEFAULT_DROPOUTS = {
    "small": 0.2,
    "large": 0.2,
    "dense_net": 0.3,
    "densenet_atn_head": 0.3,
    "densenet_atn_head_v2": 0.3,
    "denset_net_atn_head": 0.3,
}


class ResidualBlock(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.shortcut(inputs))


class MNISTPresenceClassifier(nn.Module):
    """Residual feature extractor followed by a 10-logit presence head."""

    def __init__(
        self,
        widths: tuple[int, ...],
        blocks_per_stage: tuple[int, ...],
        *,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(widths) != len(blocks_per_stage):
            raise ValueError("widths and blocks_per_stage must have equal lengths.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.num_classes = num_classes
        self.dropout_probability = dropout
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(inplace=True),
        )

        stages: list[nn.Module] = []
        in_channels = widths[0]
        for stage_index, (out_channels, block_count) in enumerate(
            zip(widths, blocks_per_stage)
        ):
            if block_count < 1:
                raise ValueError("Every stage must contain at least one block.")
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stages.append(ResidualBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(widths[-1], num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(images))
        pooled = self.pool(features).flatten(1)
        return self.classifier(self.dropout(pooled))


class DenseLayer(nn.Module):
    """DenseNet-BC bottleneck layer that appends newly generated features."""

    def __init__(
        self,
        in_channels: int,
        *,
        growth_rate: int,
        bottleneck_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        bottleneck_channels = bottleneck_factor * growth_rate
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck_channels,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        new_features = self.dropout(self.layers(features))
        return torch.cat((features, new_features), dim=1)


class DenseBlock(nn.Module):
    """Sequence of densely connected bottleneck layers."""

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        *,
        growth_rate: int,
        bottleneck_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(
                DenseLayer(
                    current_channels,
                    growth_rate=growth_rate,
                    bottleneck_factor=bottleneck_factor,
                    dropout=dropout,
                )
            )
            current_channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class TransitionBlock(nn.Sequential):
    """Compress channels and halve spatial resolution between dense blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )


class ClassAttentionHead(nn.Module):
    """Class-specific spatial attention pooling for multi-label logits."""

    def __init__(self, channels: int, num_classes: int = 10) -> None:
        super().__init__()
        self.attention = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.classifier = nn.Parameter(torch.empty(num_classes, channels))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.classifier, std=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Pool a separate feature vector for each class using attention."""
        attention = self.attention(features).flatten(2).softmax(dim=-1)
        spatial_features = features.flatten(2)
        pooled = torch.einsum("bkh,bch->bkc", attention, spatial_features)
        return (pooled * self.classifier).sum(dim=-1) + self.bias


class AdaptiveGeMPool2d(nn.Module):
    """Adaptive generalized-mean pooling with a learnable exponent."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        if p <= 0.0:
            raise ValueError("p must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        output_size: int | tuple[int, int] = 1,
    ) -> torch.Tensor:
        p = self.p.clamp_min(self.eps)
        pooled = F.adaptive_avg_pool2d(
            features.clamp_min(self.eps).pow(p),
            output_size,
        )
        return pooled.pow(p.reciprocal())


class DenseNet121PresenceClassifier(nn.Module):
    """Small-image DenseNet-BC-121 followed by a multi-label logits head."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        block_config = (6, 12, 24, 16)
        growth_rate = 32
        initial_filters = 64
        bottleneck_factor = 4
        compression = 0.5
        dense_dropout = 0.0

        modules: list[nn.Module] = [
            nn.Conv2d(
                1,
                initial_filters,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        ]
        current_channels = initial_filters
        for block_index, num_layers in enumerate(block_config):
            block = DenseBlock(
                current_channels,
                num_layers,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                dropout=dense_dropout,
            )
            modules.append(block)
            current_channels = block.out_channels
            if block_index < len(block_config) - 1:
                compressed_channels = max(
                    int(current_channels * compression),
                    growth_rate,
                )
                modules.append(
                    TransitionBlock(current_channels, compressed_channels)
                )
                current_channels = compressed_channels

        modules.extend(
            [
                nn.BatchNorm2d(current_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.num_classes = num_classes
        self.dropout_probability = dropout
        self.features = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        return self.classifier(self.dropout(pooled))


class DenseNet121AttentionPresenceClassifier(DenseNet121PresenceClassifier):
    """DenseNet-BC-121 with a class-specific spatial attention head."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout)
        channels = self.classifier.in_features
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = ClassAttentionHead(channels, num_classes)
        nn.init.kaiming_normal_(
            self.classifier.attention.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.classifier.attention.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.features(images))
        return self.classifier(features)


class DenseNet121MultiScaleAttentionPresenceClassifier(
    DenseNet121PresenceClassifier
):
    """DenseNet-BC-121 with GeM multi-scale fusion and class attention."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        dropout: float = 0.3,
        projection_channels: int = 256,
    ) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout)
        if projection_channels < 1:
            raise ValueError("projection_channels must be positive.")

        self.projection_channels = projection_channels
        scale_channels = (512, 1024, 1024)
        self.scale_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        projection_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(projection_channels),
                    nn.ReLU(inplace=True),
                )
                for channels in scale_channels
            ]
        )
        self.scale_pools = nn.ModuleList(
            [AdaptiveGeMPool2d() for _ in scale_channels]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                len(scale_channels) * projection_channels,
                projection_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(projection_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = ClassAttentionHead(projection_channels, num_classes)
        self._initialize_fusion_weights()

    def _initialize_fusion_weights(self) -> None:
        for module in [*self.scale_projections, self.fusion]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(
            self.classifier.attention.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.classifier.attention.bias)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Fuse projected outputs from dense blocks two, three, and four."""
        features = images
        scale_features = []
        for layer_index, layer in enumerate(self.features):
            features = layer(features)
            if layer_index in (3, 5, 9):
                scale_features.append(features)

        target_size = scale_features[-1].shape[-2:]
        pooled_features = [
            pool(project(scale), target_size)
            for scale, project, pool in zip(
                scale_features,
                self.scale_projections,
                self.scale_pools,
            )
        ]
        return self.fusion(torch.cat(pooled_features, dim=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.forward_features(images))
        return self.classifier(features)


def build_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    num_classes: int = 10,
    dropout: float | None = None,
) -> nn.Module:
    """Build one of the supported classifier architectures."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {MODEL_NAMES}.")
    if dropout is None:
        dropout = MODEL_DEFAULT_DROPOUTS[model_name]

    if model_name == "small":
        return MNISTPresenceClassifier(
            (12, 24, 48, 72),
            (1, 1, 1, 1),
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_name == "large":
        return MNISTPresenceClassifier(
            (28, 56, 112, 168),
            (2, 2, 2, 2),
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_name == "dense_net":
        return DenseNet121PresenceClassifier(
            num_classes=num_classes,
            dropout=dropout,
        )
    if model_name in {"densenet_atn_head", "denset_net_atn_head"}:
        return DenseNet121AttentionPresenceClassifier(
            num_classes=num_classes,
            dropout=dropout,
        )
    return DenseNet121MultiScaleAttentionPresenceClassifier(
        num_classes=num_classes,
        dropout=dropout,
    )


def build_classifier_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[nn.Module, str]:
    """Restore the architecture recorded in a classifier checkpoint."""
    model_name = str(checkpoint["model_name"])
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Checkpoint contains unknown model name {model_name!r}.")
    config = checkpoint.get("model_config", {})
    saved_dropout = config.get("dropout")
    model = build_classifier(
        model_name,
        num_classes=int(config.get("num_classes", 10)),
        dropout=None if saved_dropout is None else float(saved_dropout),
    )
    return model, model_name


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
