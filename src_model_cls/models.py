"""CNN architectures for multi-label MNIST classification."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAMES = ("small", "large", "dense_net", "ConvNeXt")
DEFAULT_MODEL_NAME = "small"
MODEL_DEFAULT_DROPOUTS = {
    "small": 0.2,
    "large": 0.2,
    "dense_net": 0.3,
    "ConvNeXt": 0.0,
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


class ConvNeXtLayerNorm(nn.Module):
    """Layer normalization for channels-last or channels-first tensors."""

    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"Unsupported data format: {data_format!r}.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = normalized_shape

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                inputs,
                (self.normalized_shape,),
                self.weight,
                self.bias,
                self.eps,
            )

        mean = inputs.mean(dim=1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        return (
            self.weight[:, None, None] * normalized
            + self.bias[:, None, None]
        )


def drop_path(
    inputs: torch.Tensor,
    drop_probability: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Drop complete residual paths independently for each sample."""
    if drop_probability == 0.0 or not training:
        return inputs
    keep_probability = 1.0 - drop_probability
    shape = (inputs.shape[0], *([1] * (inputs.ndim - 1)))
    random_tensor = keep_probability + torch.rand(
        shape,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    random_tensor.floor_()
    return inputs * random_tensor / keep_probability


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_probability < 1.0:
            raise ValueError("drop_probability must be in [0, 1).")
        self.drop_probability = drop_probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return drop_path(inputs, self.drop_probability, self.training)


class GlobalResponseNorm(nn.Module):
    """ConvNeXt V2 global response normalization for NHWC tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        response = torch.norm(inputs, p=2, dim=(1, 2), keepdim=True)
        normalized_response = response / (
            response.mean(dim=-1, keepdim=True) + self.eps
        )
        return inputs + self.gamma * (inputs * normalized_response) + self.beta


class ConvNeXtV2Block(nn.Module):
    """Depthwise ConvNeXt V2 block with GRN and stochastic depth."""

    def __init__(self, dim: int, drop_path_probability: float = 0.0) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.expand = nn.Linear(dim, 4 * dim)
        self.activation = nn.GELU()
        self.grn = GlobalResponseNorm(4 * dim)
        self.project = nn.Linear(4 * dim, dim)
        self.drop_path = (
            DropPath(drop_path_probability)
            if drop_path_probability > 0.0
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.depthwise_conv(inputs)
        features = features.permute(0, 2, 3, 1)
        features = self.norm(features)
        features = self.expand(features)
        features = self.activation(features)
        features = self.grn(features)
        features = self.project(features)
        features = features.permute(0, 3, 1, 2)
        return inputs + self.drop_path(features)


class ConvNeXtV2PresenceClassifier(nn.Module):
    """ConvNeXt V2 Femto adapted to grayscale 64x64 presence prediction."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        dropout: float = 0.0,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        dims: tuple[int, ...] = (48, 96, 192, 384),
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt V2 requires four depths and four dims.")
        if any(depth < 1 for depth in depths):
            raise ValueError("Every ConvNeXt stage must contain at least one block.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1).")

        self.num_classes = num_classes
        self.dropout_probability = dropout
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, dims[0], kernel_size=4, stride=4),
                    ConvNeXtLayerNorm(
                        dims[0],
                        eps=1e-6,
                        data_format="channels_first",
                    ),
                )
            ]
        )
        for stage_index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    ConvNeXtLayerNorm(
                        dims[stage_index],
                        eps=1e-6,
                        data_format="channels_first",
                    ),
                    nn.Conv2d(
                        dims[stage_index],
                        dims[stage_index + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
            )

        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        block_index = 0
        for depth, dim in zip(depths, dims):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtV2Block(dim, drop_rates[block_index]))
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dims[-1], num_classes)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = images
        for downsample, stage in zip(self.downsample_layers, self.stages):
            features = stage(downsample(features))
        return self.norm(features.mean(dim=(-2, -1)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.forward_features(images)))


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
    return ConvNeXtV2PresenceClassifier(
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
