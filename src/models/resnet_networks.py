import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate, functional
from .blocks import BaseSNNModel, SpikingConvBlock, SpikingResidualBlock


class SpikingResNet18(BaseSNNModel):
    """
    Architecture Spiking ResNet-18.
    """

    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        dropout: float = 0.5,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        sg = surrogate.ATan(alpha=2.0)

        self.network = nn.Sequential(
            SpikingConvBlock(
                in_channels,
                64,
                kernel_size=7,
                padding=3,
                use_batch_norm=True,
                use_max_pool=True,
                surrogate_func=sg,
                v_threshold=v_threshold,
            ),
            SpikingResidualBlock(
                64, 64, stride=1, surrogate_func=sg, v_threshold=v_threshold
            ),
            SpikingResidualBlock(
                64, 64, stride=1, surrogate_func=sg, v_threshold=v_threshold
            ),
            SpikingResidualBlock(
                64, 128, stride=2, surrogate_func=sg, v_threshold=v_threshold
            ),
            SpikingResidualBlock(
                128, 128, stride=1, surrogate_func=sg, v_threshold=v_threshold
            ),
            SpikingResidualBlock(
                128, 256, stride=2, surrogate_func=sg, v_threshold=v_threshold
            ),
            SpikingResidualBlock(
                256, 256, stride=1, surrogate_func=sg, v_threshold=v_threshold
            ),
            SpikingResidualBlock(
                256, 512, stride=2, surrogate_func=sg, v_threshold=v_threshold
            ),
            SpikingResidualBlock(
                512, 512, stride=1, surrogate_func=sg, v_threshold=v_threshold
            ),
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(512, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")


class SpikingResNet34(BaseSNNModel):
    """
    ResNet-34 Spiking architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        dropout: float = 0.5,
        init_stride: int = 4,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        sg = surrogate.ATan(alpha=2.0)

        self.network = nn.Sequential(
            SpikingConvBlock(
                in_channels,
                64,
                kernel_size=7,
                stride=init_stride,
                padding=3,
                use_batch_norm=True,
                use_max_pool=True,
                surrogate_func=sg,
                v_threshold=v_threshold,
            ),
            self._make_layer(64, 64, 3, stride=1, sg=sg, v_threshold=v_threshold),
            self._make_layer(64, 128, 4, stride=2, sg=sg, v_threshold=v_threshold),
            self._make_layer(128, 256, 6, stride=2, sg=sg, v_threshold=v_threshold),
            self._make_layer(256, 512, 3, stride=2, sg=sg, v_threshold=v_threshold),
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(512, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")

    def _make_layer(self, in_channels, out_channels, blocks, stride, sg, v_threshold):
        layers = []
        layers.append(
            SpikingResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                surrogate_func=sg,
                v_threshold=v_threshold,
            )
        )
        for _ in range(1, blocks):
            layers.append(
                SpikingResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    surrogate_func=sg,
                    v_threshold=v_threshold,
                )
            )
        return nn.Sequential(*layers)
