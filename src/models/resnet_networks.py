import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate, functional
from .blocks import BaseSNNModel, SpikingConvBlock, SpikingResidualBlock


class SpikingResNet18(BaseSNNModel):
    """
    Architecture Spiking ResNet-18.
    """

    def __init__(self, in_channels: int, out_classes: int, dropout: float = 0.5):
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
            ),
            SpikingResidualBlock(64, 64, stride=1, surrogate_func=sg),
            SpikingResidualBlock(64, 64, stride=1, surrogate_func=sg),
            SpikingResidualBlock(64, 128, stride=2, surrogate_func=sg),
            SpikingResidualBlock(128, 128, stride=1, surrogate_func=sg),
            SpikingResidualBlock(128, 256, stride=2, surrogate_func=sg),
            SpikingResidualBlock(256, 256, stride=1, surrogate_func=sg),
            SpikingResidualBlock(256, 512, stride=2, surrogate_func=sg),
            SpikingResidualBlock(512, 512, stride=1, surrogate_func=sg),
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(512, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")
