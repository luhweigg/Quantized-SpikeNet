import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate, functional
from .blocks import BaseSNNModel, SpikingResNetStage, SpikingResNetStem


class SpikingResNet18(BaseSNNModel):
    """
    Architecture Spiking ResNet-18.
    """

    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        dropout: float = 0.5,
        init_stride: int = 2,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        sg = surrogate.ATan(alpha=2.0)

        self.network = nn.Sequential(
            SpikingResNetStem(in_channels, init_stride, sg, v_threshold),
            SpikingResNetStage(64, 64, 2, stride=1, surrogate_func=sg, v_threshold=v_threshold),
            SpikingResNetStage(64, 128, 2, stride=2, surrogate_func=sg, v_threshold=v_threshold),
            SpikingResNetStage(128, 256, 2, stride=2, surrogate_func=sg, v_threshold=v_threshold),
            SpikingResNetStage(256, 512, 2, stride=2, surrogate_func=sg, v_threshold=v_threshold),
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
            SpikingResNetStem(in_channels, init_stride, sg, v_threshold),
            SpikingResNetStage(64, 64, 3, stride=1, surrogate_func=sg, v_threshold=v_threshold),
            SpikingResNetStage(64, 128, 4, stride=2, surrogate_func=sg, v_threshold=v_threshold),
            SpikingResNetStage(128, 256, 6, stride=2, surrogate_func=sg, v_threshold=v_threshold),
            SpikingResNetStage(256, 512, 3, stride=2, surrogate_func=sg, v_threshold=v_threshold),
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(512, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")
