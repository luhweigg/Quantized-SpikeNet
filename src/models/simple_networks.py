import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from .blocks import SpikingConvBlock, BaseSNNModel


class SpikingMLP(BaseSNNModel):
    """
    Simple MLP architecture for SNN.
    """

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.5
    ):
        super().__init__()
        self.network = nn.Sequential(
            layer.Flatten(),
            layer.Linear(input_size, hidden_size, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(dropout),
            layer.Linear(hidden_size, output_size, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")


class CompactSpikingCNN(BaseSNNModel):
    """
    SNN Architecture for small resolution images
    """

    def __init__(self, in_channels: int, out_classes: int, dropout: float = 0.4):
        super().__init__()
        self.network = nn.Sequential(
            SpikingConvBlock(in_channels, 32),
            SpikingConvBlock(32, 64),
            SpikingConvBlock(64, 128),
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(128, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")
