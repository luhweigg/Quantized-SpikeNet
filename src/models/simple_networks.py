from spikingjelly.activation_based import functional, layer, neuron, surrogate
from torch import nn

from .blocks import BaseSNNModel, SpikingConvBlock


class SpikingMLP(BaseSNNModel):
    """
    Simple MLP architecture for SNN.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        self.network = nn.Sequential(
            layer.Flatten(),
            layer.Linear(input_size, hidden_size, bias=False),
            neuron.LIFNode(
                surrogate_function=surrogate.ATan(), v_threshold=v_threshold
            ),
            layer.Dropout(dropout),
            layer.Linear(hidden_size, output_size, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")


class CompactSpikingCNN(BaseSNNModel):
    """
    SNN Architecture for small resolution images
    """

    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        dropout: float = 0.4,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        self.network = nn.Sequential(
            SpikingConvBlock(in_channels, 32, v_threshold=v_threshold),
            SpikingConvBlock(32, 64, v_threshold=v_threshold),
            SpikingConvBlock(64, 128, v_threshold=v_threshold),
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(128, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")


class Spiking1DCNN(BaseSNNModel):
    """
    1D CNN architecture specialized for event-based audio processing (SHD).
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
            layer.Conv1d(
                in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False
            ),
            layer.BatchNorm1d(32),
            neuron.LIFNode(surrogate_function=sg, v_threshold=v_threshold),
            layer.MaxPool1d(2),
            layer.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            layer.BatchNorm1d(64),
            neuron.LIFNode(surrogate_function=sg, v_threshold=v_threshold),
            layer.MaxPool1d(2),
            layer.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm1d(128),
            neuron.LIFNode(surrogate_function=sg, v_threshold=v_threshold),
            layer.AdaptiveAvgPool1d(1),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(128, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")
