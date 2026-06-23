import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from src.domain.ports import ISNNModel

class SpikingMLP(nn.Module, ISNNModel):
    """
    Simple MLP architecture for SNN.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            layer.Flatten(),
            layer.Linear(input_size, hidden_size, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(dropout),
            layer.Linear(hidden_size, output_size, bias=True),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).mean(dim=0)
        
    def reset_states(self) -> None:
        functional.reset_net(self)

class CompactSpikingCNN(nn.Module, ISNNModel):
    """
    SNN Architecture for small resolution images 
    """
    def __init__(self, in_channels: int, out_classes: int, dropout: float = 0.4):
        super().__init__()
        self.network = nn.Sequential(
            layer.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=True),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),

            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(128, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).mean(dim=0)

    def reset_states(self) -> None:
        functional.reset_net(self)

class SpikingVGG5(nn.Module, ISNNModel):
    """
    Deep convolutional SNN architecture (5 block, VGG-type) for SNN.
    """
    def __init__(self, in_channels: int, out_classes: int, dropout: float = 0.5):
        super().__init__()
        self.network = nn.Sequential(
            layer.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
            neuron.LIFNode(),
            layer.MaxPool2d(2, 2),

            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(512, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).mean(dim=0)
        
    def reset_states(self) -> None:
        functional.reset_net(self)