import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from src.domain.ports import ISNNModel

class GenericMLPSNN(nn.Module, ISNNModel):
    """
    Architecture MLP générique pour SNN.
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

class DeepConvSNN(nn.Module, ISNNModel):
    """
    Architecture convolutive profonde (type VGG) pour SNN.
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

            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(512 * 4 * 4, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).mean(dim=0)
        
    def reset_states(self) -> None:
        functional.reset_net(self)