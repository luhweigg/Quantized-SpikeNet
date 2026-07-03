# src/models/simple.py

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from .blocks import SpikingConvBlock, SpikingLinearBlock

class SpikingMLP(nn.Module):
    """
    Simple MLP architecture for SNN, adapted with NeuronFactory.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.5, neuron_type: str = "LIF", **neuron_kwargs):
        super().__init__()
        self.network = nn.Sequential(
            layer.Flatten(start_dim=2),
            SpikingLinearBlock(input_size, hidden_size, neuron_type=neuron_type, **neuron_kwargs),
            layer.Dropout(dropout),
            layer.Linear(hidden_size, output_size, bias=True)
        )
        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).mean(dim=0)

class CompactSpikingCNN(nn.Module):
    """
    Compact CNN architecture for quick SNN tests, adapted with NeuronFactory.
    """
    def __init__(self, input_channels: int = 1, num_classes: int = 10, neuron_type: str = "LIF", **neuron_kwargs):
        super().__init__()
        self.features = nn.Sequential(
            SpikingConvBlock(input_channels, 32, kernel_size=3, padding=1, neuron_type=neuron_type, **neuron_kwargs),
            layer.MaxPool2d(2, 2),
            SpikingConvBlock(32, 64, kernel_size=3, padding=1, neuron_type=neuron_type, **neuron_kwargs),
            layer.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            layer.Flatten(),
            SpikingLinearBlock(64 * 7 * 7, 512, neuron_type=neuron_type, **neuron_kwargs),
            layer.Linear(512, num_classes)
        )
        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x