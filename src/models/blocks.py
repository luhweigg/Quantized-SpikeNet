import torch.nn as nn
from spikingjelly.activation_based import layer
from .neurons import NeuronFactory

class SpikingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, neuron_type="LIF", **neuron_kwargs):
        super().__init__()
        self.block = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            layer.BatchNorm2d(out_channels),
            NeuronFactory.build(neuron_type, **neuron_kwargs)
        )

    def forward(self, x):
        return self.block(x)

class SpikingLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, neuron_type="LIF", **neuron_kwargs):
        super().__init__()
        self.block = nn.Sequential(
            layer.Linear(in_features, out_features, bias=False),
            layer.BatchNorm1d(out_features),
            NeuronFactory.build(neuron_type, **neuron_kwargs)
        )

    def forward(self, x):
        return self.block(x)