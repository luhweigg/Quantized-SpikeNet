import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from .neurons import NeuronFactory

class SpikingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, neuron_type="LIF", **neuron_kwargs):
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.neuron = NeuronFactory.build(neuron_type, **neuron_kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.neuron(x)

class SpikingLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, neuron_type="LIF", **neuron_kwargs):
        super().__init__()
        self.linear = layer.Linear(in_features, out_features, bias=False)
        self.bn = layer.BatchNorm1d(out_features)
        self.neuron = NeuronFactory.build(neuron_type, **neuron_kwargs)

    def forward(self, x):
        x = self.linear(x)
        
        if x.dim() == 3:
            x = x.unsqueeze(-1)
            x = self.bn(x)
            x = self.neuron(x)
            x = x.squeeze(-1)
        else:
            x = self.bn(x)
            x = self.neuron(x)
            
        return x