import torch.nn as nn
from .blocks import SpikingConvBlock, SpikingLinearBlock

class SpikingVGG11(nn.Module):
    def __init__(self, num_classes=10, neuron_type="LIF", **neuron_kwargs):
        super().__init__()
        self.features = nn.Sequential(
            SpikingConvBlock(3, 64, 3, 1, 1, neuron_type, **neuron_kwargs),
            nn.MaxPool2d(2, 2),
            SpikingConvBlock(64, 128, 3, 1, 1, neuron_type, **neuron_kwargs),
            nn.MaxPool2d(2, 2),
            SpikingConvBlock(128, 256, 3, 1, 1, neuron_type, **neuron_kwargs),
            SpikingConvBlock(256, 256, 3, 1, 1, neuron_type, **neuron_kwargs),
            nn.MaxPool2d(2, 2),
            SpikingConvBlock(256, 512, 3, 1, 1, neuron_type, **neuron_kwargs),
            SpikingConvBlock(512, 512, 3, 1, 1, neuron_type, **neuron_kwargs),
            nn.MaxPool2d(2, 2),
            SpikingConvBlock(512, 512, 3, 1, 1, neuron_type, **neuron_kwargs),
            SpikingConvBlock(512, 512, 3, 1, 1, neuron_type, **neuron_kwargs),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            SpikingLinearBlock(512, 4096, neuron_type, **neuron_kwargs),
            SpikingLinearBlock(4096, 4096, neuron_type, **neuron_kwargs),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                T, B, C, H, W = x.shape
                x = layer(x.flatten(0, 1)).reshape(T, B, C, H // 2, W // 2).contiguous()
            else:
                x = layer(x)

        x = x.flatten(2)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                T, B, C = x.shape
                x = layer(x.flatten(0, 1)).reshape(T, B, -1).contiguous()
            else:
                x = layer(x)
        return x