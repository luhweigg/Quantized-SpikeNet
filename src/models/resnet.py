# src/models/resnet.py

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from .blocks import SpikingConvBlock

class SpikingBasicBlock(nn.Module):
    """
    Spiking Basic Block for ResNet18 and ResNet34.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, neuron_type="LIF", **neuron_kwargs):
        super().__init__()
        self.conv1 = SpikingConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, neuron_type=neuron_type, **neuron_kwargs)
        self.conv2 = SpikingConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, neuron_type=neuron_type, **neuron_kwargs)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            # The shortcut must also be a spiking block to align temporal dynamics.
            self.shortcut = SpikingConvBlock(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, neuron_type=neuron_type, **neuron_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.shortcut(x)

class SpikingBottleneck(nn.Module):
    """
    Spiking Bottleneck Block for ResNet50, ResNet101, and ResNet152.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, neuron_type="LIF", **neuron_kwargs):
        super().__init__()
        self.conv1 = SpikingConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, neuron_type=neuron_type, **neuron_kwargs)
        self.conv2 = SpikingConvBlock(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, neuron_type=neuron_type, **neuron_kwargs)
        self.conv3 = SpikingConvBlock(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, padding=0, neuron_type=neuron_type, **neuron_kwargs)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = SpikingConvBlock(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0, neuron_type=neuron_type, **neuron_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + self.shortcut(x)

class SpikingResNet(nn.Module):
    """
    Generic Spiking ResNet architecture adapted for S4 SSL temporal dynamics.
    """
    def __init__(self, block, num_blocks, in_channels=3, num_classes=10, neuron_type="LIF", **neuron_kwargs):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolutional layer. No MaxPool here as it degrades early SNN information (especially on low-res images like CIFAR).
        self.conv1 = SpikingConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1, neuron_type=neuron_type, **neuron_kwargs)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, neuron_type=neuron_type, **neuron_kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, neuron_type=neuron_type, **neuron_kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, neuron_type=neuron_type, **neuron_kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, neuron_type=neuron_type, **neuron_kwargs)
        
        # We use spikingjelly wrappers to process [T, B, C, H, W] automatically.
        self.pool = layer.AdaptiveAvgPool2d((1, 1))
        self.flatten = layer.Flatten()
        self.linear = layer.Linear(512 * block.expansion, num_classes)
        functional.set_step_mode(self, step_mode="m")

    def _make_layer(self, block, out_channels, num_blocks, stride, neuron_type, **neuron_kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, neuron_type, **neuron_kwargs))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

def spiking_resnet18(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingResNet(SpikingBasicBlock, [2, 2, 2, 2], in_channels, num_classes, neuron_type, **kwargs)

def spiking_resnet34(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingResNet(SpikingBasicBlock, [3, 4, 6, 3], in_channels, num_classes, neuron_type, **kwargs)

def spiking_resnet50(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingResNet(SpikingBottleneck, [3, 4, 6, 3], in_channels, num_classes, neuron_type, **kwargs)

def spiking_resnet101(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingResNet(SpikingBottleneck, [3, 4, 23, 3], in_channels, num_classes, neuron_type, **kwargs)