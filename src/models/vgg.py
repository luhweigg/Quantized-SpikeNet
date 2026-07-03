import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from .blocks import SpikingConvBlock, SpikingLinearBlock

class SpikingVGG(nn.Module):
    def __init__(self, vgg_name: str, in_channels: int = 3, num_classes: int = 10, neuron_type: str = "LIF", **neuron_kwargs):
        super().__init__()
        self.cfg = {
            'VGG5': [64, 'M', 128, 'M', 256, 'M'],
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        if vgg_name not in self.cfg:
            raise ValueError(f"VGG configuration '{vgg_name}' not supported.")
            
        self.features = self._make_layers(self.cfg[vgg_name], in_channels, neuron_type, **neuron_kwargs)
        classifier_input_dim = 256 if vgg_name == 'VGG5' else 512
        
        self.classifier = nn.Sequential(
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            SpikingLinearBlock(classifier_input_dim, 4096, neuron_type, **neuron_kwargs),
            SpikingLinearBlock(4096, 4096, neuron_type, **neuron_kwargs),
            layer.Linear(4096, num_classes)
        )
        functional.set_step_mode(self, step_mode="m")

    def _make_layers(self, cfg, in_channels, neuron_type, **neuron_kwargs):
        layers = []
        current_channels = in_channels
        for x in cfg:
            if x == 'M':
                layers.append(layer.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(SpikingConvBlock(current_channels, x, kernel_size=3, padding=1, neuron_type=neuron_type, **neuron_kwargs))
                current_channels = x
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.mean(0)

def spiking_vgg5(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG5', in_channels, num_classes, neuron_type, **kwargs)

def spiking_vgg11(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG11', in_channels, num_classes, neuron_type, **kwargs)

def spiking_vgg13(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG13', in_channels, num_classes, neuron_type, **kwargs)

def spiking_vgg16(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG16', in_channels, num_classes, neuron_type, **kwargs)

def spiking_vgg19(in_channels=3, num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG19', in_channels, num_classes, neuron_type, **kwargs)