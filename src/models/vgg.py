import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from .blocks import SpikingConvBlock, SpikingLinearBlock

class SpikingVGG(nn.Module):
    """
    Spiking version of the VGG architecture compatible with S4 SSL training.
    Supports dynamic generation of VGG5, VGG11, VGG13, VGG16, and VGG19 topologies.
    """
    def __init__(self, vgg_name: str, num_classes: int = 10, neuron_type: str = "LIF", **neuron_kwargs):
        super().__init__()
        self.cfg = {
            'VGG5' : [64, 'M', 128, 'M', 256, 'M', 256, 'M', 512, 'M'],
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        if vgg_name not in self.cfg:
            raise ValueError(f"VGG configuration '{vgg_name}' not supported.")
            
        self.features = self._make_layers(self.cfg[vgg_name], neuron_type, **neuron_kwargs)
        
        self.classifier = nn.Sequential(
            # layer.Flatten() natively handles [T, B, C, H, W] by keeping T and B intact 
            # and flattening the spatial dimensions (C, H, W).
            layer.Flatten(),
            SpikingLinearBlock(512, 4096, neuron_type, **neuron_kwargs),
            SpikingLinearBlock(4096, 4096, neuron_type, **neuron_kwargs),
            # The final layer is a spikingjelly Linear wrapper, not a PyTorch nn.Linear.
            # This ensures it outputs [T, B, num_classes] without crashing.
            layer.Linear(4096, num_classes)
        )
        functional.set_step_mode(self, step_mode="m")

    def _make_layers(self, cfg, neuron_type, **neuron_kwargs):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(layer.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(SpikingConvBlock(in_channels, x, kernel_size=3, padding=1, neuron_type=neuron_type, **neuron_kwargs))
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def spiking_vgg5(num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG5', num_classes, neuron_type, **kwargs)

def spiking_vgg11(num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG11', num_classes, neuron_type, **kwargs)

def spiking_vgg13(num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG13', num_classes, neuron_type, **kwargs)

def spiking_vgg16(num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG16', num_classes, neuron_type, **kwargs)

def spiking_vgg19(num_classes=10, neuron_type="LIF", **kwargs):
    return SpikingVGG('VGG19', num_classes, neuron_type, **kwargs)