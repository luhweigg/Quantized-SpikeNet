import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from .blocks import SpikingConvBlock, BaseSNNModel


class SpikingVGG5(BaseSNNModel):
    """
    Deep convolutional SNN architecture (5 block, VGG-type) for SNN.
    """

    def __init__(self, in_channels: int, out_classes: int, dropout: float = 0.5, init_stride: int = 1):
        super().__init__()
        sg = surrogate.ATan(alpha=1.5)
        self.network = nn.Sequential(
            SpikingConvBlock(
                in_channels, 64, stride=init_stride, use_batch_norm=False, surrogate_func=sg
            ),
            SpikingConvBlock(64, 128, use_batch_norm=True, surrogate_func=sg),
            SpikingConvBlock(128, 256, use_batch_norm=True, surrogate_func=sg),
            SpikingConvBlock(256, 256, use_batch_norm=True, surrogate_func=sg),
            SpikingConvBlock(256, 512, use_batch_norm=True, surrogate_func=sg),
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(512, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")


class SpikingVGG11(BaseSNNModel):
    """
    Deeper VGG-11 SNN architecture adapted for complex backgrounds and high ego-motion (N-EPIC Kitchens).
    """

    def __init__(self, in_channels: int, out_classes: int, dropout: float = 0.5, init_stride: int = 1):
        super().__init__()
        sg = surrogate.ATan(alpha=2.0)
        v_th = 0.5

        self.network = nn.Sequential(
            SpikingConvBlock(
                in_channels,
                64,
                stride=init_stride,
                use_batch_norm=True,
                v_threshold=v_th,
                surrogate_func=sg,
            ),
            SpikingConvBlock(
                64, 128, use_batch_norm=True, v_threshold=v_th, surrogate_func=sg
            ),
            SpikingConvBlock(
                128,
                256,
                use_batch_norm=True,
                use_max_pool=False,
                v_threshold=v_th,
                surrogate_func=sg,
            ),
            SpikingConvBlock(
                256, 256, use_batch_norm=True, v_threshold=v_th, surrogate_func=sg
            ),
            SpikingConvBlock(
                256,
                512,
                use_batch_norm=True,
                use_max_pool=False,
                v_threshold=v_th,
                surrogate_func=sg,
            ),
            SpikingConvBlock(
                512, 512, use_batch_norm=True, v_threshold=v_th, surrogate_func=sg
            ),
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(512, 4096, bias=True),
            neuron.LIFNode(tau=2.0, v_threshold=v_th, surrogate_function=sg),
            layer.Dropout(dropout),
            layer.Linear(4096, out_classes, bias=True),
        )
        functional.set_step_mode(self, step_mode="m")
