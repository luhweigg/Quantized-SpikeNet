import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional
from src.domain.ports import ISNNModel


class BaseSNNModel(nn.Module, ISNNModel):
    """
    Base class for Spiking Neural Network models.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).mean(dim=0)

    def reset_states(self) -> None:
        functional.reset_net(self)


class SpikingConvBlock(nn.Sequential):
    """
    Convolutive block with optional batch normalization, max pooling, and LIF neuron activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = False,
        use_max_pool: bool = True,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate_func=None,
    ):
        layers = [
            layer.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batch_norm,
            )
        ]

        if use_batch_norm:
            layers.append(layer.BatchNorm2d(out_channels))

        if surrogate_func is not None:
            layers.append(
                neuron.LIFNode(
                    tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_func
                )
            )
        else:
            layers.append(neuron.LIFNode(tau=tau, v_threshold=v_threshold))

        if use_max_pool:
            layers.append(layer.MaxPool2d(2, 2))

        super().__init__(*layers)


class SpikingResidualBlock(nn.Module):
    """
    Bloc résiduel pour architectures ResNet SNN.
    L'addition s'effectue avant l'activation SNN finale.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, surrogate_func=None
    ):
        super().__init__()
        self.conv1 = layer.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = layer.BatchNorm2d(out_channels)
        self.node1 = (
            neuron.LIFNode(surrogate_function=surrogate_func)
            if surrogate_func
            else neuron.LIFNode()
        )

        self.conv2 = layer.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = layer.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                layer.BatchNorm2d(out_channels),
            )

        self.node2 = (
            neuron.LIFNode(surrogate_function=surrogate_func)
            if surrogate_func
            else neuron.LIFNode()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.node1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.node2(out)
        return out
