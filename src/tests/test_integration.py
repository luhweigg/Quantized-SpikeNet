import torch

from src.engine import quantize_weights
from src.models import CompactSpikingCNN, SpikingMLP, SpikingVGG5


def test_deepconv_snn_accepts_downsampled_cifar10_shape():
    model = SpikingVGG5(in_channels=2, out_classes=10)
    x = torch.zeros(4, 2, 2, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 10)


def test_quantization_returns_metadata():
    model = SpikingMLP(input_size=2312, hidden_size=256, output_size=10)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    weights, metadata = quantize_weights(model, return_metadata=True)

    assert weights
    assert metadata
    for name, tensor in weights.items():
        assert tensor.dtype == torch.int8
        assert name in metadata
        assert metadata[name]["qmin"] == -128
        assert metadata[name]["qmax"] == 127
        assert metadata[name]["zero_point"] == 0


def test_cifar_compact_and_deep_shapes_match_for_cifar10():
    compact = CompactSpikingCNN(in_channels=2, out_classes=10)
    deep = SpikingVGG5(in_channels=2, out_classes=10)
    x = torch.zeros(4, 2, 2, 32, 32)

    with torch.no_grad():
        out_compact = compact(x)
        out_deep = deep(x)

    assert out_compact.shape == (2, 10)
    assert out_deep.shape == (2, 10)


def test_cifar_compact_has_fewer_parameters_than_deep():
    compact = CompactSpikingCNN(in_channels=2, out_classes=10)
    deep = SpikingVGG5(in_channels=2, out_classes=10)

    compact_params = sum(p.numel() for p in compact.parameters())
    deep_params = sum(p.numel() for p in deep.parameters())

    assert compact_params < deep_params
