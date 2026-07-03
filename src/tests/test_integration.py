import torch

from src.data_loaders import cifar10_dvs_loader
from src.engine import quantize_weights
from src.models import SpikingMLP, CompactSpikingCNN, spiking_vgg5


def test_deepconv_snn_accepts_downsampled_cifar10_shape():
    model = spiking_vgg5(in_channels=2, num_classes=10)
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


def test_cifar10_dvs_split_is_deterministic(monkeypatch):
    class DummyCIFAR10DVS(torch.utils.data.Dataset):
        def __init__(self, save_to=None, transform=None):
            self.transform = transform
            self.samples = list(range(10))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            sample = torch.zeros(2, 32, 32, 2)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, index

    monkeypatch.setattr(
        cifar10_dvs_loader.tonic.datasets, "CIFAR10DVS", DummyCIFAR10DVS
    )

    train_loader_a, test_loader_a = cifar10_dvs_loader.get_cifar10_loaders(
        batch_size=2, n_time_bins=4, num_workers=0, split_seed=123
    )
    train_loader_b, test_loader_b = cifar10_dvs_loader.get_cifar10_loaders(
        batch_size=2, n_time_bins=4, num_workers=0, split_seed=123
    )

    assert train_loader_a.dataset.indices == train_loader_b.dataset.indices
    assert test_loader_a.dataset.indices == test_loader_b.dataset.indices


def test_cifar_compact_and_deep_shapes_match_for_cifar10():
    compact = CompactSpikingCNN(in_channels=2, num_classes=10)
    deep = spiking_vgg5(in_channels=2, num_classes=10)
    x = torch.zeros(4, 2, 2, 32, 32)

    with torch.no_grad():
        out_compact = compact(x)
        out_deep = deep(x)

    assert out_compact.shape == (2, 10)
    assert out_deep.shape == (2, 10)


def test_cifar_compact_has_fewer_parameters_than_deep():
    compact = CompactSpikingCNN(in_channels=2, num_classes=10)
    deep = spiking_vgg5(in_channels=2, num_classes=10)

    compact_params = sum(p.numel() for p in compact.parameters())
    deep_params = sum(p.numel() for p in deep.parameters())

    assert compact_params < deep_params
