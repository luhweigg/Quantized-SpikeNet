import json
from pathlib import Path

from torch import nn

from .resnet_networks import SpikingResNet18, SpikingResNet34
from .simple_networks import CompactSpikingCNN, Spiking1DCNN, SpikingMLP
from .vgg_networks import SpikingVGG3, SpikingVGG4, SpikingVGG5, SpikingVGG8

ARCHITECTURES = {
    "SpikingMLP": SpikingMLP,
    "CompactSpikingCNN": CompactSpikingCNN,
    "Spiking1DCNN": Spiking1DCNN,
    "SpikingVGG3": SpikingVGG3,
    "SpikingVGG4": SpikingVGG4,
    "SpikingVGG5": SpikingVGG5,
    "SpikingVGG8": SpikingVGG8,
    "SpikingResNet18": SpikingResNet18,
    "SpikingResNet34": SpikingResNet34,
}

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def get_model(
    architecture: str,
    dataset: str,
    v_threshold: float | None = None,
    config_dir: str | Path | None = None,
    **overrides,
) -> nn.Module:
    """
    Build an architecture with the hyper-parameters stored in configs/<dataset>.json.

    The firing threshold is not part of a state_dict, so it has to be given here to
    rebuild a trained model faithfully.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Available: {sorted(ARCHITECTURES)}"
        )

    config_path = Path(config_dir or DEFAULT_CONFIG_DIR) / f"{dataset}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        architectures = json.load(f)["architectures"]

    if architecture not in architectures:
        raise ValueError(
            f"Architecture '{architecture}' is not configured for dataset '{dataset}'."
        )

    params = dict(architectures[architecture])
    if v_threshold is not None:
        params["v_threshold"] = float(v_threshold)
    params.update(overrides)

    return ARCHITECTURES[architecture](**params)
