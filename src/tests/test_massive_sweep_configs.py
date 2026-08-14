import json
from pathlib import Path

import pytest
import torch

from src.engine import evaluate, train_one_epoch
from src.engine.builder import ARCHITECTURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

DATASET_SHAPES = {
    "cifar10_dvs": {"kind": "2d", "channels": 2, "height": 28, "width": 28},
    "dvs_gesture": {"kind": "2d", "channels": 2, "height": 128, "width": 128},
    "dvs_lip": {"kind": "2d", "channels": 2, "height": 128, "width": 128},
    "ncaltech101": {"kind": "2d", "channels": 2, "height": 128, "width": 128},
    "nepic_kitchens": {"kind": "2d", "channels": 2, "height": 128, "width": 128},
    "nmnist": {"kind": "2d", "channels": 2, "height": 34, "width": 34},
    "shd": {"kind": "1d", "channels": 1, "length": 700},
    "smnist": {"kind": "1d", "channels": 2, "length": 99},
    "ssc": {"kind": "1d", "channels": 1, "length": 700},
    "edat24": {"kind": "2d", "channels": 2, "height": 64, "width": 64},
}

MLP_INPUT_SIZES = {
    "cifar10_dvs": 1568,
    "dvs_gesture": 32768,
    "dvs_lip": 32768,
    "ncaltech101": 32768,
    "nepic_kitchens": 116736,
    "nmnist": 2312,
    "shd": 700,
}


class SingleBatchLoader:
    def __init__(self, events: torch.Tensor, targets: torch.Tensor):
        self._batch = (events, targets)

    def __iter__(self):
        yield self._batch

    def __len__(self):
        return 1


def _configured_pairs():
    pairs = []
    for config_path in sorted(CONFIG_DIR.glob("*.json")):
        dataset = config_path.stem
        config = json.loads(config_path.read_text())
        for architecture_name in config["architectures"]:
            pairs.append(
                pytest.param(
                    dataset, architecture_name, id=f"{dataset}-{architecture_name}"
                )
            )
    return pairs


def _build_events(
    dataset: str, architecture_name: str, architecture_params: dict
) -> torch.Tensor:
    batch_size = 2
    time_steps = 1

    if architecture_name == "SpikingMLP":
        input_size = MLP_INPUT_SIZES[dataset]
        assert architecture_params["input_size"] == input_size
        return torch.rand(time_steps, batch_size, 1, 1, input_size)

    if architecture_name == "Spiking1DCNN":
        shape = DATASET_SHAPES[dataset]
        return torch.rand(
            time_steps,
            batch_size,
            shape["channels"],
            shape["length"],
        )

    shape = DATASET_SHAPES[dataset]
    return torch.rand(
        time_steps,
        batch_size,
        shape["channels"],
        shape["height"],
        shape["width"],
    )


def _expected_classes(architecture_params: dict) -> int:
    if "out_classes" in architecture_params:
        return architecture_params["out_classes"]
    return architecture_params["output_size"]


@pytest.mark.parametrize("dataset,architecture_name", _configured_pairs())
def test_all_configured_dataset_architecture_pairs_run_one_epoch(
    dataset, architecture_name
):
    config_path = CONFIG_DIR / f"{dataset}.json"
    config = json.loads(config_path.read_text())
    architecture_params = config["architectures"][architecture_name]

    assert config["default_architecture"] in config["architectures"]
    assert architecture_name in ARCHITECTURES

    device = torch.device("cpu")
    model = ARCHITECTURES[architecture_name](**architecture_params).to(device)

    events = _build_events(dataset, architecture_name, architecture_params)
    targets = torch.full(
        (2,), _expected_classes(architecture_params) - 1, dtype=torch.long
    )

    if architecture_name == "SpikingMLP":
        assert architecture_params["input_size"] == MLP_INPUT_SIZES[dataset]
        assert events.shape[-1] == architecture_params["input_size"]
    elif architecture_name == "Spiking1DCNN":
        assert architecture_params["in_channels"] == DATASET_SHAPES[dataset]["channels"]
        assert events.shape[2] == architecture_params["in_channels"]
    else:
        assert architecture_params["in_channels"] == DATASET_SHAPES[dataset]["channels"]
        assert events.shape[2] == architecture_params["in_channels"]

    train_loader = SingleBatchLoader(events, targets)
    test_loader = SingleBatchLoader(events, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        scaler=None,
        accumulation_steps=1,
    )

    (
        test_loss,
        test_acc,
        sparsity,
        energy_joules,
        power_watts,
        total_spikes,
        total_elements,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
        measure_consumption=True,
    )

    output = model(events.to(device, dtype=torch.float32))

    assert output.shape == (2, _expected_classes(architecture_params))
    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(test_loss))
    assert 0.0 <= train_acc <= 100.0
    assert 0.0 <= test_acc <= 100.0
    assert torch.isfinite(torch.tensor(sparsity))
    assert torch.isfinite(torch.tensor(energy_joules))
    assert torch.isfinite(torch.tensor(power_watts))
    assert total_spikes >= 0
    assert total_elements >= 0
