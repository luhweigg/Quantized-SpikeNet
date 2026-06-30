import argparse
from pathlib import Path

import pytest
import torch

import main as training_main


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        batch_size = x.shape[1] if x.ndim > 1 else 1
        return torch.zeros(batch_size, 10)

    def reset_states(self):
        return None


class DummyOptimizer:
    def __init__(self):
        self._state = {"lr": 1e-3, "steps": 0}

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)


class DummyScheduler:
    def __init__(self):
        self._state = {"steps": 0}

    def step(self):
        self._state["steps"] += 1

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)


def _fake_build_components(*_args, **_kwargs):
    model = DummyModel()
    train_loader = [
        (
            torch.zeros(4, 2, 2, 8, 8),
            torch.zeros(2, dtype=torch.long),
        )
    ]
    test_loader = train_loader
    optimizer = DummyOptimizer()
    scheduler = DummyScheduler()
    criterion = torch.nn.CrossEntropyLoss()
    scaler = None
    return model, train_loader, test_loader, optimizer, scheduler, criterion, scaler


def _latest_run_dir(base_dir: Path, dataset: str) -> Path:
    candidates = sorted((base_dir / dataset).glob("run_*"))
    assert candidates, "No run directory found"
    return candidates[-1]


def test_main_resume_from_checkpoint_path(monkeypatch, tmp_path):
    calls = {"train": 0}

    def fake_train_one_epoch(*_args, **_kwargs):
        calls["train"] += 1
        return 0.5, 50.0

    def fake_evaluate(*_args, **_kwargs):
        return 0.4, 51.0, 90.0, 0.001, 0.1

    def fake_quantize(*_args, **_kwargs):
        return {"weight": torch.tensor([1], dtype=torch.int8)}, {
            "weight": {
                "scale": 1.0,
                "zero_point": 0,
                "num_bits": 8,
                "qmin": -128,
                "qmax": 127,
            }
        }

    monkeypatch.setattr(training_main, "build_components", _fake_build_components)
    monkeypatch.setattr(training_main, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(training_main, "evaluate", fake_evaluate)
    monkeypatch.setattr(training_main, "quantize_weights", fake_quantize)

    save_root = tmp_path / "saved_models"

    args_first = argparse.Namespace(
        dataset="nmnist",
        epochs=2,
        batch_size=2,
        lr=1e-3,
        Time=4,
        save_dir=str(save_root),
        resume=None,
        use_wandb=False,
        wandb_project="quantized_spikenet",
    )
    monkeypatch.setattr(training_main, "parse_args", lambda: args_first)
    training_main.main()

    assert calls["train"] == 2

    run_dir = _latest_run_dir(save_root, "nmnist")
    checkpoint_path = run_dir / "checkpoint_latest.pth"
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "rng_state" in checkpoint

    args_resume = argparse.Namespace(
        dataset="nmnist",
        epochs=3,
        batch_size=2,
        lr=1e-3,
        Time=4,
        save_dir=str(save_root),
        resume=str(checkpoint_path),
        use_wandb=False,
        wandb_project="quantized_spikenet",
    )
    monkeypatch.setattr(training_main, "parse_args", lambda: args_resume)
    training_main.main()

    assert calls["train"] == 3
    assert (run_dir / "nmnist_quantized.pth").exists()


def test_main_invalid_resume_path_raises(monkeypatch, tmp_path):
    bad_resume = tmp_path / "missing_resume_dir"

    args = argparse.Namespace(
        dataset="nmnist",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        Time=4,
        save_dir=str(tmp_path / "saved_models"),
        resume=str(bad_resume),
        use_wandb=False,
        wandb_project="quantized_spikenet",
    )
    monkeypatch.setattr(training_main, "parse_args", lambda: args)

    with pytest.raises(FileNotFoundError):
        training_main.main()


def test_main_quantized_export_contract(monkeypatch, tmp_path):
    def fake_train_one_epoch(*_args, **_kwargs):
        return 0.6, 45.0

    def fake_evaluate(*_args, **_kwargs):
        return 0.5, 46.0, 88.0, 0.001, 0.1

    def fake_quantize(*_args, **_kwargs):
        return {
            "weight": torch.tensor([1, -2], dtype=torch.int8),
            "bias": torch.tensor([0], dtype=torch.int8),
        }, {
            "weight": {
                "scale": 0.01,
                "zero_point": 0,
                "num_bits": 8,
                "qmin": -128,
                "qmax": 127,
            },
            "bias": {
                "scale": 0.02,
                "zero_point": 0,
                "num_bits": 8,
                "qmin": -128,
                "qmax": 127,
            },
        }

    monkeypatch.setattr(training_main, "build_components", _fake_build_components)
    monkeypatch.setattr(training_main, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(training_main, "evaluate", fake_evaluate)
    monkeypatch.setattr(training_main, "quantize_weights", fake_quantize)

    save_root = tmp_path / "saved_models"
    args = argparse.Namespace(
        dataset="nmnist",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        Time=4,
        save_dir=str(save_root),
        resume=None,
        use_wandb=False,
        wandb_project="quantized_spikenet",
    )
    monkeypatch.setattr(training_main, "parse_args", lambda: args)

    training_main.main()

    run_dir = _latest_run_dir(save_root, "nmnist")
    quantized_path = run_dir / "nmnist_quantized.pth"
    assert quantized_path.exists()

    payload = torch.load(quantized_path, map_location="cpu", weights_only=False)
    assert set(payload.keys()) == {"weights", "metadata"}

    weight_names = set(payload["weights"].keys())
    metadata_names = set(payload["metadata"].keys())
    assert weight_names == metadata_names

    for name, tensor in payload["weights"].items():
        assert tensor.dtype == torch.int8
        info = payload["metadata"][name]
        assert info["scale"] > 0
        assert info["zero_point"] == 0
        assert info["num_bits"] == 8
        assert info["qmin"] == -128
        assert info["qmax"] == 127
