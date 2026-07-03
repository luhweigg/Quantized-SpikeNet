import pytest
import torch
from spikingjelly.activation_based import functional
from src.models import SpikingMLP, spiking_vgg5


@pytest.mark.parametrize(
    "model_class, model_params, input_shape",
    [
        (
            SpikingMLP,
            {"input_size": 2312, "hidden_size": 256, "output_size": 10},
            (16, 2, 2, 34, 34),
        ),
        (spiking_vgg5, {"in_channels": 2, "num_classes": 11}, (4, 2, 2, 128, 128)),
    ],
)
def test_eval_mode_determinism_and_regularization(
    model_class, model_params, input_shape
):
    """
    Verify that:
    1. Dropout/BatchNorm is active in train() mode.
    2. Models are 100% deterministic in eval() mode (no randomness).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**model_params).to(device)
    x = torch.rand(*input_shape).to(device)
    model.train()
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]

    if dropout_layers:
        assert all(layer.training for layer in dropout_layers), (
            "Les couches de Dropout doivent être actives en train()"
        )

    model.eval()
    out_eval_1 = model(x)
    functional.reset_net(model)
    out_eval_2 = model(x)

    assert torch.equal(out_eval_1, out_eval_2), (
        f"Comportement stochastique anormal détecté en mode eval sur {model_class.__name__}."
    )

    if dropout_layers:
        assert all(not layer.training for layer in dropout_layers), (
            "Les couches de Dropout doivent être désactivées en eval()"
        )
