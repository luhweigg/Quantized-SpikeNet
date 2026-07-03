import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from src.models import SpikingMLP, spiking_vgg5


def test_gradient_flow_NMNIST():
    """
    Verify that the gradients are properly computed during backpropagation,
    that they reach all layers without vanishing
    and that they do not contain NaN or infinite values (exploding).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(67)
    model = SpikingMLP(input_size=2312, hidden_size=256, num_classes=10).to(device)

    batch_size, T, C, H, W = 4, 16, 2, 34, 34
    events = (torch.rand(T, batch_size, C, H, W) > 0.5).float().to(device)
    targets = torch.randint(0, 10, (batch_size,)).to(device)

    criterion = nn.CrossEntropyLoss()

    output = model(events)
    loss = criterion(output, targets)
    loss.backward()

    has_active_gradients = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient manquant sur : {name}"

            assert not torch.isnan(param.grad).any(), (
                f"Gradient NaN detecte sur : {name}"
            )
            assert not torch.isinf(param.grad).any(), (
                f"Gradient infini detecte sur : {name}"
            )

            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-7:
                has_active_gradients = True

    assert has_active_gradients, (
        "Vanishing gradient : aucun gradient significatif n'a ete propage."
    )

    functional.reset_net(model)


def test_gradient_flow_DVSGesture():
    """
    Verify that the gradients are properly computed during backpropagation for DVSGesture_SNN,
    that they reach all layers without vanishing
    and that they do not contain NaN or infinite values (exploding).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(67)
    model = spiking_vgg5(in_channels=2, num_classes=11).to(device)

    batch_size, T, C, H, W = 2, 4, 2, 128, 128
    events = (torch.rand(T, batch_size, C, H, W) > 0.9).float().to(device)
    targets = torch.randint(0, 11, (batch_size,)).to(device)

    criterion = nn.CrossEntropyLoss()

    output = model(events)
    loss = criterion(output, targets)
    loss.backward()

    has_active_gradients = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient manquant sur : {name}"
            assert not torch.isnan(param.grad).any(), (
                f"Gradient NaN détecté sur : {name}"
            )
            assert not torch.isinf(param.grad).any(), (
                f"Gradient infini détecté sur : {name}"
            )

            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-7:
                has_active_gradients = True

    assert has_active_gradients, (
        "Vanishing gradient : aucun gradient significatif n'a été propagé."
    )

    functional.reset_net(model)
