import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import (
    CompactSpikingCNN,
    SpikingMLP,
    SpikingVGG3,
    SpikingVGG5,
    SpikingVGG11,
    SpikingResNet18,
)
from src.data_loaders import (
    get_nmnist_loaders,
    get_cifar10_loaders,
    get_dvs_gesture_loaders,
    get_nepic_kitchens_loaders,
)

DATA_LOADERS = {
    "nmnist": get_nmnist_loaders,
    "cifar10": get_cifar10_loaders,
    "dvs_gesture": get_dvs_gesture_loaders,
    "nepic_kitchens": get_nepic_kitchens_loaders,
}

ARCHITECTURES = {
    "SpikingMLP": SpikingMLP,
    "CompactSpikingCNN": CompactSpikingCNN,
    "SpikingVGG3": SpikingVGG3,
    "SpikingVGG5": SpikingVGG5,
    "SpikingVGG11": SpikingVGG11,
    "SpikingResNet18": SpikingResNet18,
}


def build_components(
    dataset, arch_name, arch_params, batch_size, time_steps, lr, epochs, device
):
    if dataset not in DATA_LOADERS:
        raise ValueError(f"Dataset {dataset} non supporté.")

    train_loader, test_loader = DATA_LOADERS[dataset](batch_size, time_steps)

    arch_class = ARCHITECTURES[arch_name]
    model = arch_class(**arch_params).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * 2)
    criterion = nn.CrossEntropyLoss()
    scaler = None

    return model, train_loader, test_loader, optimizer, scheduler, criterion, scaler
