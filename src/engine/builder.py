import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import (
    CompactSpikingCNN,
    SpikingMLP,
    Spiking1DCNN,
    SpikingVGG3,
    SpikingVGG4,
    SpikingVGG5,
    SpikingVGG8,
    SpikingResNet18,
    SpikingResNet34,
)
from src.data_loaders import (
    get_cifar10_loaders,
    get_dvs_gesture_loaders,
    get_dvs_lip_loaders,
    get_fmnist_loaders,
    get_hardvs_loaders,
    get_ncaltech101_loaders,
    get_nepic_kitchens_loaders,
    get_nmnist_loaders,
    get_shd_loaders,
    get_smnist_loaders,
    get_ssc_loaders,
)

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

DATA_LOADERS = {
    "cifar10": get_cifar10_loaders,
    "dvs_gesture": get_dvs_gesture_loaders,
    "dvs_lip": get_dvs_lip_loaders,
    "fmnist": get_fmnist_loaders,
    "hardvs": get_hardvs_loaders,
    "ncaltech101": get_ncaltech101_loaders,
    "nepic_kitchens": get_nepic_kitchens_loaders,
    "nmnist": get_nmnist_loaders,
    "shd": get_shd_loaders,
    "smnist": get_smnist_loaders,
    "ssc": get_ssc_loaders,
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
