import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import (
    SpikingMLP,
    CompactSpikingCNN,
    spiking_vgg5,
    spiking_vgg11,
    spiking_vgg13,
    spiking_vgg16,
    spiking_vgg19,
    spiking_resnet18,
    spiking_resnet34,
    spiking_resnet50
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
    "SpikingVGG5": spiking_vgg5,
    "SpikingVGG11": spiking_vgg11,
    "SpikingVGG13": spiking_vgg13,
    "SpikingVGG16": spiking_vgg16,
    "SpikingVGG19": spiking_vgg19,
    "SpikingResNet18": spiking_resnet18,
    "SpikingResNet34": spiking_resnet34,
    "SpikingResNet50": spiking_resnet50
}

def build_components(dataset, model_name, neuron_type, model_config, batch_size, time_steps, lr, epochs, device):
    if dataset not in DATA_LOADERS:
        raise ValueError(f"Dataset {dataset} non supporté.")

    train_loader, test_loader = DATA_LOADERS[dataset](batch_size, time_steps)

    if model_name not in ARCHITECTURES:
        raise ValueError(f"Architecture {model_name} non supportée.")

    arch_class = ARCHITECTURES[model_name]
    
    params = model_config.get("params", {})
    params["neuron_type"] = neuron_type
    
    model = arch_class(**params).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = None

    return model, train_loader, test_loader, optimizer, scheduler, criterion, scaler