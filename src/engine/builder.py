import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models.architectures import SpikingMLP, CompactSpikingCNN, SpikingVGG5
from src.data_loaders import get_nmnist_loaders, get_cifar10_loaders, get_dvs_gesture_loaders

DATA_LOADERS = {
    'nmnist': get_nmnist_loaders,
    'cifar10': get_cifar10_loaders,
    'dvs_gesture': get_dvs_gesture_loaders
}

ARCHITECTURES = {
    'SpikingMLP': SpikingMLP,
    'CompactSpikingCNN': CompactSpikingCNN,
    'SpikingVGG5': SpikingVGG5,
}

def build_components(dataset, model_config, batch_size, time_steps, lr, epochs, device):
    if dataset not in DATA_LOADERS:
        raise ValueError(f"Dataset {dataset} non supporté.")

    train_loader, test_loader = DATA_LOADERS[dataset](batch_size, time_steps)
    
    arch_class = ARCHITECTURES[model_config['architecture']]
    model = arch_class(**model_config['params']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    return model, train_loader, test_loader, optimizer, scheduler, criterion, scaler