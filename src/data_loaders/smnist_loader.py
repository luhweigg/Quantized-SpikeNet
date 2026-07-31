import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

def smnist_collate_fn(batch):
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    events_list = [torch.as_tensor(b[0]).float() for b in batch]
    
    events = torch.stack(events_list)
    events = events.transpose(0, 1)
    return events, targets

def get_smnist_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    sensor_size = (99, 1, 2)
    
    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps),
    ])

    train_set = tonic.datasets.SMNIST(save_to="./data", train=True, transform=transform)
    test_set = tonic.datasets.SMNIST(save_to="./data", train=False, transform=transform)

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=smnist_collate_fn, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=smnist_collate_fn, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, test_loader