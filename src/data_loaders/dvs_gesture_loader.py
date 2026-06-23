import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import random_split, DataLoader

def custom_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets

def get_dvs_gesture_loaders(batch_size=64, n_time_bins=10, num_workers=4):
    """
    Get DVS Gesture data loaders.
    """
    tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
    tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"

    frame_transform = transforms.Compose([
        transforms.ToFrame(sensor_size=(128, 128, 2), n_time_bins=n_time_bins)
    ])

    train_set = tonic.datasets.DVSGesture(save_to='./data', train=True, transform=frame_transform)
    test_set = tonic.datasets.DVSGesture(save_to='./data', train=False, transform=frame_transform)

    cached_train = tonic.DiskCachedDataset(train_set, cache_path='./data/cache/dvs_gesture/train')
    cached_test = tonic.DiskCachedDataset(test_set, cache_path='./data/cache/dvs_gesture/test')

    train_loader = DataLoader(
        cached_train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        cached_test, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader