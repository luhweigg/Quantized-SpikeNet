import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets

def get_nepic_kitchens_loaders(batch_size=32, n_time_bins=15, num_workers=4):
    """
    Data loader for N-EPIC Kitchens with 8 action classes.
    Assumes tonic.datasets.NEPICKitchens exists or has been custom implemented.
    """
    sensor_size = (346, 260, 2)
    
    train_set = tonic.datasets.NEPICKitchens(save_to="./data", train=True)
    test_set = tonic.datasets.NEPICKitchens(save_to="./data", train=False)

    train_transform = transforms.Compose(
        [
            transforms.SpatialJitter(sensor_size=sensor_size, clip_outliers=True),
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)]
    )

    cached_train = tonic.DiskCachedDataset(
        train_set,
        cache_path="./data/cache/nepic_kitchens/raw_train",
        transform=train_transform,
    )
    
    cached_test = tonic.DiskCachedDataset(
        test_set,
        cache_path="./data/cache/nepic_kitchens/raw_test",
        transform=test_transform,
    )

    train_loader = DataLoader(
        cached_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        cached_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader