import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_dsec_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    sensor_size = (480, 640, 2)

    transform = transforms.Compose(
        [
            transforms.Downsample(spatial_factor=0.5),
            transforms.ToFrame(sensor_size=(240, 320, 2), n_time_bins=time_steps),
        ]
    )

    train_set = tonic.datasets.DSEC(
        save_to="./data", split="train", data_selection="events_left"
    )
    test_set = tonic.datasets.DSEC(
        save_to="./data", split="test", data_selection="events_left"
    )

    cached_train = tonic.DiskCachedDataset(
        train_set, cache_path="./data/cache/dsec/train", transform=transform
    )
    cached_test = tonic.DiskCachedDataset(
        test_set, cache_path="./data/cache/dsec/test", transform=transform
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
