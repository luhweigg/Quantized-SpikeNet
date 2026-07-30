import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_poker_dvs_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    sensor_size = tonic.datasets.POKERDVS.sensor_size

    train_transform = transforms.Compose(
        [
            transforms.RandomFlipPolarity(),
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps),
        ]
    )

    train_set = tonic.datasets.POKERDVS(save_to="./data", train=True)
    test_set = tonic.datasets.POKERDVS(save_to="./data", train=False)

    cached_train = tonic.DiskCachedDataset(
        train_set, cache_path="./data/cache/poker_dvs/train", transform=train_transform
    )
    cached_test = tonic.DiskCachedDataset(
        test_set, cache_path="./data/cache/poker_dvs/test", transform=test_transform
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
