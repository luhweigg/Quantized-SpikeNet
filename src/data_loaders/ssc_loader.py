import tonic
import torch
from tonic import transforms
from torch.utils.data import DataLoader


def audio_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    events = events.view(events.shape[0], events.shape[1], 1, -1)
    return events, targets


def get_ssc_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    sensor_size = tonic.datasets.SSC.sensor_size
    transform = transforms.Compose(
        [
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps),
        ]
    )

    train_set = tonic.datasets.SSC(save_to="./data", split="train")
    test_set = tonic.datasets.SSC(save_to="./data", split="test")

    cached_train = tonic.DiskCachedDataset(
        train_set, cache_path="./data/cache/ssc/train", transform=transform
    )
    cached_test = tonic.DiskCachedDataset(
        test_set, cache_path="./data/cache/ssc/test", transform=transform
    )

    train_loader = DataLoader(
        cached_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=audio_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        cached_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=audio_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
