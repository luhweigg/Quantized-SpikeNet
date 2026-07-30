import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader, random_split


def custom_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_ucf101_dvs_loaders(
    batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42
):
    sensor_size = tonic.datasets.UCF101DVS.sensor_size

    transform = transforms.Compose(
        [
            transforms.Downsample(spatial_factor=0.5),
            transforms.ToFrame(sensor_size=(120, 170, 2), n_time_bins=time_steps),
        ]
    )

    full_dataset = tonic.datasets.UCF101DVS(save_to="./data")

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_subset, test_subset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    cached_train = tonic.DiskCachedDataset(
        train_subset, cache_path="./data/cache/ucf101/train", transform=transform
    )
    cached_test = tonic.DiskCachedDataset(
        test_subset, cache_path="./data/cache/ucf101/test", transform=transform
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
