import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader, random_split


def custom_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_cifar10_loaders(
    batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42
):
    tonic.datasets.CIFAR10DVS.url = "https://ndownloader.figshare.com/files/38023437"

    full_dataset = tonic.datasets.CIFAR10DVS(save_to="./data")

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(split_seed)
    train_subset, test_subset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    train_transform = transforms.Compose(
        [
            transforms.Downsample(spatial_factor=0.25),
            transforms.RandomFlipPolarity(),
            transforms.RandomCrop(sensor_size=(32, 32), target_size=(28, 28)),
            transforms.DropEvent(p=0.1),
            transforms.ToFrame(sensor_size=(28, 28, 2), n_time_bins=time_steps),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Downsample(spatial_factor=0.25),
            transforms.CenterCrop(sensor_size=(32, 32), size=(28, 28)),
            transforms.ToFrame(sensor_size=(28, 28, 2), n_time_bins=time_steps),
        ]
    )

    cached_train = tonic.DiskCachedDataset(
        train_subset,
        cache_path="./data/cache/cifar10_dvs/train_aug",
        transform=train_transform,
    )

    cached_test = tonic.DiskCachedDataset(
        test_subset,
        cache_path="./data/cache/cifar10_dvs/test_aug",
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
