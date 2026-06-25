import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import random_split


def get_cifar10_loaders(batch_size=64, n_time_bins=10, num_workers=4, split_seed=67):
    """
    Get CIFAR10-DVS data loaders.
    """
    tonic.datasets.CIFAR10DVS.url = "https://ndownloader.figshare.com/files/7712788"

    frame_transform = transforms.Compose(
        [
            transforms.Downsample(spatial_factor=0.25),
            transforms.ToFrame(sensor_size=(32, 32, 2), n_time_bins=n_time_bins),
        ]
    )

    dataset = tonic.datasets.CIFAR10DVS(save_to="./data", transform=frame_transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_set, test_set = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
