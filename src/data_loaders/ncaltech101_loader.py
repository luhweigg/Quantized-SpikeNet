import torch
import tonic
import tonic.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


def custom_collate_fn(batch):
    targets = torch.tensor([b[1] for b in batch])
    events_list = []
    
    for b in batch:
        ev = torch.as_tensor(b[0]).float()
        ev = F.interpolate(ev, size=(128, 128), mode='bilinear', align_corners=False)
        events_list.append(ev)
    events = torch.stack(events_list)
    events = events.transpose(0, 1)
    return events, targets


def get_ncaltech101_loaders(
    batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42
):
    sensor_size = tonic.datasets.NCALTECH101.sensor_size

    transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps),
        ]
    )

    dataset = tonic.datasets.NCALTECH101(save_to="./data", transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_subset, test_subset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader