import torch
import tonic
import tonic.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    events_list = []

    for b in batch:
        ev = torch.as_tensor(b[0]).float()
        ev = F.interpolate(ev, size=(128, 128), mode="bilinear", align_corners=False)
        events_list.append(ev)

    events = torch.stack(events_list)
    events = events.transpose(0, 1)
    return events, targets


def get_ncars_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    sensor_size = tonic.datasets.NCARS.sensor_size
    transform = transforms.Compose(
        [
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps),
        ]
    )

    train_set = tonic.datasets.NCARS(save_to="./data", train=True, transform=transform)
    test_set = tonic.datasets.NCARS(save_to="./data", train=False, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
