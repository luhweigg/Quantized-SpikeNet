import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets.hardvs import HARDVS


def custom_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_hardvs_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    train_set = HARDVS(
        root="./data/HARDVS",
        train=True,
        data_type="frame",
        frames_number=time_steps,
        split_by="number",
    )
    test_set = HARDVS(
        root="./data/HARDVS",
        train=False,
        data_type="frame",
        frames_number=time_steps,
        split_by="number",
    )

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
