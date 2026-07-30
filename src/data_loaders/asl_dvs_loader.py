import os
import torch
from torch.utils.data import DataLoader, random_split
from spikingjelly.datasets.asl_dvs import ASLDVS


def custom_collate_fn(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_asl_dvs_loaders(
    batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42
):
    os.makedirs("./data/ASL-DVS", exist_ok=True)

    dataset = ASLDVS(
        root="./data/ASL-DVS",
        data_type="frame",
        frames_number=time_steps,
        split_by="number",
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_set, test_set = random_split(
        dataset, [train_size, test_size], generator=generator
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
