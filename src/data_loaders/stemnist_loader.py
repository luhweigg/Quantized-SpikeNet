import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class STEMNIST(Dataset):
    def __init__(self, root_dir, time_steps=16):
        self.time_steps = time_steps
        self.samples = []
        self.classes = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".h5") and "spikes" in file:
                    cls_name = os.path.basename(root)
                    if cls_name not in self.classes:
                        self.classes.append(cls_name)

        self.classes = sorted(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".h5") and "spikes" in file:
                    cls_name = os.path.basename(root)
                    self.samples.append(
                        (os.path.join(root, file), self.class_to_idx[cls_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        frames = torch.zeros((self.time_steps, 2, 16, 16), dtype=torch.float32)

        with h5py.File(file_path, "r") as f:
            if "timestamp" not in f.keys():
                return frames, label

            t = f["timestamp"][:]
            taxel = f["taxel ID"][:]
            p = f["polarity"][:]

        if len(t) == 0:
            return frames, label

        x = taxel % 16
        y = taxel // 16

        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = ((t - t_min) / (t_max - t_min) * (self.time_steps - 1)).astype(int)
        else:
            t_norm = np.zeros_like(t, dtype=int)

        p_norm = (p > 0).astype(int)

        for i in range(len(x)):
            frames[t_norm[i], p_norm[i], y[i], x[i]] += 1.0

        frames = torch.clamp(frames, 0, 5)
        return frames, label


def custom_collate_fn_stemnist(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_stemnist_loaders(
    batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42
):
    dataset = STEMNIST(root_dir="./data/STEMNIST", time_steps=time_steps)

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
        collate_fn=custom_collate_fn_stemnist,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_stemnist,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
