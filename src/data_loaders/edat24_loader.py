import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class EDAT24(Dataset):
    def __init__(self, root_dir, time_steps=16):
        self.root_dir = root_dir
        self.time_steps = time_steps
        self.samples = []
        self.classes = {"idle": 0, "pick": 1, "place": 2, "screw": 3}

        for cls_name, cls_label in self.classes.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                if file.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, file), cls_label))

        if not self.samples:
            raise FileNotFoundError(f"Aucun fichier .npy trouvé dans {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        events = np.load(file_path)
        frames = torch.zeros((self.time_steps, 2, 64, 64), dtype=torch.float32)

        if len(events) == 0:
            return frames, label

        x = events[:, 0]
        y = events[:, 1]
        t = events[:, 2]
        p = events[:, 3] if events.shape[1] > 3 else np.ones_like(t)

        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = ((t - t_min) / (t_max - t_min) * (self.time_steps - 1)).astype(int)
        else:
            t_norm = np.zeros_like(t, dtype=int)

        x_norm = np.clip((x / 240.0 * 64).astype(int), 0, 63)
        y_norm = np.clip((y / 180.0 * 64).astype(int), 0, 63)
        p_norm = (p > 0).astype(int)

        for i in range(len(events)):
            frames[t_norm[i], p_norm[i], y_norm[i], x_norm[i]] += 1.0

        frames = torch.clamp(frames, 0, 5)

        return frames, label


def custom_collate_fn_edat(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_edat24_loaders(
    batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42
):
    dataset = EDAT24(root_dir="./data/EDAT24", time_steps=time_steps)

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
        collate_fn=custom_collate_fn_edat,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_edat,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
