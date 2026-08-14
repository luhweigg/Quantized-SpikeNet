import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split


class CeoliniGestures(Dataset):
    def __init__(self, root_dir, time_steps=16):
        self.time_steps = time_steps
        pkl_path = os.path.join(root_dir, "relax21_cropped_dvs_emg_spikes.pkl")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Fichier introuvable: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.y = data["y"][0]
        self.dvs = data["dvs"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        events = self.dvs[idx]
        label = int(self.y[idx])

        frames = torch.zeros((self.time_steps, 2, 64, 64), dtype=torch.float32)

        if events.shape[1] == 0:
            return frames, label

        x, y, t, p = events[0, :], events[1, :], events[2, :], events[3, :]

        t_min, t_max = t.min(), t.max()
        if t_max > t_min:
            t_norm = ((t - t_min) / (t_max - t_min) * (self.time_steps - 1)).astype(int)
        else:
            t_norm = np.zeros_like(t, dtype=int)

        x_norm = np.clip((x / 128.0 * 64).astype(int), 0, 63)
        y_norm = np.clip((y / 128.0 * 64).astype(int), 0, 63)
        p_norm = (p > 0).astype(int)

        for i in range(len(x)):
            frames[t_norm[i], p_norm[i], y_norm[i], x_norm[i]] += 1.0

        frames = torch.clamp(frames, 0, 5)
        return frames, label


def custom_collate_fn_ceolini(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events, targets


def get_ceolini_gestures_loaders(
    batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42
):
    dataset = CeoliniGestures(root_dir="./data/CeoliniGestures", time_steps=time_steps)

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
        collate_fn=custom_collate_fn_ceolini,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_ceolini,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
