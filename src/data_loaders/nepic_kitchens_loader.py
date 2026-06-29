import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class NEPICKitchens(Dataset):
    """
    Dataset loader for N-EPIC Kitchens pre-processed voxel grids.
    """
    def __init__(self, root_dir: str, split: str = "train", time_steps: int = 16):
        self.root_dir = root_dir
        self.split = split
        self.time_steps = time_steps
        self.samples = []

        domains = ["D1", "D2", "D3"]

        for domain in domains:
            pkl_path = os.path.join(root_dir, f"{domain}_{split}.pkl")
            if not os.path.exists(pkl_path):
                continue

            df = pd.read_pickle(pkl_path)

            for idx, row in df.iterrows():
                video_id = row.get("video_id", "")
                verb_class = row.get("verb_class", 0)

                uid = row.get("uid", idx)
                file_name = f"event_{int(uid):010d}.npy"
                file_path = os.path.join(root_dir, "voxels_xy_3", video_id, file_name)

                if os.path.exists(file_path):
                    self.samples.append((file_path, verb_class))

        if not self.samples:
            raise FileNotFoundError(
                f"Missing or invalid data for split '{split}' in {root_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        file_path, label = self.samples[index]
        data = np.load(file_path)
        tensor_data = torch.from_numpy(data).float()

        if tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(0)
        elif tensor_data.ndim == 4 and tensor_data.shape[0] == 2:
            tensor_data = tensor_data.permute(1, 0, 2, 3)

        t_dim = tensor_data.shape[0]
        if t_dim > self.time_steps:
            indices = torch.linspace(0, t_dim - 1, self.time_steps).long()
            tensor_data = tensor_data[indices]
        elif t_dim < self.time_steps:
            pad = torch.zeros((self.time_steps - t_dim, *tensor_data.shape[1:]))
            tensor_data = torch.cat((tensor_data, pad), dim=0)

        return tensor_data, label


def custom_collate_fn(batch):
    data, targets = torch.utils.data.default_collate(batch)
    data = data.transpose(0, 1)
    return data, targets


def get_nepic_kitchens_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    """
    Initializes and returns train and test DataLoaders for N-EPIC Kitchens.
    """
    data_dir = "./data/N-EPIC-Kitchens"

    train_set = NEPICKitchens(root_dir=data_dir, split="train", time_steps=time_steps)
    test_set = NEPICKitchens(root_dir=data_dir, split="test", time_steps=time_steps)

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