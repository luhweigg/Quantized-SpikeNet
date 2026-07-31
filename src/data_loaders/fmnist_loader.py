import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_fmnist_loaders(batch_size: int, time_steps: int, num_workers: int = 4):

    def custom_collate_fn(batch):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
        events_list = []

        for b in batch:
            img = b[0]
            img = F.interpolate(
                img.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
            ).squeeze(0)
            spikes = torch.rand(time_steps, 1, 128, 128) < img.unsqueeze(0)
            spikes = spikes.float()
            spikes = spikes.repeat(1, 2, 1, 1)

            events_list.append(spikes)

        events = torch.stack(events_list)
        events = events.transpose(0, 1)

        return events, targets

    transform = transforms.ToTensor()

    train_set = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
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
