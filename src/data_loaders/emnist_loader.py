import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_emnist_loaders(batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42):
    
    def poisson_rate_coding_collate_fn(batch):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
        images = torch.stack([b[0] for b in batch])
        images = F.interpolate(images, size=(128, 128), mode="bilinear", align_corners=False)
        images = images.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
        spikes = (torch.rand_like(images) < images).float()
        
        return spikes, targets

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.EMNIST(
        root="./data", 
        split="balanced", 
        train=True, 
        download=True, 
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    
    train_subset, test_subset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        collate_fn=poisson_rate_coding_collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, 
        collate_fn=poisson_rate_coding_collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader