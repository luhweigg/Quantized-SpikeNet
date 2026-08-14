import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from spikingjelly.datasets.asl_dvs import ASLDVS

def custom_collate_fn_asl(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    T, B, C, H, W = events.shape
    events = events.reshape(T * B, C, H, W)
    events = F.interpolate(events.float(), size=(64, 64), mode='bilinear', align_corners=False)
    events = events.view(T, B, C, 64, 64)
    
    return events, targets

def get_asl_dvs_loaders(batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42):
    dataset = ASLDVS(root="./data/ASL-DVS", data_type='frame', frames_num=time_steps)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_subset, test_subset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_asl, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn_asl, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader