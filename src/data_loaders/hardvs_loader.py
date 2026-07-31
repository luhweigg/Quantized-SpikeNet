import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.datasets.hardvs import HARDVS

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

def get_hardvs_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    root_dir = "./data/hardvs"
    
    os.makedirs(root_dir, exist_ok=True)
    
    train_set = HARDVS(
        root=root_dir, 
        train_test_val="train",
        data_type="frame", 
        frames_number=time_steps, 
        split_by="number"
    )
    
    test_set = HARDVS(
        root=root_dir, 
        train_test_val="test",
        data_type="frame", 
        frames_number=time_steps, 
        split_by="number"
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader