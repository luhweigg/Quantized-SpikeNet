import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from spikingjelly.datasets.nav_gesture import NAVGestureWalk

def custom_collate_fn_nav(batch):
    events_list, targets = [], []
    for x, y in batch:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = F.interpolate(x.float(), size=(64, 64), mode='bilinear', align_corners=False)
        events_list.append(x)
        targets.append(y)
    
    events = torch.stack(events_list)
    events = events.transpose(0, 1) 
    targets = torch.tensor(targets, dtype=torch.long)
    return events, targets

def get_nav_gesture_loaders(batch_size: int, time_steps: int, num_workers: int = 4, split_seed: int = 42):
    dataset = NAVGestureWalk(root="./data/NavGesture", data_type='frame', frames_number=time_steps, split_by='number')
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    train_subset, test_subset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_nav, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn_nav, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader