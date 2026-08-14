import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets.nav_gesture import NavGesture

def custom_collate_fn_nav(batch):
    events, targets = torch.utils.data.default_collate(batch)
    events = events.transpose(0, 1)
    return events.float(), targets

def get_nav_gesture_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    train_set = NavGesture(root="./data/NavGesture", train=True, data_type='frame', frames_num=time_steps)
    test_set = NavGesture(root="./data/NavGesture", train=False, data_type='frame', frames_num=time_steps)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_nav, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn_nav, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader