import torch
import tonic
import tonic.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tonic.datasets.pokerdvs131 import POKERDVS131

def custom_collate_fn(batch):
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    events_list = []
    for b in batch:
        ev = torch.as_tensor(b[0]).float()
        ev = F.interpolate(ev, size=(128, 128), mode='bilinear', align_corners=False)
        events_list.append(ev)
        
    events = torch.stack(events_list)
    events = events.transpose(0, 1)
    return events, targets

def get_pokerdvs131_loaders(batch_size: int, time_steps: int, num_workers: int = 4):
    sensor_size = POKERDVS131.sensor_size
    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps),
    ])

    train_set = POKERDVS131(save_to="./data", train=True, transform=transform)
    test_set = POKERDVS131(save_to="./data", train=False, transform=transform)

    if len(train_set.targets) > 0 and isinstance(train_set.targets[0], str):
        classes = sorted(list(set(train_set.targets)))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        train_set.target_transform = lambda target: class_to_idx[str(target)]
        test_set.target_transform = lambda target: class_to_idx[str(target)]

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, 
        collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader