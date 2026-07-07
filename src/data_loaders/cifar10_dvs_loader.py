import os
import time
import zipfile
import subprocess
import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import random_split

CIFAR10DVS_RESOURCES = [
    ("airplane", "https://ndownloader.figshare.com/files/7712788"),
    ("automobile", "https://ndownloader.figshare.com/files/7712791"),
    ("bird", "https://ndownloader.figshare.com/files/7712794"),
    ("cat", "https://ndownloader.figshare.com/files/7712812"),
    ("deer", "https://ndownloader.figshare.com/files/7712815"),
    ("dog", "https://ndownloader.figshare.com/files/7712818"),
    ("frog", "https://ndownloader.figshare.com/files/7712842"),
    ("horse", "https://ndownloader.figshare.com/files/7712845"),
    ("ship", "https://ndownloader.figshare.com/files/7712848"),
    ("truck", "https://ndownloader.figshare.com/files/7712851")
]

def download_and_extract_cifar10dvs(data_dir="./data/CIFAR10DVS"):
    os.makedirs(data_dir, exist_ok=True)
    
    for class_name, url in CIFAR10DVS_RESOURCES:
        class_dir = os.path.join(data_dir, class_name)
        zip_path = os.path.join(data_dir, f"{class_name}.zip")
        
        if os.path.exists(class_dir) and len([f for f in os.listdir(class_dir) if f.endswith('.aedat')]) > 0:
            continue
            
        print(f"\n--- Traitement de la classe : {class_name} ---")
        
        while not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:
            cmd = [
                "wget",
                "--show-progress",
                "-q",
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64)",
                "-O", zip_path,
                url
            ]
            subprocess.run(cmd)
            
            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000000:
                break
            else:
                time.sleep(15)
                
        print(f"Extraction de {class_name}.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(zip_path)
        time.sleep(5)

def get_cifar10_loaders(batch_size=64, n_time_bins=10, num_workers=4, split_seed=67):
    download_and_extract_cifar10dvs("./data/CIFAR10DVS")
    
    tonic.datasets.CIFAR10DVS.download = lambda self: None
    tonic.datasets.CIFAR10DVS.md5 = None

    frame_transform = transforms.Compose(
        [
            transforms.Downsample(spatial_factor=0.25),
            transforms.ToFrame(sensor_size=(32, 32, 2), n_time_bins=n_time_bins),
        ]
    )

    dataset = tonic.datasets.CIFAR10DVS(save_to="./data", transform=frame_transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(split_seed)
    
    train_set, test_set = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader