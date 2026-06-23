import os
import shutil
import torch
import csv

class CSVLogger:
    """
    Simple logger to record metrics to a CSV file.
    """
    def __init__(self, filepath, headers):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log(self, row):
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
class EarlyStopping:
    """
    Stops training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_checkpoint(state, is_best, save_dir, filname='checkpoint_latest.pth'):
    """
    Save the complete state of the training and make a copy if it's the best model.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filname)
    torch.save(state, save_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth')
        shutil.copyfile(save_path, best_path)

def load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device):
    """
    Load the training state from a checkpoint if it exists, including model weights, optimizer state, scheduler state, and RNG states.
    """
    start_epoch = 0
    best_acc = 0.0
    
    if os.path.exists(resume_path):
        print(f"=> Chargement du checkpoint trouvé : '{resume_path}'")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"=> Reprise de l'entraînement à l'époque {start_epoch+1} (Ancien record : {best_acc:.2f}%)")
    
    return start_epoch, best_acc