import csv
import os
import random
import shutil

import torch


class CSVLogger:
    """
    Simple logger to record metrics to a CSV file.
    """

    def __init__(self, filepath, headers):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log(self, row):
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class EarlyStopping:
    """
    Stops training if a monitored metric does not improve after a given patience.
    """

    def __init__(self, patience=7, delta=0.0, mode="min"):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode

    def __call__(self, value, model=None):
        if self.mode == "max":
            score = value
            threshold = (
                self.best_score + self.delta if self.best_score is not None else None
            )
            improved = self.best_score is None or score > threshold
        else:
            score = -value
            threshold = (
                self.best_score + self.delta if self.best_score is not None else None
            )
            improved = self.best_score is None or score > threshold

        if self.best_score is None:
            self.best_score = score
        elif not improved:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def capture_rng_state():
    """Capture RNG state so training can be resumed deterministically."""
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore RNG state saved by capture_rng_state()."""
    if not state:
        return

    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(state, is_best, save_dir, filname="checkpoint_latest.pth"):
    """
    Save the complete state of the training and make a copy if it's the best model.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filname)
    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(save_dir, "model_best.pth")
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
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"=> Reprise de l'entraînement à l'époque {start_epoch + 1} (Ancien record : {best_acc:.2f}%)"
        )

    return start_epoch, best_acc
