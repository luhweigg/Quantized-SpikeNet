import csv
import os
import random
import shutil

import torch
import torch.nn as nn
import torch.distributed as dist


class CSVLogger:
    """
    Simple logger to record metrics to a CSV file.
    """

    def __init__(self, filepath, headers):
        self.filepath = filepath
        # Initialize the file with headers if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log(self, row):
        """
        Append a row of metrics to the CSV file.
        """
        # Open in append mode ("a") to add new metric rows without overwriting existing data
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
        """
        Evaluates the monitored value and triggers early stopping if necessary.
        """
        if self.mode == "max":
            score = value
            threshold = (
                self.best_score + self.delta if self.best_score is not None else None
            )
            improved = self.best_score is None or score > threshold
        else:
            # Invert the score for 'min' mode to reuse the greater-than comparison logic
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


def off_diagonal(x):
    """
    Extracts the off-diagonal elements of a square matrix x.
    """
    n, m = x.shape
    # Advanced tensor manipulation: 
    # 1. Flatten the N x N matrix into a 1D tensor of size N*N.
    # 2. Drop the very last element (length becomes N*N - 1).
    # 3. Reshape into a rectangular matrix of shape (N - 1, N + 1). 
    #    Due to the shift, the original diagonal elements align perfectly in the first column [:, 0].
    # 4. Slice out the first column [:, 1:] to keep only the off-diagonal elements, and flatten again.
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(torch.nn.Module):
    """
    Computes the Barlow Twins loss, optionally adapted for Spiking Neural Networks (SNNs).
    
    This loss aligns representations across views while minimizing redundancy.
    """
    def __init__(self, device, world_size, cross_temporal=False, simplified_loss=False, lambda_param=5e-3, proj_dim=1024):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device
        self.world_size = world_size
        self.cross_temporal = cross_temporal
        self.simplified_loss = simplified_loss
        # Disable affine parameters (gamma, beta) because Barlow Twins explicitly requires empirical normalization
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        """
        Computes the Barlow Twins loss between two sets of representations z_a and z_b.
        """
        if not self.cross_temporal:
            # Standard Barlow Twins on static embeddings (time steps merged into batch dimension)
            z_a = z_a.flatten(0, 1)
            z_b = z_b.flatten(0, 1)
            N, _ = z_a.shape

            # Cross-correlation matrix between normalized features
            c = self.bn(z_a).T @ self.bn(z_b)
            c.div_(N * self.world_size)

            # Aggregate cross-correlation matrix across all GPUs in distributed training
            if self.world_size > 1:
                torch.distributed.all_reduce(c)
            # Invariance term: diagonal elements should be close to 1
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            # Redundancy reduction term: off-diagonal elements should be close to 0
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.lambda_param * off_diag
        else:
            # Spiking-aware temporal objectives
            T, B, D = z_a.shape
            # Boudary Temporal Loss: Focuses on initial (t=1) and final (t=T) time steps
            # Reduces the quadratic complexity O(T^2) of the Cross Temporal
            if self.simplified_loss:
                z = torch.zeros((4, B, D), device=z_a.device)
                z[0, ...] = z_a[0, ...]     # View A, first time step
                z[1, ...] = z_a[T - 1, ...] # View A, last time step
                z[2, ...] = z_b[0, ...]     # View B, first time step
                z[3, ...] = z_b[T - 1, ...] # View B, last time step
                iteration = 2
            else:
                # Cross Temporal Loss: Aligns representations across all pairs of time steps
                z = torch.cat((z_a, z_b), dim=0)
                iteration = T
            
            loss = 0
            # Compute pairwise cross-correlations across chosen time steps and views
            for i in range(2 * iteration):
                for j in range(2 * iteration):
                    if i == j:
                        continue    # Skip self-correlations
                    z_a_t = z[i, ...]
                    z_b_t = z[j, ...]
                    N = B

                    # Compute the cross-correlation for the specific time step pair
                    c = self.bn(z_a_t).T @ self.bn(z_b_t)
                    c.div_(N * self.world_size)
                    if self.world_size > 1:
                        torch.distributed.all_reduce(c)
                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = off_diagonal(c).pow_(2).sum()
                    loss += on_diag + self.lambda_param * off_diag
            # Average the accumulated loss over the number of computed cross-correlation terms
            loss /= iteration
        return loss


def capture_rng_state():
    """
    Capture RNG state so training can be resumed deterministically.
    """
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """
    Restore RNG state saved by capture_rng_state().
    """
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
        # weights_only=False is required to safely load complex Python objects like optimizers states
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
