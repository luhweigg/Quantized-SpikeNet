import os
import tempfile
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import SpikingMLP
from src.engine import train_one_epoch
from src.engine import save_checkpoint


def set_seed(seed=67):
    """
    Set a fixed random seed for reproducibility across runs.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_checkpoint_resume_parity():
    """
    Verify that an interrupted training followed by a resume yields EXACTLY the same result as a continuous training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(67)
    batch_size, T, C, H, W = 4, 16, 2, 34, 34
    events = (torch.rand(T, batch_size, C, H, W) > 0.5).float().to(device)
    targets = torch.randint(0, 10, (batch_size,)).to(device)
    fake_loader = [(events, targets)]
    criterion = nn.CrossEntropyLoss()

    set_seed(67)
    model_A = SpikingMLP(input_size=2312, hidden_size=256, output_size=10).to(device)
    opt_A = torch.optim.AdamW(model_A.parameters(), lr=1e-3)
    sch_A = CosineAnnealingLR(opt_A, T_max=4)

    for _ in range(4):
        loss_A, _ = train_one_epoch(model_A, fake_loader, opt_A, criterion, device)
        sch_A.step()

    weights_A = {k: v.clone() for k, v in model_A.state_dict().items()}

    set_seed(67)
    model_B = SpikingMLP(input_size=2312, hidden_size=256, output_size=10).to(device)
    opt_B = torch.optim.AdamW(model_B.parameters(), lr=1e-3)
    sch_B = CosineAnnealingLR(opt_B, T_max=4)

    for _ in range(2):
        train_one_epoch(model_B, fake_loader, opt_B, criterion, device)
        sch_B.step()

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_state = {
            "epoch": 2,
            "state_dict": model_B.state_dict(),
            "optimizer": opt_B.state_dict(),
            "scheduler": sch_B.state_dict(),
            "best_acc": 0.0,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        save_checkpoint(checkpoint_state, False, tmp_dir)

        model_C = SpikingMLP(input_size=2312, hidden_size=256, output_size=10).to(
            device
        )
        opt_C = torch.optim.AdamW(model_C.parameters(), lr=1e-3)
        sch_C = CosineAnnealingLR(opt_C, T_max=4)

        resume_path = os.path.join(tmp_dir, "checkpoint_latest.pth")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        model_C.load_state_dict(checkpoint["state_dict"])
        opt_C.load_state_dict(checkpoint["optimizer"])
        sch_C.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]

        torch.set_rng_state(checkpoint["rng_state"])
        if checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

        for _ in range(start_epoch, 4):
            loss_C, _ = train_one_epoch(model_C, fake_loader, opt_C, criterion, device)
            sch_C.step()

    assert abs(loss_A - loss_C) < 1e-6, (
        f"Désynchro de la Loss ! Continu: {loss_A}, Reprise: {loss_C}"
    )

    weights_C = model_C.state_dict()
    for key in weights_A.keys():
        assert torch.equal(weights_A[key], weights_C[key]), (
            f"Désynchro des poids sur la couche : {key}"
        )
