import os
import argparse
import json
import torch
import wandb
from datetime import datetime
from tqdm import tqdm
from src.engine import (
    build_components,
    train_one_epoch,
    evaluate,
    quantize_weights,
    save_checkpoint,
    load_checkpoint,
    capture_rng_state,
    CSVLogger,
    EarlyStopping,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="nmnist",
        choices=["nmnist", "cifar10", "dvs_gesture", "nepic_kitchens"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--Time", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="quantized_spikenet")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join("configs", f"{args.dataset}.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    print(
        f"Device: {device} | Dataset: {args.dataset} | Architecture: {model_config['architecture']} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Time: {args.Time}"
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config={**vars(args), **model_config})

    if args.resume:
        if args.resume.endswith(".pth"):
            if not os.path.isfile(args.resume):
                raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        else:
            resume_candidate = os.path.join(args.resume, "checkpoint_latest.pth")
            if not os.path.isfile(resume_candidate):
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_candidate}"
                )

    model, train_loader, test_loader, optimizer, scheduler, criterion, scaler = (
        build_components(
            args.dataset,
            model_config,
            args.batch_size,
            args.Time,
            args.lr,
            args.epochs,
            device,
        )
    )

    if args.resume:
        resume_path = (
            args.resume
            if args.resume.endswith(".pth")
            else os.path.join(args.resume, "checkpoint_latest.pth")
        )
        args.save_dir = (
            args.resume
            if not args.resume.endswith(".pth")
            else os.path.dirname(args.resume)
        )
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(args.save_dir, args.dataset, f"run_{current_time}")
        resume_path = os.path.join(args.save_dir, "checkpoint_latest.pth")

    os.makedirs(args.save_dir, exist_ok=True)

    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
    else:
        start_epoch, best_acc = 0, 0.0

    early_stopping = EarlyStopping(patience=7, delta=0.001, mode="max")

    csv_path = os.path.join(args.save_dir, "training_log.csv")
    csv_logger = CSVLogger(
        csv_path,
        [
            "epoch",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "test_sparsity",
            "energy_nano_joules",
            "power_nano_watts",
        ],
    )

    pbar = tqdm(
        range(start_epoch, args.epochs),
        desc="Global Training",
        initial=start_epoch,
        total=args.epochs,
    )

    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        test_loss, test_acc, test_sparsity, energy_joules, power_watts = evaluate(
            model, test_loader, criterion, device, measure_consumption=True
        )
        scheduler.step()

        csv_logger.log(
            [
                epoch + 1,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                test_sparsity,
                energy_joules * 1e9,
                power_watts * 1e9,
            ]
        )

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        checkpoint_state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
        }
        if scaler is not None:
            checkpoint_state["scaler"] = scaler.state_dict()

        save_checkpoint(checkpoint_state, is_best, args.save_dir)

        if is_best:
            tqdm.write(
                f"Epoch {epoch + 1}: New best model saved (Acc: {test_acc:.2f}%)"
            )

        pbar.set_postfix(
            {
                "Tr Loss": f"{train_loss:.4f}",
                "Tr Acc": f"{train_acc:.2f}%",
                "Te Loss": f"{test_loss:.4f}",
                "Te Acc": f"{test_acc:.2f}%",
                "Sparsity": f"{test_sparsity:.2f}%",
            }
        )

        if args.use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/sparsity": test_sparsity,
                    "test/energy_joules": energy_joules * 1e9,
                    "test/power_watts": power_watts * 1e9,
                    "epoch": epoch + 1,
                }
            )

        early_stopping(test_acc)
        if early_stopping.early_stop:
            tqdm.write(
                f"Early stopping triggered at epoch {epoch + 1}. Best validation accuracy: {best_acc:.2f}%"
            )
            break

    base_path = os.path.join(args.save_dir, f"{args.dataset}_base.pth")
    torch.save(model.state_dict(), base_path)

    quantized_weights, quantization_metadata = quantize_weights(
        model, num_bits=8, return_metadata=True
    )
    _, _, _, _, _ = evaluate(model, test_loader, criterion, device)

    quantized_path = os.path.join(args.save_dir, f"{args.dataset}_quantized.pth")
    torch.save(
        {"weights": quantized_weights, "metadata": quantization_metadata},
        quantized_path,
    )
    print(f"Integer weights (int8) ready for the FPGA saved in : {quantized_path}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
