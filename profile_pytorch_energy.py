import torch
import time
import os
from datetime import datetime
from spikingjelly.activation_based.base import MemoryModule
from src.models import SpikingMLP, SpikingVGG4, SpikingVGG5

DATASET = "dvs_gesture"  # Choices : "nmnist", "cifar10", "dvs_gesture"
SIMULATION_TIME_MS = 20
NOISE_RATE_HZ = 50.0

ENERGY_PER_SPIKE_JOULES = 0.9e-12

WEIGHTS_PATHS = {
    "nmnist": "networks/nmnist_best.pth",
    "cifar10": "networks/cifar10_best.pth",
    "dvs_gesture": "networks/dvs-gesture_best.pth",
}


def generate_poisson_noise(shape, rate_hz, time_ms, device):
    prob_per_ms = rate_hz / 1000.0
    noise_shape = (time_ms, 1, *shape)

    print(f"[*] Generation of the poisson noise ({rate_hz} Hz) during {time_ms} ms...")
    noise_tensor = (torch.rand(noise_shape, device=device) < prob_per_ms).float()
    return noise_tensor


class EnergyProfiler:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.total_spikes = 0
        self.total_neurons = 0

        def hook(module, inputs, output):
            spikes = output.detach().sum().item()
            neurons = output.detach().numel()
            self.total_spikes += spikes
            self.total_neurons += neurons

        for m in model.modules():
            if isinstance(m, MemoryModule):
                self.hooks.append(m.register_forward_hook(hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def run_pytorch_profiling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Deployment PyTorch Profiling on : {device.type.upper()} ===")

    if DATASET == "nmnist":
        model = SpikingMLP(input_size=2312, hidden_size=256, output_size=10).to(device)
        input_shape = (2, 34, 34)
    elif DATASET == "cifar10":
        model = SpikingVGG4(in_channels=2, out_classes=10).to(device)
        input_shape = (2, 32, 32)
    elif DATASET == "dvs_gesture":
        model = SpikingVGG5(in_channels=2, out_classes=11, init_stride=2).to(device)
        input_shape = (2, 128, 128)
    else:
        raise ValueError("Dataset inconnu.")

    pth_path = WEIGHTS_PATHS.get(DATASET, "")
    if os.path.exists(pth_path):
        print(f"[*] Loading trained weights from {pth_path}...")
        state_dict = torch.load(pth_path, map_location=device, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        print(
            "[!] WARNING: No weights found at paths. Forcing high random weights to avoid a dead network!"
        )
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.uniform_(m.weight, 0.5, 1.0)

    model.eval()

    noise_events = generate_poisson_noise(
        input_shape, NOISE_RATE_HZ, SIMULATION_TIME_MS, device
    )

    profiler = EnergyProfiler(model)
    print(f"[*] Beginning of the simulation ({SIMULATION_TIME_MS} ms)...")

    start_time = time.perf_counter()

    with torch.no_grad():
        _ = model(noise_events)

    end_time = time.perf_counter()
    real_execution_time = end_time - start_time

    model.reset_states()
    profiler.remove_hooks()

    total_spikes = profiler.total_spikes
    energy_joules = total_spikes * ENERGY_PER_SPIKE_JOULES
    theoretical_power_watts = energy_joules / (SIMULATION_TIME_MS / 1000.0)

    report_text = f"""
==================================================
 PYTORCH ENERGY REPORT : {DATASET.upper()}
==================================================
 Architecture        : {model.__class__.__name__}
 Simulated Time      : {SIMULATION_TIME_MS} ms
 Injected Noise      : {NOISE_RATE_HZ} Hz (Poisson)
--------------------------------------------------
 Total Simulated Neurons : {profiler.total_neurons:,}
 Total Generated Spikes  : {int(total_spikes):,}
--------------------------------------------------
 [A] THEORETICAL NETWORK ENERGY (Pure Computation)
 Energy (Joules)         : {energy_joules:.6e} J
 Net Power               : {theoretical_power_watts * 1000:.4f} mW
--------------------------------------------------
 [B] HARDWARE OVERHEAD ({device.type.upper()})
 Real Execution Time     : {real_execution_time:.4f} seconds
=================================================="""

    print(report_text)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    report_dir = os.path.join("reports", DATASET, timestamp)
    os.makedirs(report_dir, exist_ok=True)

    report_file_path = os.path.join(report_dir, "pytorch_energy_report.txt")
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    history_file_path = os.path.join("reports", DATASET, "history.log")
    with open(history_file_path, "a", encoding="utf-8") as f:
        log_line = f"[{timestamp}] {model.__class__.__name__} | Spikes: {int(total_spikes):,} | Energy: {energy_joules:.6e} J | Theoretical Power: {theoretical_power_watts * 1000:.4f} mW\n"
        f.write(log_line)

    print(f"\n Detailed report saved in: {report_file_path}")
    print(f" History updated at : {history_file_path}")


if __name__ == "__main__":
    run_pytorch_profiling()
