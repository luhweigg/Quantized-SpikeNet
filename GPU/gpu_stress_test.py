import torch
import time
import os
import sys
import threading
import subprocess
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from spikingjelly.activation_based.base import MemoryModule
from src.models import SpikingMLP, SpikingVGG4, SpikingVGG5, SpikingVGG8

DATASET = (
    "nepic_kitchens"  # Choices : "nmnist", "cifar10_dvs", "dvs_gesture", "nepic_kitchens"
)
SIMULATION_TIME_MS = 20
NOISE_RATE_HZ = 50.0
ENERGY_PER_SPIKE_JOULES = 0.9e-12
ASSUMED_POWER_WATTS = 250.0 if torch.cuda.is_available() else 80.0

WEIGHTS_PATHS = {
    "nmnist": "networks/nmnist_best.pth",
    "cifar10_dvs": "networks/cifar10_dvs_best.pth",
    "dvs_gesture": "networks/dvs_gesture_best.pth",
    "nepic_kitchens": "networks/nepic_kitchens_best.pth",
}


def generate_poisson_noise(shape, rate_hz, time_ms, device):
    """
    Generate a tensor of random spikes mimicking the SpikeSourcePoisson of SpiNNaker.
    rate_hz: e.g., 50Hz = 50 spikes / 1000 ms = probability of 0.05 per ms.
    """
    prob_per_ms = rate_hz / 1000.0
    noise_shape = (time_ms, 1, *shape)

    print(f"[*] Generation of the poisson noise ({rate_hz} Hz) during {time_ms} ms...")
    noise_tensor = (torch.rand(noise_shape, device=device) < prob_per_ms).float()
    return noise_tensor


class GPUPowerMonitor:
    """
    Background thread to measure real GPU power draw using nvidia-smi.
    """

    def __init__(self):
        self.measurements = []
        self.is_running = False
        self.thread = None

    def _monitor_loop(self):
        while self.is_running:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                power_val = float(result.stdout.strip().split("\n")[0])
                self.measurements.append(power_val)
            except Exception:
                pass
            time.sleep(0.2)

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join()

        if self.measurements:
            return sum(self.measurements) / len(self.measurements)
        return None


class EnergyProfiler:
    """
    Attach hooks to the network to count each spike.
    """

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
    elif DATASET == "cifar10_dvs":
        model = SpikingVGG4(in_channels=2, out_classes=10).to(device)
        input_shape = (2, 32, 32)
    elif DATASET == "dvs_gesture":
        model = SpikingVGG5(in_channels=2, out_classes=11, init_stride=2).to(device)
        input_shape = (2, 128, 128)
    elif DATASET == "nepic_kitchens":
        model = SpikingVGG8(in_channels=1, out_classes=8, init_stride=4).to(device)
        input_shape = (1, 304, 384)
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
            "[!] WARNING: No weights found. Forcing high random weights to avoid a dead network!"
        )
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.uniform_(m.weight, 0.5, 1.0)

    model.eval()

    noise_events = generate_poisson_noise(
        input_shape, NOISE_RATE_HZ, SIMULATION_TIME_MS, device
    )

    profiler = EnergyProfiler(model)
    gpu_monitor = GPUPowerMonitor()

    print(f"[*] Beginning of the simulation ({SIMULATION_TIME_MS} ms)...")

    gpu_monitor.start()
    start_time = time.perf_counter()

    with torch.no_grad():
        _ = model(noise_events)

    end_time = time.perf_counter()
    avg_measured_power = gpu_monitor.stop()

    real_execution_time = end_time - start_time

    model.reset_states()
    profiler.remove_hooks()

    total_spikes = profiler.total_spikes
    energy_joules = total_spikes * ENERGY_PER_SPIKE_JOULES
    theoretical_power_watts = energy_joules / (SIMULATION_TIME_MS / 1000.0)

    if avg_measured_power is not None:
        used_power_watts = avg_measured_power
        power_source = "Live Measured (nvidia-smi)"
    else:
        used_power_watts = ASSUMED_POWER_WATTS
        power_source = "Assumed (Fallback)"

    real_hardware_energy = used_power_watts * real_execution_time

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
    --------------------------------------------------
    [C] REAL HARDWARE ENERGY (Physical Estimation)
    Power Source            : {power_source}
    Average Power Draw      : {used_power_watts:.2f} W
    Real Energy Consumed    : {real_hardware_energy:.2f} Joules
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
        log_line = f"[{timestamp}] {model.__class__.__name__} | Spikes: {int(total_spikes):,} | Real GPU Energy: {real_hardware_energy:.2f} J\n"
        f.write(log_line)

    print(f"\n Detailed report saved in: {report_file_path}")
    print(f" History updated at      : {history_file_path}")


if __name__ == "__main__":
    run_pytorch_profiling()
