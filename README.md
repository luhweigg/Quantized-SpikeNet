# ⚡ Quantized-SpikeNet

> A modular PyTorch framework for training Convolutional Spiking Neural Networks (CSNNs) on event-based neuromorphic datasets.

The primary goal of this project is to train robust SNNs and export **Fake-Quantized INT8 weights**. This ensures the models are strictly formatted, lightweight, and ready for physical deployment on neuromorphic hardware accelerators like **FPGA** and **SpiNNaker**.

---

## ✨ Key Features

- **Hardware-Ready Export:** Automatically extracts and saves INT8 quantized weights and metadata for seamless FPGA deployment.
- **Robust Training Pipeline:** Built-in regularizations (L2 Weight Decay, Dropout) to prevent overfitting on complex event data.
- **Smart Experiment Tracking:**
  - Dynamic run directories (`run_YYYYMMDD_HHMMSS`) to safely store weights without overriding.
  - Automatic **WandB** integration and local **CSV logging** for metric tracking.
- **Early Stopping & Checkpointing:** Saves the best model based on validation accuracy and halts training if the model stagnates.
- **State Restoration:** Flawless training resumption (restores optimizer, scheduler, scaler, and RNG states).

---

## 🧠 Supported Architectures & Datasets

The framework provides specific architectures tuned for event-based vision tasks:

| Dataset | Default Architecture | Task | Classes |
| :--- | :--- | :--- | :---: |
| **N-MNIST** | `SpikingMLP` | Neuromorphic Digit Recognition | 10 |
| **CIFAR10-DVS** | `CompactSpikingCNN` / `SpikingVGG4` | Complex Object Classification | 10 |
| **DVS Gesture** | `SpikingVGG5` | Dynamic Hand Gesture Recognition | 11 |
| **N-EPIC Kitchens** | `SpikingVGG11` / `SpikingResNet18` | Ego-centric Action Recognition | Multi |

*Other available architectures: `SpikingVGG3`.*

---

## ⚙️ Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for blazing-fast dependency management. Ensure you have **Python 3.12+** installed on your machine.

**1. Clone the repository:**
```bash
git clone git@github.com:luhweigg/Quantized-SpikeNet.git
cd Quantized-SpikeNet
```

**2. Install dependencies & setup virtual environment:**
```bash
uv sync
```

---

## 🚀 Usage

The main entry point is `main.py`. The script automatically handles data caching, training loops, metric logging, checkpointing, and final weight quantization.

### Basic Training Run

Launch a standard training session. The framework will automatically create a timestamped folder for your run.

```bash
uv run python main.py --dataset dvs_gesture --architecture SpikingVGG5 --epochs 50 --batch_size 16 --Time 20 --use_wandb
```

### Resume an Interrupted Run

Did your machine crash? No problem. Pass the path to your run directory, and the framework will pick up exactly where it left off.

```bash
uv run python main.py --dataset dvs_gesture --resume saved_models/dvs_gesture/run_20231027_143000
```

### 🎛️ Available Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | `str` | `nmnist` | Target dataset (`nmnist`, `cifar10`, `dvs_gesture`, `nepic_kitchens`) |
| `--architecture` | `str` | `None` | Specific model to use (e.g., `SpikingVGG11`, `SpikingResNet18`) |
| `--epochs` | `int` | `20` | Maximum number of training epochs |
| `--batch_size` | `int` | `64` | Number of samples per batch |
| `--lr` | `float` | `1e-3` | Initial learning rate |
| `--Time` | `int` | `16` | Number of time bins for the event-to-frame transform |
| `--resume` | `str` | `None` | Path to a run directory to resume training |
| `--use_wandb` | `flag` | `False` | Enable Weights & Biases logging |

---

## 📁 Output Structure

For every run, the framework generates a comprehensive package of artifacts:

```text
saved_models/
└── dvs_gesture/
    └── run_20231027_143000/
        ├── checkpoint_latest.pth           # Full state for easy resumption
        ├── model_best.pth                  # Best performing model 
        ├── dvs_gesture_base.pth            # Final floating-point weights
        ├── dvs_gesture_quantized.pth       # INT8 weights ready for FPGA
        └── training_log_SpikingVGG5.csv    # Complete epoch-by-epoch metrics
```

---

## 🧪 Testing

The repository includes a comprehensive test suite to ensure the SNN's dynamics, hardware quantization constraints, and deterministic checkpoints remain fully functional.

Run the tests using `pytest`:

```bash
uv run pytest
```
