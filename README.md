# Quantized-SpikeNet

A modular PyTorch farmework for training Convolution Spiking Neural Networks (CSNNs) on event-based neuromorphic datasets (N-MNIST, CIFAR10-DVS, DVS Gesture).

The primary goal of this project is to train SNNs and export **Fake-Quantized INT8 weights**, ensuring the models are strictly formatted and ready for physical deployment on hardware accelerators like **FPGA** and **SpiNNaker**.

## Installation

This project uses `uv` for dependency management. Ensure you have Python 3.12+.

1. Clone the repository:
```bash
git clone git@github.com:luhweigg/Quantized-SpikeNet.git
cd Quantized-SpikeNet
```

2. Install dependencies and setup the virtual environment:
```bash
uv sync
```

---

## Usage

The main entry point for training and evaluation is main.py. The script automatically handles caching, training, checkpointing and final weight quantization.

### Basic training run

### Available Arguments

---

The repository includes a comprehensive test suite to ensure the SNN's dynamics, hardware quantization constraints, and deterministic checkpoints remain intact.

Run the test using `pytest`:
```bash
uv run pytest
```

---