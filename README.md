# Quantized-SpikeNet

Training of spiking neural networks (SNNs) on event-based datasets, and evaluation of their accuracy and energy consumption on the SpiNNaker neuromorphic platform.

Each model is trained with [SpikingJelly](https://github.com/fangwei123456/spikingjelly) for one combination of dataset, architecture, number of time steps `T` and firing threshold `Vth`. Every trained model is then evaluated in two ways:

- **theoretical energy**: number of synaptic operations (SOPs) per inference, multiplied by an energy per SOP (0.9 pJ by default);
- **physical energy**: the converted network is run on a SpiNNaker board, and the energy is taken from the sPyNNaker energy report, together with the accuracy obtained on the board.

Comparing both energies with the accuracy across `T` and `Vth` gives the accuracy/energy trade-off studied in the article.

## Pipeline

All commands are run from the repository root.

| Step | Script | Output |
| :--- | :----- | :----- |
| 1. Download the datasets not available through Tonic | `download_custom_datasets.py` | `data/` |
| 2. Train one model, or the full grid | `main.py`, `massive_sweep.py` | `saved_models/` |
| 3. Extract the test samples used for evaluation | `extract.py` | `spinnaker_data/` |
| 4. Gather the trained models into an evaluation plan | `SpiNNaker/prepare_spinnaker_suite.py` | `spinnaker_plan.json`, `spinnaker_models/` |
| 5. Compute the theoretical energy | `SpiNNaker/theoretical_energy_eval.py` | `theoretical_energy_results.csv` |
| 6. Deploy on SpiNNaker (EBRAINS SANDS) | `SpiNNaker/run_on_sands.ipynb` | `results_spinnaker/` |
| 7. Collect the physical energy and build the final table | `SpiNNaker/spinnaker_energy.py` | `results_spinnaker/final_energy_results.csv` |

Steps 1 to 5 run on a workstation (a GPU is recommended for training). Steps 6 and 7 run on the SpiNNaker JupyterLab; they are described in [`SpiNNaker/README.md`](SpiNNaker/README.md).

## Repository layout

```text
.
├── main.py                       # Training of one model, INT8 export
├── massive_sweep.py              # Training of the full grid (datasets × architectures × T × Vth)
├── extract.py                    # Test samples as spike times, for the energy evaluations
├── download_custom_datasets.py   # EDAT24, Ceolini gestures and STEMNIST (Zenodo)
├── configs/                      # One file per dataset: allowed architectures and their hyperparameters
├── src/
│   ├── data_loaders/             # One loader per dataset (event-to-frame conversion with T bins)
│   ├── models/                   # MLP, 1D/2D CNN, VGG and ResNet SNNs, model factory
│   ├── engine/                   # Training loop, evaluation, checkpoints, quantization
│   ├── domain/                   # Model interface
│   └── tests/                    # Test suite (pytest)
└── SpiNNaker/                    # Theoretical energy and SpiNNaker deployment (see SpiNNaker/README.md)
```

## Installation

The project uses [`uv`](https://github.com/astral-sh/uv) and Python 3.12 or 3.13.

```bash
git clone https://github.com/luhweigg/Quantized-SpikeNet.git
cd Quantized-SpikeNet
uv sync
```

## Datasets

Datasets are stored in `data/`. Most of them are downloaded automatically by [Tonic](https://github.com/neuromorphs/tonic) the first time they are used. Three datasets are downloaded from Zenodo with:

```bash
uv run python download_custom_datasets.py
```

N-EPIC-Kitchens must be downloaded manually into `data/N-EPIC-Kitchens/` (files `D1_train.pkl` to `D3_test.pkl` and the `voxels_xy_3/` folder).

| `--dataset`        | Dataset            | Classes | Source            | Architectures (default first)                                    |
| :----------------- | :----------------- | :-----: | :---------------- | :--------------------------------------------------------------- |
| `nmnist`           | N-MNIST            |   10    | Tonic             | SpikingMLP, CompactSpikingCNN, SpikingVGG3/4/5/8, SpikingResNet18/34 |
| `smnist`           | S-MNIST (spiking sequential MNIST) |   10    | Tonic             | Spiking1DCNN                                                     |
| `shd`              | Spiking Heidelberg Digits | 20 | Tonic          | Spiking1DCNN, SpikingMLP                                         |
| `ssc`              | Spiking Speech Commands | 35 | Tonic            | Spiking1DCNN                                                     |
| `cifar10_dvs`      | CIFAR10-DVS        |   10    | Tonic             | SpikingVGG4, SpikingMLP, CompactSpikingCNN, SpikingVGG3/8, SpikingResNet18/34 |
| `dvs_gesture`      | DVS Gesture        |   11    | Tonic             | SpikingVGG5, SpikingMLP, CompactSpikingCNN, SpikingVGG3/4/8, SpikingResNet18/34 |
| `dvs_lip`          | DVS-Lip            |   100   | Tonic             | SpikingResNet18, SpikingMLP, CompactSpikingCNN, SpikingVGG3/4/5/8, SpikingResNet34 |
| `ncaltech101`      | N-Caltech101       |   101   | Tonic             | SpikingVGG5, SpikingMLP, CompactSpikingCNN, SpikingVGG3/4/8, SpikingResNet18/34 |
| `nepic_kitchens`   | N-EPIC-Kitchens    |    8    | Manual            | SpikingVGG8, SpikingMLP, CompactSpikingCNN, SpikingVGG3/4/5, SpikingResNet18/34 |
| `edat24`           | EDAT24             |    4    | Zenodo (script)   | CompactSpikingCNN, SpikingVGG4                                   |
| `ceolini_gestures` | Ceolini gestures   |    5    | Zenodo (script)   | CompactSpikingCNN, SpikingVGG4                                   |
| `stemnist`         | STEMNIST           |   35    | Zenodo (script)   | CompactSpikingCNN, SpikingVGG4                                   |

The architectures allowed for each dataset, and their hyperparameters (input size, channels, dropout), are defined in `configs/<dataset>.json`.

## Training

All models use LIF neurons (SpikingJelly, `tau = 2`) trained with surrogate gradients, AdamW (weight decay `1e-2`) and a cosine learning-rate schedule. Training stops early when the test accuracy has not improved for 7 epochs.

```bash
# One model
uv run python main.py --dataset shd --architecture SpikingMLP --time 16 --v_threshold 1.0

# Resume an interrupted run
uv run python main.py --dataset shd --resume saved_models/shd/run_SpikingMLP_T16_Vth1.0_20261101_120000
```

| Argument               | Default              | Description                                                      |
| :--------------------- | :------------------- | :--------------------------------------------------------------- |
| `--dataset`            | `nmnist`             | Dataset key (see the table above)                                |
| `--architecture`       | from the config      | Architecture; must be listed in `configs/<dataset>.json`         |
| `--time`               | `16`                 | Number of time steps `T` (time bins of the event-to-frame conversion) |
| `--v_threshold`        | `1.0`                | Firing threshold `Vth` of the LIF neurons                        |
| `--alpha`              | model default        | Sharpness of the ATan surrogate gradient (VGG and ResNet only)   |
| `--epochs`             | `20`                 | Maximum number of epochs                                         |
| `--batch_size`         | `64`                 | Batch size                                                       |
| `--accumulation_steps` | `1`                  | Gradient accumulation steps (effective batch = batch size × steps) |
| `--lr`                 | `1e-3`               | Initial learning rate                                            |
| `--save_dir`           | `./saved_models`     | Root folder of the runs                                          |
| `--resume`             | `None`               | Run folder (or `.pth` checkpoint) to resume from                 |
| `--use_wandb`          | off                  | Log metrics to Weights & Biases                                  |
| `--wandb_project`      | `Quantized-SpikeNet` | W&B project prefix (the dataset name is appended)                |

Each run writes its own folder:

```text
saved_models/<dataset>/run_<architecture>_T<T>_Vth<Vth>_<UTC date>_<UTC time>/
├── checkpoint_latest.pth      # Full training state (model, optimizer, scheduler, RNG), used by --resume
├── model_best.pth             # Checkpoint with the best test accuracy (used for the energy evaluations)
├── training_log.csv           # Loss, accuracy, sparsity and spike-based energy estimate per epoch
├── <dataset>_base.pth         # Floating-point weights at the end of training
└── <dataset>_quantized.pth    # INT8 weights and per-tensor scales (symmetric quantization)
```

### Full grid

`massive_sweep.py` trains every architecture of every dataset for `T ∈ {4, 8, 16, 32}` and `Vth ∈ {0.5, 1.0, 1.5}` (20 epochs, effective batch size 64 through gradient accumulation). Failed runs are listed in `sweep_errors.log`. The sweep logs to Weights & Biases: log in with `uv run wandb login`, or set `WANDB_MODE=offline`.

```bash
uv run python massive_sweep.py
```

## Energy evaluation

```bash
uv run python extract.py                               # 30 test samples per dataset and T
uv run python SpiNNaker/prepare_spinnaker_suite.py     # best checkpoint of each configuration -> spinnaker_plan.json
uv run python SpiNNaker/theoretical_energy_eval.py     # SOPs and theoretical energy of every model of the plan
```

- `extract.py` stores the first 30 samples of each test split, for every `T`, as spike times (`spinnaker_data/<dataset>/T_<T>/`). The same samples are used for the theoretical energy and on SpiNNaker.
- `prepare_spinnaker_suite.py` keeps, for each (dataset, architecture, `T`, `Vth`), the `model_best.pth` of the most recent run and copies it to `spinnaker_models/`.
- `theoretical_energy_eval.py` writes `theoretical_energy_results.csv` (one row per model: neurons, synapses, accuracy, SOPs per inference, theoretical energy in joules), with per-run and per-layer details in `theoretical_energy_results_details.csv` and `theoretical_energy_results_per_layer.csv`. SOPs count the synaptic events triggered by the spikes of the network's neurons; add `--include-input-sops` to also count the events of the input layer.

The deployment on SpiNNaker and the measurement of the physical energy are described in [`SpiNNaker/README.md`](SpiNNaker/README.md).

## Tests

```bash
uv run pytest
```

The tests use synthetic inputs and do not need the datasets, except `src/tests/test_all_datasets.py`, which trains one epoch on real data for the datasets listed at the top of the file (none by default).

The CI (`.github/workflows/ci.yml`) checks the formatting with `uv run ruff format --check .` and runs the test suite.
