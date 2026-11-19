# SpiNNaker deployment and energy evaluation

This folder evaluates the energy of the trained models in two ways:

- **theoretical energy**, computed in PyTorch from the number of synaptic operations (SOPs);
- **physical energy and on-board accuracy**, obtained by converting each model into a sPyNNaker network and running it on a SpiNNaker board (SpiNN-5, sPyNNaker 7.4.1, through the EBRAINS SANDS JupyterLab).

The network run on the board is the trained network itself, unfolded neuron by neuron: same weights (up to the board's 16-bit fixed point), same neuron dynamics and same test samples. A software emulation of the converted network in fixed point serves as a reference to check that the board computed what it was asked to.

## Files

| File | Role |
| :--- | :--- |
| `prepare_spinnaker_suite.py` | Selects the best checkpoint of each configuration in `saved_models/` and writes the evaluation plan `spinnaker_plan.json` |
| `theoretical_energy_eval.py` | Counts the SOPs of every model of the plan and writes `theoretical_energy_results.csv` |
| `spinnaker_faithful_eval.py` | Converts each model of the plan, emulates it, deploys it on the board and records accuracy and reliability counters |
| `spinnaker_energy.py` | Reads the sPyNNaker energy reports and builds the final table (theoretical and physical energy) |
| `run_on_sands.ipynb` | Runs the two previous scripts on SANDS, in the order described below |

## 1. Preparation (workstation)

From the repository root, once the models are trained:

```bash
uv run python extract.py
uv run python SpiNNaker/prepare_spinnaker_suite.py
uv run python SpiNNaker/theoretical_energy_eval.py
```

Each entry of `spinnaker_plan.json` gives the dataset, architecture, `T`, `Vth`, the weights and the 30 sample files of one model. The run identifier used everywhere is `<dataset>_<architecture>_T<T>_Vth<Vth>`.

Then pack everything the board needs:

```bash
tar czf sands_bundle.tar.gz SpiNNaker src configs spinnaker_plan.json spinnaker_models spinnaker_data theoretical_energy_results.csv
```

## 2. Deployment (SANDS)

1. Log in to the EBRAINS JupyterLab ([lab.jsc.ebrains.eu](https://lab.jsc.ebrains.eu/)) and work inside `work/`, the only folder kept between sessions.
2. Upload `sands_bundle.tar.gz` to `work/` and extract it from a terminal: `cd work && tar xzf sands_bundle.tar.gz`.
3. Open `SpiNNaker/run_on_sands.ipynb` with the **sPyNNaker 7.4.1** kernel and run the cells in order. The notebook installs PyTorch (CPU) and SpikingJelly into the kernel if they are missing.

| Section | What it does | Duration |
| :------ | :----------- | :------- |
| 1–2 | Working folder, dependencies, environment check | < 1 min |
| 3 | Emulation of every run without the board: input bits, minimal time scale factor, cores needed. Sets `FACTEUR` (largest factor needed, shared by all runs) and `PIRE` (most loaded run) | a few minutes |
| 4 | Test of the most loaded run on the board at `FACTEUR`. Expected: `fiable=True`, `tardifs=0`, `accord` ≥ 90 % | a few minutes |
| 5 | All runs on the board at `FACTEUR`; already successful runs are skipped if the cell is run again | about 3 h + 0.2 h × `FACTEUR` / 10 |
| 6 | Accuracy and reliability table, list of runs to redo | < 1 min |
| 7 | Physical energy and final table | < 1 min |

A model must fit on one SpiNN-5 board (700 cores and 30 million synapses at most). In practice only the MLP and small CNN models of N-MNIST, S-MNIST, SHD and SSC fit, hence the list `DATASETS` of the notebook; larger models are reported as `TOO_LARGE` and keep only their theoretical energy.

The SpiNNaker results were obtained with a time scale factor of 30 for every run (72 runs: 6 networks × 3 `Vth` × 4 `T`, 30 samples each).

`spinnaker_faithful_eval.py` can also be called directly; its main options are:

| Option | Default | Description |
| :----- | :------ | :---------- |
| `--plan` | `spinnaker_plan.json` | Evaluation plan |
| `--results-dir` | `results_spinnaker` | Output folder |
| `--dataset`, `--architecture`, `--limit` | all | Restrict the runs of the plan |
| `--emulate` | off | Do not use the board: convert and emulate only |
| `--time-scale-factor` | `30` | Slow-down of the board (integer from 1 to 200), or `auto` for the minimum estimated per run |
| `--max-cores` | `700` | Core budget (a SpiNN-5 board has about 750 usable cores) |
| `--max-spike-buffer` | `8192` | Largest input buffer per core, in spikes |
| `--retry-unreliable` | off | Also rerun successful runs whose `reliable` is `False` |
| `--no-bitplane` | off | Send input counts as one packet per event (see below) |

## 3. Conversion of a model

- **Layers.** Each `Conv1d`, `Conv2d` and `Linear` layer is unfolded into an explicit list of synapses. Batch normalisation is folded into the preceding convolution and global average pooling into the following `Linear` layer; max pooling over spikes is a logical OR (a neuron that fires when any input of its window fires). Only such sequential models can be unfolded; other structures (padded or grouped convolutions, overlapping pooling, other neuron types) are reported as `UNSUPPORTED`.
- **Neurons.** Each LIF layer becomes a population of LIF neurons with instantaneous synapses and `tau_m = 1 / ln 2` ms, so that the membrane potential is halved at each 1 ms step, as in SpikingJelly's LIF with `tau = 2`. The thresholds are the trained `Vth`.
- **Output.** The output layer is computed on the host from the spike counts of the last hidden population: logits are the sum over the `T` steps of the output `Linear` layer, as in PyTorch.
- **Inputs.** Samples are sent one after the other through a `SpikeSourceArray`. Each sample occupies `T` steps, plus one step per layer for the spikes to cross the network and 16 rest steps for the membranes to return to rest. An input channel that emits `c` events in one step is split into one source per bit of `c` (weight `2^j` for bit `j`): the injected current is identical and fewer packets are sent.
- **Fixed point.** The board stores weights on 16 bits. The scaling of each population is chosen from the largest summed weight seen in emulation, so that inputs neither saturate nor lose their small weights.
- **Real time.** A core must process all the spikes it receives within one time step, which lasts `time_scale_factor` ms of real time. The script estimates this cost (8 µs per received spike and 0.04 µs per synaptic event, within 80 % of a step) and chooses, for each population, the number of neurons per core (128 to 8) and, with `auto`, the smallest factor (multiple of 10) for which every core keeps up and the network fits on the board. Input buffers are sized from the largest burst seen in emulation (256 to 8192 spikes).

## 4. Outputs

```text
results_spinnaker/
├── summary_emulation.csv        # Runs launched with --emulate
├── summary_benchmark.csv        # Runs on the board (one row per attempt; the last row of a run is the valid one)
├── predictions/<run_id>.csv     # Label and PyTorch, fixed-point and board predictions of each sample
├── hardware_reports/<dataset>/<run_id>/   # sPyNNaker reports of the run, including energy_report_1.rpt
├── spinnaker_energy.csv         # Physical energy of each successful run
└── final_energy_results.csv     # theoretical_energy_results.csv with energie_reelle_spinnaker filled in
```

`status` is `SUCCESS` (run on the board), `EMULATED` (`--emulate`), `TOO_LARGE`, `UNSUPPORTED`, `EMULATION_ERROR` or `HARDWARE_ERROR` (message in `error_msg`).

Main columns of `summary_benchmark.csv`:

| Column | Meaning |
| :----- | :------ |
| `accuracy_pytorch_pct` | Accuracy of the PyTorch model on the 30 samples |
| `accuracy_emulated_pct`, `accuracy_fixed_point_pct` | Accuracy of the converted network emulated in floating point and in the board's fixed point |
| `accuracy_pct` | Accuracy on the board |
| `agreement_fixed_point_pct` | Share of samples for which the board gives the same answer as the fixed-point emulation |
| `lost_input_packets`, `late_spikes`, `timer_overruns`, `router_lost_packets` | Losses read from the sPyNNaker provenance (`late_by_population` details the late spikes per layer) |
| `reliable` | `True` when all the loss counters above are zero |
| `time_scale_factor`, `needed_time_scale_factor`, `n_cores`, `min_neurons_per_core`, `input_bits` | Deployment settings of the run |

A run is valid when `reliable` is `True` and `agreement_fixed_point_pct` is at least 90 %.

## 5. Energy

**Theoretical energy** (`theoretical_energy_eval.py`): SOPs per inference × 0.9 pJ (`--energy-per-sop`). A SOP is one synaptic event triggered by a spike of the network; the events of the input layer are excluded unless `--include-input-sops` is given. `nombre_neurones` counts the hidden and output neurons, not the inputs.

**Physical energy** (`spinnaker_energy.py`): taken from the energy report that sPyNNaker writes after each run (`write_energy_report = True`; the script creates `spynnaker.cfg` with this option if the file does not exist). sPyNNaker computes it from a model of the board, not from a power meter: 0.35 W per active chip during the execution, 0.034 W per core while it is active, and about 1 nJ per routed packet. The value used in the final table is

```text
energy_per_inference_j = exec_energy_cores_j / n_samples
```

where `exec_energy_cores_j` is the execution energy of the active chips and cores only. `spinnaker_energy.csv` also gives the execution energy of the whole boards (`exec_energy_boards_j`), of the boards and their frame (`exec_energy_j`), the total energy including mapping and loading (`total_energy_j`), the execution time and the numbers of boards, chips and cores.

Because the static power of the active chips dominates, the physical energy depends mostly on the size of the network (number of chips) and on the execution time, which is proportional to `T` plus the rest steps and to the time scale factor. All runs must therefore use the same factor to be comparable.

The final table can also be rebuilt on the workstation after downloading `results_spinnaker/`:

```bash
uv run python SpiNNaker/spinnaker_energy.py --results-dir results_spinnaker
```
