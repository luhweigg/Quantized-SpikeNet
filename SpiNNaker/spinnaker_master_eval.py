import argparse
import csv
import json
import os
import shutil
import time
import numpy as np
import torch
import pyNN.spiNNaker as sim

# ==========================================
# CONFIGURATION DU BENCHMARK SPINNAKER
# ==========================================
PLAN_PATH = "spinnaker_plan.json"
RESULTS_DIR = "results_spinnaker"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_benchmark.csv")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
HW_REPORTS_DIR = os.path.join(RESULTS_DIR, "hardware_reports")

DT = 1.0
TIME_SCALE_FACTOR = 4000
REST_TIME_MS = 10.0          # Pause entre deux échantillons pour réinitialiser la membrane
MAX_SPIKES_PER_NEURON = 10   # Limite par échantillon pour protéger les buffers d'entrée

# Budget global <= 18 096 neurones par réseau (~285 cœurs max sur les ~750 utiles d'une SpiNN-5)
MAX_INPUT_NEURONS = 4096     # Plafond pour la couche SpikeSourceArray
MAX_HIDDEN_BUDGET = 14000    # Budget réparti dynamiquement entre toutes les couches cachées
TOP_K_SYNAPSES = 16          # Connexions les plus fortes conservées par neurone post-synaptique
WEIGHT_SCALE = 2.0           # Facteur d'échelle des poids pour SpiNNaker

BASE_LIF_PARAMS = {
    "v_rest": 0.0,
    "v_reset": 0.0,
    "tau_m": 1.4427,
    "tau_syn_E": 0.5,
    "tau_syn_I": 0.5,
    "tau_refrac": 0.0,
}

def init_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(HW_REPORTS_DIR, exist_ok=True)

    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "dataset", "architecture", "T", "Vth",
                "status", "accuracy_pct", "correct_samples", "total_samples",
                "sim_time_ms", "wall_time_s", "total_neurons", "total_synapses", "error_msg"
            ])

def get_completed_runs():
    """Lit le CSV pour ignorer les modèles déjà évalués avec succès."""
    completed = set()
    if not os.path.exists(SUMMARY_CSV):
        return completed
    with open(SUMMARY_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "SUCCESS":
                completed.add(row["run_id"])
    return completed

def archive_spinnaker_report(dataset, run_id):
    """Archive le rapport matériel généré par sPyNNaker."""
    report_dir = "reports"
    if not os.path.exists(report_dir):
        return ""
    subdirs = [
        os.path.join(report_dir, d) for d in os.listdir(report_dir)
        if os.path.isdir(os.path.join(report_dir, d)) and d.startswith("20")
    ]
    if not subdirs:
        return ""
    latest_report = max(subdirs, key=os.path.getmtime)
    target_path = os.path.join(HW_REPORTS_DIR, dataset, run_id)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    shutil.move(latest_report, target_path)
    return target_path


def load_multiplexed_spikes(sample_files, effective_input_size, raw_input_size, sample_duration_ms):
    """
    Charge les 30 JSON, applique le sous-échantillonnage spatial si l'entrée dépasse
    MAX_INPUT_NEURONS, et décale temporellement chaque échantillon.
    """
    all_spike_times = [[] for _ in range(effective_input_size)]
    expected_labels = []
    window_ms = sample_duration_ms + REST_TIME_MS
    stride = max(1, raw_input_size // effective_input_size)

    for idx, file_path in enumerate(sample_files):
        time_offset = idx * window_ms
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_spikes = data["spike_times"]
        expected_labels.append(int(data["label"]))

        for target_id in range(effective_input_size):
            src_id = min(target_id * stride, len(raw_spikes) - 1)
            if isinstance(raw_spikes[src_id], list) and raw_spikes[src_id]:
                valid_spikes = raw_spikes[src_id][:MAX_SPIKES_PER_NEURON]
                shifted = [float(s + time_offset) for s in valid_spikes if s < sample_duration_ms]
                all_spike_times[target_id].extend(shifted)

    return all_spike_times, expected_labels

def extract_topology_and_synapses(state_dict, effective_input_size):
    """
    Extrait les couches du .pth en répartissant dynamiquement MAX_HIDDEN_BUDGET
    et construit les listes de connexions excitatrices et inhibitrices (Top-K).
    """
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    weight_keys = [
        k for k in state_dict.keys()
        if "weight" in k and "downsample" not in k and state_dict[k].ndim >= 2
    ]

    num_hidden_layers = max(1, len(weight_keys) - 1)
    max_per_hidden_layer = max(128, MAX_HIDDEN_BUDGET // num_hidden_layers)

    layer_sizes = [effective_input_size]
    synapse_lists = []
    total_synapses = 0

    for idx, key in enumerate(weight_keys):
        w = state_dict[key].detach().cpu().float()
        is_last = (idx == len(weight_keys) - 1)
        w_2d = w.view(w.shape[0], -1) if w.ndim == 4 else w

        out_dim = w_2d.shape[0]
        pre_size = layer_sizes[-1]

        post_size = out_dim if is_last else min(max_per_hidden_layer, max(out_dim, 64))
        layer_sizes.append(post_size)

        exc_conns = []
        inh_conns = []

        for post_idx in range(post_size):
            row = w_2d[post_idx % out_dim]
            k = min(TOP_K_SYNAPSES, row.numel())
            top_vals, top_indices = torch.topk(torch.abs(row), k=k)

            for _, raw_pre_idx in zip(top_vals, top_indices):
                weight_val = float(row[raw_pre_idx].item()) * WEIGHT_SCALE
                if abs(weight_val) < 1e-4:
                    continue

                pre_idx = int((raw_pre_idx.item() / max(1, row.numel())) * pre_size)
                pre_idx = min(pre_idx, pre_size - 1)

                if weight_val > 0:
                    exc_conns.append((pre_idx, post_idx, abs(weight_val), DT))
                else:
                    inh_conns.append((pre_idx, post_idx, abs(weight_val), DT))

        total_synapses += len(exc_conns) + len(inh_conns)
        synapse_lists.append((exc_conns, inh_conns))

    return layer_sizes, synapse_lists, total_synapses

def run_single_evaluation(entry):
    dataset = entry["dataset"]
    arch = entry["architecture"]
    t_val = entry["T"]
    vth_val = entry["Vth"]
    weights_path = entry["weights_path"]
    meta_path = entry["metadata_path"]
    sample_files = entry["samples"]

    run_id = f"{dataset}_{arch}_T{t_val}_Vth{vth_val}"
    print(f"\n{'='*65}\n▶ ÉVALUATION SPINNAKER : {run_id}\n{'='*65}")

    if not os.path.exists(weights_path):
        print(f"[!] Poids introuvables ({weights_path}), modèle ignoré.")
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw_input_size = int(meta["input_neurons"])
    time_step_ms = float(meta.get("time_step_ms", 1.0))
    sample_duration_ms = float(t_val * time_step_ms)
    window_ms = sample_duration_ms + REST_TIME_MS
    num_samples = len(sample_files)
    total_sim_time = num_samples * window_ms

    effective_input_size = min(raw_input_size, MAX_INPUT_NEURONS)

    spike_times, labels = load_multiplexed_spikes(
        sample_files, effective_input_size, raw_input_size, sample_duration_ms
    )
    state_dict = torch.load(weights_path, map_location="cpu")

    layer_sizes, synapse_lists, total_synapses = extract_topology_and_synapses(
        state_dict, effective_input_size
    )
    total_neurons = sum(layer_sizes)
    print(f"[*] Topologie : {layer_sizes} ({total_neurons} neurones | {total_synapses} synapses)")

    start_wall = time.time()
    sim.setup(timestep=DT, time_scale_factor=TIME_SCALE_FACTOR)
    sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 64)
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)

    try:
        populations = [
            sim.Population(
                layer_sizes[0],
                sim.SpikeSourceArray(spike_times=spike_times),
                label="InputLayer",
            )
        ]

        for idx, size in enumerate(layer_sizes[1:]):
            is_output = (idx == len(layer_sizes[1:]) - 1)
            params = BASE_LIF_PARAMS.copy()
            params["v_thresh"] = 10000.0 if is_output else float(vth_val)

            pop = sim.Population(
                size,
                sim.IF_curr_exp(**params),
                label="OutputLayer" if is_output else f"Hidden_{idx}",
            )
            if is_output:
                pop.record(["v"])
            populations.append(pop)

        for idx, (exc_conns, inh_conns) in enumerate(synapse_lists):
            if exc_conns:
                sim.Projection(
                    populations[idx],
                    populations[idx + 1],
                    sim.FromListConnector(exc_conns),
                    receptor_type="excitatory",
                )
            if inh_conns:
                sim.Projection(
                    populations[idx],
                    populations[idx + 1],
                    sim.FromListConnector(inh_conns),
                    receptor_type="inhibitory",
                )

        print(f"[*] Simulation matérielle ({total_sim_time} ms pour {num_samples} échantillons)...")
        sim.run(total_sim_time)

        v_data = populations[-1].get_data("v").segments[0].filter(name="v")[0].magnitude
        status = "SUCCESS"
        error_msg = ""

    except Exception as e:
        status = "HARDWARE_ERROR"
        error_msg = str(e).replace("\n", " ")[:150]
        v_data = None
        print(f"[!] Erreur matérielle sur {run_id} : {error_msg}")

    finally:
        sim.end()

    wall_time = round(time.time() - start_wall, 2)

    if status == "SUCCESS" and v_data is not None:
        predictions = []
        voltages_per_sample = []

        for i in range(num_samples):
            start_idx = int((i * window_ms) / DT)
            end_idx = int((i * window_ms + sample_duration_ms) / DT)
            sample_v = v_data[start_idx:end_idx]

            if len(sample_v) > 0:
                delta_v = sample_v[-1] - sample_v[0]
                pred = int(np.argmax(delta_v))
                voltages_per_sample.append([round(float(x), 4) for x in delta_v])
            else:
                pred = -1
                voltages_per_sample.append([])
            predictions.append(pred)

        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        acc = round((correct / num_samples) * 100.0, 2)
        archive_spinnaker_report(dataset, run_id)

        detail_path = os.path.join(PREDICTIONS_DIR, f"{run_id}.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "dataset": dataset,
                "architecture": arch,
                "T": t_val,
                "Vth": vth_val,
                "total_neurons": total_neurons,
                "total_synapses": total_synapses,
                "accuracy_pct": acc,
                "ground_truth": labels,
                "predictions": predictions,
                "voltages_per_sample": voltages_per_sample,
            }, f, indent=4)

        print(f"SUCCÈS : Accuracy = {acc}% ({correct}/{num_samples}) en {wall_time}s")
    else:
        acc = 0.0
        correct = 0

    with open(SUMMARY_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id, dataset, arch, t_val, vth_val,
            status, acc, correct, num_samples,
            total_sim_time, wall_time, total_neurons, total_synapses, error_msg
        ])

    return status == "SUCCESS"

def main():
    parser = argparse.ArgumentParser(description="Évaluation automatisée sur SpiNNaker")
    parser.add_argument("--dataset", type=str, default=None, help="Filtrer sur un seul dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de modèles à tester (ex: 1 pour vérifier)")
    args = parser.parse_args()

    init_directories()
    if not os.path.exists(PLAN_PATH):
        raise FileNotFoundError(f"Plan introuvable : {PLAN_PATH}. Lance d'abord prepare_spinnaker_suite.py")

    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        test_plan = json.load(f)

    completed_runs = get_completed_runs()
    executed_count = 0

    for entry in test_plan:
        if args.dataset and entry["dataset"] != args.dataset:
            continue

        run_id = f"{entry['dataset']}_{entry['architecture']}_T{entry['T']}_Vth{entry['Vth']}"
        if run_id in completed_runs:
            print(f"[-] Déjà validé dans le CSV, ignoré : {run_id}")
            continue

        run_single_evaluation(entry)
        executed_count += 1

        if args.limit and executed_count >= args.limit:
            print(f"\n[*] Limite de test atteinte ({args.limit} modèle(s)). Arrêt propre.")
            break

if __name__ == "__main__":
    main()