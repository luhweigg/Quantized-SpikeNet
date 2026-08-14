import json
import os
import sys
import torch
import random
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine.builder import DATA_LOADERS


def extract_and_serialize_samples(
    datasets: List[str], time_steps: int, dt: float, output_dir: str
) -> None:
    """
    Extrait un echantillon pour chaque dataset specifie, applique un jitter temporel pour eviter la congestion SpiNNaker, et le serialise en JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in datasets:
        if dataset_name not in DATA_LOADERS:
            continue

        _, test_loader = DATA_LOADERS[dataset_name](1, time_steps)
        events_tensor, targets_tensor = next(iter(test_loader))

        sample_events = events_tensor[:, 0, ...]
        target_label = targets_tensor[0].item()

        flattened_events = sample_events.view(time_steps, -1)
        num_neurons = flattened_events.shape[1]

        spike_times = [[] for _ in range(num_neurons)]

        for t in range(time_steps):
            active_indices = flattened_events[t].nonzero(as_tuple=True)[0]
            for idx in active_indices:
                jitter = random.uniform(0, dt * 0.9)
                spike_times[idx.item()].append(float(t * dt) + jitter)

        payload = {
            "dataset": dataset_name,
            "label": target_label,
            "num_neurons": num_neurons,
            "spike_times": spike_times,
        }

        output_file = os.path.join(output_dir, f"sample_{dataset_name}.json")
        with open(output_file, "w") as f:
            json.dump(payload, f)


if __name__ == "__main__":
    target_datasets = ["nmnist", "cifar10_dvs", "dvs_gesture", "nepic_kitchens"]
    extract_and_serialize_samples(target_datasets, 20, 1.0, "networks")
