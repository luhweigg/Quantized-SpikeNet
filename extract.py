import os
import json
import torch
from src.engine.builder import DATA_LOADERS

NUM_SAMPLES = 30         
TIME_STEPS_LIST = [4, 8, 16, 32]
TIME_STEP_MS = 1.0       

OUTPUT_DIR = "./spinnaker_data"

def extract_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset in DATA_LOADERS.keys():
        print(f"\n{'='*50}")
        print(f"EXTRACTION POUR : {dataset.upper()}")
        print(f"{'='*50}")

        for t in TIME_STEPS_LIST:
            try:
                _, test_loader = DATA_LOADERS[dataset](
                    batch_size=NUM_SAMPLES, 
                    time_steps=t
                )
            except Exception as e:
                print(f"[!] Impossible de charger {dataset} à T={t} : {e}")
                continue

            try:
                x_batch, y_batch = next(iter(test_loader))
            except Exception as e:
                print(f"[!] Erreur de lecture pour {dataset} à T={t} : {e}")
                continue

            T = x_batch.shape[0]
            B = x_batch.shape[1]
            spatial_shape = list(x_batch.shape[2:]) 
            input_neurons_count = x_batch[0, 0].numel()

            dataset_dir = os.path.join(OUTPUT_DIR, dataset, f"T_{t}")
            os.makedirs(dataset_dir, exist_ok=True)

            meta_info = {
                "time_steps": T,
                "spatial_shape": spatial_shape,
                "input_neurons": input_neurons_count,
                "time_step_ms": TIME_STEP_MS
            }
            with open(os.path.join(dataset_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta_info, f, indent=4)

            for i in range(min(B, NUM_SAMPLES)):
                spike_times = [[] for _ in range(input_neurons_count)]
                label = int(y_batch[i].item())

                for current_t in range(T):
                    frame = x_batch[current_t, i].flatten()
                    spiking_neurons = torch.nonzero(frame > 0).squeeze(-1)

                    for neuron_idx in spiking_neurons:
                        physical_time = float(current_t * TIME_STEP_MS)
                        spike_times[neuron_idx.item()].append(physical_time)

                file_path = os.path.join(dataset_dir, f"sample_{i:02d}.json")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump({"label": label, "spike_times": spike_times}, f)

            print(f"{dataset.upper()} | T={t} : {min(B, NUM_SAMPLES)} JSON générés ({spatial_shape}).")

if __name__ == "__main__":
    extract_and_save()