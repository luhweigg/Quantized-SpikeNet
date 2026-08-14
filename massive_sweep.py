import subprocess
import os
import json
import time

DATASETS = [
    "cifar10_dvs",
    "dvs_gesture",
    "dvs_lip",
    "ncaltech101",
    "nepic_kitchens",
    "nmnist",
    "shd",
    "smnist",
    "ssc",
    "edat24",
    "ceolini_gestures",
    "stemnist",
]

TIME_STEPS = [4, 8, 16, 32]
THRESHOLDS = [0.5, 1.0, 1.5]
EPOCHS = 20

TARGET_BATCH_SIZE = 64

SAFE_PHYSICAL_BATCH = {
    "nmnist": 64,
    "smnist": 64,
    "shd": 64,
    "ssc": 64,
    "edat24": 64,
    "ceolini_gestures": 64,
    "stemnist": 64,
    "cifar10_dvs": 32,
    "dvs_gesture": 16,
    "dvs_lip": 16,
    "ncaltech101": 16,
    "nepic_kitchens": 8,
}


def get_valid_architectures(dataset: str) -> list:
    config_path = os.path.join("configs", f"{dataset}.json")
    if not os.path.exists(config_path):
        return []
    with open(config_path, "r") as f:
        config = json.load(f)
    return list(config["architectures"].keys())


def main():
    experiments = []
    for dataset in DATASETS:
        valid_archs = get_valid_architectures(dataset)
        for arch in valid_archs:
            for t in TIME_STEPS:
                for vth in THRESHOLDS:
                    experiments.append((dataset, arch, t, vth))

    total_exp = len(experiments)
    print(f"Démarrage de la Grid Search : {total_exp} entraînements préparés.")

    for i, (dataset, arch, t, v_th) in enumerate(experiments, 1):
        physical_batch = SAFE_PHYSICAL_BATCH[dataset]

        if arch in ["SpikingVGG8", "SpikingResNet18", "SpikingResNet34"] and t >= 16:
            physical_batch = max(2, physical_batch // 2)

        accum_steps = max(1, TARGET_BATCH_SIZE // physical_batch)

        print(
            f"\n[{i}/{total_exp}] RUN: {dataset.upper()} | {arch} | T={t} | Vth={v_th}"
        )
        print(
            f"-> Physical Batch: {physical_batch} | Accumulation: {accum_steps} (Effective: {physical_batch * accum_steps})"
        )

        cmd = [
            "python",
            "main.py",
            "--dataset",
            dataset,
            "--architecture",
            arch,
            "--Time",
            str(t),
            "--v_threshold",
            str(v_th),
            "--epochs",
            str(EPOCHS),
            "--batch_size",
            str(physical_batch),
            "--accumulation_steps",
            str(accum_steps),
            "--use_wandb",
        ]

        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            elapsed = (time.time() - start_time) / 60
            print(f"Succès ({elapsed:.1f} min)")
        except subprocess.CalledProcessError:
            print(f"Échec pour {dataset} - {arch}. Passage au suivant.")
            with open("sweep_errors.log", "a") as f:
                f.write(f"Failed: {dataset} | {arch} | T={t} | Vth={v_th}\n")


if __name__ == "__main__":
    main()
