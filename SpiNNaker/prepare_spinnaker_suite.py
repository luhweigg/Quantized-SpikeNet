import glob
import json
import os
import shutil

ACTIVE_DATASETS = [
    "ceolini_gestures",
    "cifar10_dvs",
    "dvs_gesture",
    "dvs_lip",
    "edat24",
    "ncaltech101",
    "nepic_kitchens",
    "nmnist",
    "shd",
    "smnist",
    "ssc",
    "stemnist",
]

VALID_T = [4, 8, 16, 32]
VALID_VTH = [0.5, 1.0, 1.5]

SAVED_MODELS_DIR = "saved_models"
READY_MODELS_DIR = "spinnaker_models"
DATA_DIR = "spinnaker_data"
OUTPUT_PLAN = "spinnaker_plan.json"


def parse_run_folder(run_folder):
    """`run_<arch>_T<t>_Vth<v>_<date>_<heure>` -> (arch, t, vth), ou None."""
    parts = run_folder.split("_")
    if len(parts) < 4 or parts[0] != "run":
        return None
    try:
        t_val = int(parts[2].removeprefix("T"))
        vth_val = float(parts[3].removeprefix("Vth"))
    except ValueError:
        return None
    if t_val not in VALID_T or vth_val not in VALID_VTH:
        return None
    return parts[1], t_val, vth_val


def build_clean_suite():
    if os.path.exists(READY_MODELS_DIR):
        shutil.rmtree(READY_MODELS_DIR)
    os.makedirs(READY_MODELS_DIR)

    test_plan = []
    summary_counts = {}

    for dataset in ACTIVE_DATASETS:
        dataset_path = os.path.join(SAVED_MODELS_DIR, dataset)
        if not os.path.isdir(dataset_path):
            print(f"[!] Dossier introuvable : {dataset_path}")
            continue

        selected_runs = {}
        for run_folder in sorted(os.listdir(dataset_path)):
            key = parse_run_folder(run_folder)
            best_pth = os.path.join(dataset_path, run_folder, "model_best.pth")
            if key is not None and os.path.isfile(best_pth):
                selected_runs[key] = best_pth

        dataset_out_dir = os.path.join(READY_MODELS_DIR, dataset)
        os.makedirs(dataset_out_dir, exist_ok=True)
        count = 0

        for (arch, t_val, vth_val), src_pth in sorted(selected_runs.items()):
            data_folder = os.path.join(DATA_DIR, dataset, f"T_{t_val}")
            meta_path = os.path.join(data_folder, "metadata.json")
            samples = sorted(glob.glob(os.path.join(data_folder, "sample_*.json")))
            if not os.path.exists(meta_path) or not samples:
                print(f"[!] Données SpiNNaker manquantes pour {dataset} T={t_val}")
                continue

            dst_pth = os.path.join(dataset_out_dir, f"{arch}_T{t_val}_Vth{vth_val}.pth")
            shutil.copy2(src_pth, dst_pth)
            test_plan.append(
                {
                    "dataset": dataset,
                    "architecture": arch,
                    "T": t_val,
                    "Vth": vth_val,
                    "weights_path": dst_pth,
                    "metadata_path": meta_path,
                    "samples": samples,
                }
            )
            count += 1

        summary_counts[dataset] = count

    with open(OUTPUT_PLAN, "w", encoding="utf-8") as f:
        json.dump(test_plan, f, indent=4)

    print("\n" + "=" * 50)
    print("RÉCAPITULATIF DES MODÈLES PRÊTS POUR SPINNAKER")
    print("=" * 50)
    for ds, cnt in summary_counts.items():
        print(f" - {ds:<18} : {cnt} modèles")
    print("=" * 50)
    print(f"Total : {len(test_plan)} modèles enregistrés dans '{OUTPUT_PLAN}'")


if __name__ == "__main__":
    build_clean_suite()
