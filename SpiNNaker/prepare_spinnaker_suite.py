import glob
import json
import os
import shutil

ACTIVE_DATASETS = [
    "ceolini_gestures", "cifar10_dvs", "dvs_gesture", "dvs_lip",
    "edat24", "ncaltech101", "nepic_kitchens", "nmnist",
    "shd", "smnist", "ssc", "stemnist"
]

VALID_T = [4, 8, 16, 32]
VALID_VTH = [0.5, 1.0, 1.5]

SAVED_MODELS_DIR = "spinnaker_models"
READY_MODELS_DIR = "spinnaker_models"
DATA_DIR = "spinnaker_data"
OUTPUT_PLAN = "spinnaker_plan.json"

def build_clean_suite():
    if os.path.exists(READY_MODELS_DIR):
        shutil.rmtree(READY_MODELS_DIR)
    os.makedirs(READY_MODELS_DIR, exist_ok=True)

    test_plan = []
    summary_counts = {}

    for dataset in ACTIVE_DATASETS:
        dataset_path = os.path.join(SAVED_MODELS_DIR, dataset)
        if not os.path.isdir(dataset_path):
            print(f"[!] Dossier introuvable : {dataset_path}")
            continue

        selected_runs = {}

        for run_folder in sorted(os.listdir(dataset_path)):
            run_path = os.path.join(dataset_path, run_folder)
            if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
                continue

            parts = run_folder.split("_")
            if len(parts) < 4:
                continue

            arch = parts[1]
            t_str = parts[2]
            vth_str = parts[3]

            try:
                t_val = int(t_str.replace("T", ""))
                vth_val = float(vth_str.replace("Vth", ""))
            except ValueError:
                continue

            if t_val not in VALID_T or vth_val not in VALID_VTH:
                continue

            best_pth = os.path.join(run_path, "model_best.pth")
            if not os.path.exists(best_pth):
                continue

            selected_runs[(arch, t_val, vth_val)] = best_pth

        dataset_out_dir = os.path.join(READY_MODELS_DIR, dataset)
        os.makedirs(dataset_out_dir, exist_ok=True)
        count = 0

        for (arch, t_val, vth_val), src_pth in sorted(selected_runs.items()):
            clean_name = f"{arch}_T{t_val}_Vth{vth_val}.pth"
            dst_pth = os.path.join(dataset_out_dir, clean_name)
            shutil.copy2(src_pth, dst_pth)

            target_data_folder = os.path.join(DATA_DIR, dataset, f"T_{t_val}")
            meta_path = os.path.join(target_data_folder, "metadata.json")
            samples = sorted(glob.glob(os.path.join(target_data_folder, "sample_*.json")))

            if not os.path.exists(meta_path) or not samples:
                print(f"[!] Données SpiNNaker manquantes pour {dataset} T={t_val}")
                continue

            test_plan.append({
                "dataset": dataset,
                "architecture": arch,
                "T": t_val,
                "Vth": vth_val,
                "weights_path": dst_pth,
                "metadata_path": meta_path,
                "samples": samples
            })
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