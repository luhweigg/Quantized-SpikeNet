import os
import shutil

ACTIVE_DATASETS = [
    "ceolini_gestures", "dvs_lip", "dvs_gesture", 
    "cifar10_dvs"
    "edat24", "ncaltech101", "nepic_kitchens", "nmnist", 
    "shd", "smnist", "ssc", "stemnist"
]

SOURCE_DIR = "saved_models"
OUTPUT_DIR = "spinnaker_models"

def collect_and_rename_models():
    if not os.path.exists(SOURCE_DIR):
        print(f"[!] Le dossier {SOURCE_DIR} n'existe pas.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_copied = 0

    found_datasets = sorted(os.listdir(SOURCE_DIR))
    
    for dataset in found_datasets:
        dataset_path = os.path.join(SOURCE_DIR, dataset)
        
        if not os.path.isdir(dataset_path):
            continue
            
        if dataset not in ACTIVE_DATASETS:
            print(f"[-] Ignoré (Hors de la liste active) : {dataset}")
            continue

        print(f"\n[*] Traitement du dataset : {dataset.upper()}")
        
        dataset_out_dir = os.path.join(OUTPUT_DIR, dataset)
        os.makedirs(dataset_out_dir, exist_ok=True)

        runs = sorted(os.listdir(dataset_path))
        for run_folder in runs:
            run_path = os.path.join(dataset_path, run_folder)
            
            if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
                continue

            parts = run_folder.split('_')
            if len(parts) < 4:
                continue
                
            arch = parts[1]
            t_str = parts[2]
            vth_str = parts[3]
            
            best_model_path = os.path.join(run_path, "model_best.pth")
            
            if os.path.exists(best_model_path):
                new_filename = f"{arch}_{t_str}_{vth_str}.pth"
                target_path = os.path.join(dataset_out_dir, new_filename)
                
                shutil.copy2(best_model_path, target_path)
                print(f"    -> Copié : {new_filename}")
                total_copied += 1
            else:
                print(f"[!] Fichier model_best.pth introuvable dans {run_folder}")

    print(f"\nOpération terminée. {total_copied} modèles optimaux ont été copiés et triés dans '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    collect_and_rename_models()