import os
import json
import glob

MODELS_DIR = "spinnaker_models"
DATA_DIR = "spinnaker_data"
OUTPUT_PLAN = "spinnaker_plan.json"

VALID_T = [4, 8, 16, 32]
VALID_VTH = [0.5, 1.0, 1.5]

def build_test_plan():
    test_plan = []
    
    if not os.path.exists(MODELS_DIR):
        print(f"[!] Erreur: Le dossier '{MODELS_DIR}' n'existe pas.")
        return
        
    datasets = sorted(os.listdir(MODELS_DIR))
    summary = {}
    
    for dataset in datasets:
        dataset_path = os.path.join(MODELS_DIR, dataset)
        if not os.path.isdir(dataset_path): 
            continue
            
        count = 0
        items = sorted(os.listdir(dataset_path))
        
        for item in items:
            item_path = os.path.join(dataset_path, item)
            
            if os.path.isfile(item_path) and item.endswith(".pth"):
                name_no_ext = item[:-4]
                parts = name_no_ext.split('_')
                if len(parts) < 3:
                    continue
                arch, t_str, vth_str = parts[0], parts[1], parts[2]
                weights_path = item_path

            elif os.path.isdir(item_path) and item.startswith("run_"):
                parts = item.split('_')
                if len(parts) < 4:
                    continue
                arch, t_str, vth_str = parts[1], parts[2], parts[3]
                weights_path = os.path.join(item_path, "model_best.pth")
                if not os.path.exists(weights_path):
                    continue
            else:
                continue
            
            try:
                t_val = int(t_str.replace("T", ""))
                vth_val = float(vth_str.replace("Vth", ""))
            except ValueError:
                continue

            if t_val not in VALID_T or vth_val not in VALID_VTH:
                continue
                
            target_data_folder = os.path.join(DATA_DIR, dataset, f"T_{t_val}")
            meta_path = os.path.join(target_data_folder, "metadata.json")
            
            if not os.path.exists(target_data_folder) or not os.path.exists(meta_path):
                print(f"[!] Données manquantes pour {dataset} à T={t_val}. Modèle ignoré.")
                continue
                
            json_samples = sorted(glob.glob(os.path.join(target_data_folder, "sample_*.json")))
            if not json_samples:
                print(f"[!] Aucun sample_XX.json trouvé dans {target_data_folder}.")
                continue
            
            test_plan.append({
                "dataset": dataset,
                "architecture": arch,
                "T": t_val,
                "Vth": vth_val,
                "weights_path": weights_path,
                "metadata_path": meta_path,
                "samples": json_samples
            })
            count += 1
            
        summary[dataset] = count
            
    with open(OUTPUT_PLAN, "w", encoding="utf-8") as f:
        json.dump(test_plan, f, indent=4)
        
    print("\n=== RÉCAPITULATIF PAR DATASET ===")
    for ds, nb in summary.items():
        print(f" - {ds:<18} : {nb} modèles")
    print("=================================")
    print(f"Plan généré avec succès dans '{OUTPUT_PLAN}' ({len(test_plan)} modèles au total) !")

if __name__ == "__main__":
    build_test_plan()