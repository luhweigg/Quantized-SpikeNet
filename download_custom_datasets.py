import os
import sys
import urllib.request
import zipfile

DATASETS = [
    {
        "name": "EDAT24",
        "url": "https://zenodo.org/records/10688518/files/EDAT24.zip",
        "type": "zip",
        "output_dir": "data/EDAT24",
    },
    {
        "name": "CeoliniGestures",
        "url": "https://zenodo.org/records/3663616/files/relax21_cropped_dvs_emg_spikes.pkl",
        "type": "file",
        "output_dir": "data/CeoliniGestures",
        "filename": "relax21_cropped_dvs_emg_spikes.pkl",
    },
    {
        "name": "STEMNIST",
        "url": "https://zenodo.org/records/19469535/files/STEMNIST%20Dataset.zip",
        "type": "zip",
        "output_dir": "data/STEMNIST",
    },
]


def show_progress(block_num, block_size, total_size):
    """Affiche une barre de progression simple dans la console."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(
            f"\rTéléchargement : {percent:.1f}% ({downloaded / (1024 * 1024):.1f} MB)"
        )
        sys.stdout.flush()


def download_and_extract(dataset):
    print(f"\n{'=' * 50}\nTraitement du dataset : {dataset['name']}\n{'=' * 50}")

    os.makedirs(dataset["output_dir"], exist_ok=True)

    if len(os.listdir(dataset["output_dir"])) > 0:
        print(
            f"[*] {dataset['name']} semble déjà être présent dans {dataset['output_dir']}. Ignoré."
        )
        return

    temp_file = os.path.join(dataset["output_dir"], "temp_download.tmp")

    try:
        print("[*] Téléchargement depuis Zenodo...")
        urllib.request.urlretrieve(dataset["url"], temp_file, reporthook=show_progress)
        print("\n[*] Téléchargement terminé.")

        if dataset["type"] == "zip":
            print("[*] Extraction de l'archive...")
            with zipfile.ZipFile(temp_file, "r") as zip_ref:
                zip_ref.extractall(dataset["output_dir"])
            os.remove(temp_file)
            print("[*] Extraction terminée et archive supprimée.")

        elif dataset["type"] == "file":
            final_path = os.path.join(dataset["output_dir"], dataset["filename"])
            os.rename(temp_file, final_path)
            print(f"[*] Fichier sauvegardé sous : {final_path}")

    except Exception as e:
        print(f"\n[!] Erreur lors du traitement de {dataset['name']} : {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for ds in DATASETS:
        download_and_extract(ds)
    print("\n\nOpération terminée avec succès !")
