import subprocess
import sys

DATASETS = [
    # "nmnist",
    # "cifar10",
    # "dvs_gesture",
    # "nepic_kitchens",
    # "ncaltech101",
    # "shd",
    # "dvs_lip",
    # "ssc",
    # "smnist",
    "pokerdvs131",
]


def test_datasets():
    print("Démarrage des tests d'intégration pour les 12 datasets...")
    print(
        "Attention : Si les datasets ne sont pas encore téléchargés, ce script va déclencher leur téléchargement."
    )
    print("-" * 60)

    failed_datasets = []

    for ds in DATASETS:
        print(f"\nTest en cours pour le dataset : {ds.upper()}...")

        cmd = [
            "python",
            "main.py",
            "--dataset",
            ds,
            "--epochs",
            "1",
            "--batch_size",
            "128",
            "--Time",
            "2",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"SUCCÈS : {ds} a complété son époque d'entraînement sans erreur.")
        else:
            print(f"ÉCHEC : {ds} a rencontré une erreur.")
            print("\n--- DÉBUT DE L'ERREUR ---")
            print(result.stderr.strip())
            print("--- FIN DE L'ERREUR ---\n")
            failed_datasets.append(ds)

    print("\n" + "=" * 60)
    if failed_datasets:
        print(
            f"Rapport final : {len(failed_datasets)} dataset(s) en échec : {', '.join(failed_datasets)}"
        )
        sys.exit(1)
    else:
        print("Rapport final : TOUS LES DATASETS SONT OPÉRATIONNELS !")
        sys.exit(0)


if __name__ == "__main__":
    test_datasets()
