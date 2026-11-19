import argparse
import csv
import os
import re
from pathlib import Path

RESULTS_DIR = Path("results_spinnaker")
THEORETICAL_CSV = Path("theoretical_energy_results.csv")

ENERGY_COLUMNS = [
    "run_id",
    "dataset",
    "architecture",
    "T",
    "Vth",
    "n_samples",
    "accuracy_spinnaker",
    "reliable",
    "time_scale_factor",
    "energy_per_inference_j",
    "exec_time_s",
    "exec_energy_cores_j",
    "exec_energy_boards_j",
    "exec_energy_j",
    "total_energy_j",
    "n_boards",
    "n_chips",
    "n_active_chips",
    "n_cores",
    "n_active_cores",
    "report_file",
]

NUMBER = r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
REPORT_FIELDS = {
    "n_boards": (int, r"Simulation used (\d+) boards"),
    "n_chips": (int, r"made up of (\d+) chips"),
    "n_cores": (int, r"Simulation cores used: (\d+) in total"),
    "n_active_cores": (int, r"in total, (\d+) actively"),
    "n_active_chips": (int, r"actively on (\d+) chips"),
    "exec_time_s": (float, rf"Simulation execution time: {NUMBER} seconds"),
    "exec_energy_j": (float, rf"Simulation execution energy: {NUMBER} Joules"),
    "exec_energy_cores_j": (
        float,
        rf"\(active chips and cores only\): {NUMBER} Joules",
    ),
    "exec_energy_boards_j": (float, rf"\(ignoring frame power\): {NUMBER} Joules"),
    "total_energy_j": (float, rf"Total energy: {NUMBER} Joules"),
}


def parse_report(path):
    """Extrait les compteurs et énergies d'un rapport d'énergie sPyNNaker."""
    text = Path(path).read_text(encoding="utf-8")
    values = {}
    for name, (cast, pattern) in REPORT_FIELDS.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"{path} : champ « {name} » introuvable.")
        values[name] = cast(match.group(1))
    return values


def find_report(run_dir):
    """Rapport d'énergie le plus récent archivé pour un run (None s'il n'y en a pas)."""
    reports = list(Path(run_dir).rglob("energy_report*.rpt"))
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def successful_runs(summary_path):
    """Dernière ligne SUCCESS de chaque run dans summary_benchmark.csv."""
    if not summary_path.is_file():
        return {}
    runs = {}
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] == "SUCCESS":
                runs[row["run_id"]] = row
    return runs


def collect(results_dir):
    """Une ligne d'énergie par run réussi dont le rapport a été archivé."""
    results_dir = Path(results_dir)
    rows = []
    for run_id, run in successful_runs(results_dir / "summary_benchmark.csv").items():
        report = find_report(results_dir / "hardware_reports" / run["dataset"] / run_id)
        if report is None:
            print(f"[!] Pas de rapport d'énergie pour {run_id}.")
            continue
        values = parse_report(report)
        n_samples = int(run["total_samples"])
        rows.append(
            {
                "run_id": run_id,
                "dataset": run["dataset"],
                "architecture": run["architecture"],
                "T": run["T"],
                "Vth": run["Vth"],
                "n_samples": n_samples,
                "accuracy_spinnaker": run["accuracy_pct"],
                "reliable": run["reliable"],
                "time_scale_factor": run["time_scale_factor"],
                "energy_per_inference_j": values["exec_energy_cores_j"] / n_samples,
                **values,
                "report_file": report.as_posix(),
            }
        )
    return rows


def write_csv(path, columns, rows):
    """Écriture atomique : le CSV reste valide même si le script est interrompu."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def collect_to_csv(results_dir):
    """Écrit `<results-dir>/spinnaker_energy.csv` et renvoie son chemin et ses lignes."""
    results_dir = Path(results_dir)
    rows = collect(results_dir)
    output = results_dir / "spinnaker_energy.csv"
    write_csv(output, ENERGY_COLUMNS, rows)
    return output, rows


def run_key(row):
    return row["dataset"], row["architecture"], int(row["T"]), float(row["Vth"])


def merge_with_theoretical(energy_rows, theoretical_path, output_path):
    """Remplit energie_reelle_spinnaker du tableau théorique avec l'énergie mesurée."""
    measured = {run_key(r): r["energy_per_inference_j"] for r in energy_rows}
    with open(theoretical_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    for row in rows:
        row["energie_reelle_spinnaker"] = measured.get(run_key(row), "NaN")
    write_csv(output_path, columns, rows)
    return sum(1 for row in rows if row["energie_reelle_spinnaker"] != "NaN")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Énergie mesurée par sPyNNaker pour chaque run déployé."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--theoretical",
        type=Path,
        default=THEORETICAL_CSV,
        help="Tableau produit par theoretical_energy_eval.py",
    )
    args = parser.parse_args(argv)

    output, rows = collect_to_csv(args.results_dir)
    print(f"{len(rows)} runs avec un rapport d'énergie : {output}")

    if args.theoretical.is_file():
        final_path = args.results_dir / "final_energy_results.csv"
        filled = merge_with_theoretical(rows, args.theoretical, final_path)
        print(f"Tableau final ({filled} énergies réelles renseignées) : {final_path}")
    else:
        print(f"[*] {args.theoretical} absent : tableau final non généré.")


if __name__ == "__main__":
    main()
