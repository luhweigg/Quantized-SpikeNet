import argparse
import csv
import json
import logging
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (Path.cwd(), SCRIPT_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from spikingjelly.activation_based import functional, neuron

ENERGY_PER_SOP_JOULES = 0.9e-12
PENDING = "NaN"

RESULT_COLUMNS = [
    "dataset",
    "architecture",
    "Vth",
    "T",
    "nombre_neurones",
    "total_synapses",
    "accuracy",
    "energie_theorique_sops",
    "energie_theorique_joules",
    "energie_reelle_spinnaker",
]

DETAIL_COLUMNS = [
    "run_id",
    "status",
    "error",
    *RESULT_COLUMNS,
    "n_samples",
    "n_correct",
    "input_neurons",
    "hidden_neurons",
    "output_neurons",
    "input_spikes_per_sample",
    "hidden_spikes_per_sample",
    "hidden_firing_rate",
    "sops_hidden_per_sample",
    "sops_hidden_std",
    "sops_input_per_sample",
    "sops_hidden_all_samples",
    "energy_per_sop_joules",
    "input_sops_included",
    "wall_time_s",
]

LAYER_COLUMNS = [
    "run_id",
    "layer",
    "kind",
    "module_type",
    "source",
    "neurons",
    "synapses",
    "events_per_sample",
    "firing_rate",
    "sops_per_sample",
]

SYNAPTIC_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
AVG_POOL_TYPES = (
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
)
SKIP_NAME_HINTS = ("shortcut", "skip", "downsample", "residual")

log = logging.getLogger("theoretical_energy")

@dataclass
class LayerStats:
    name: str
    kind: str
    module_type: str
    source: str = ""
    neurons: int = 0
    synapses: int = 0
    events: float = 0.0
    sops: float = 0.0


def per_sample_sum(x):
    """Somme d'un tenseur [T, B, ...] sur tout sauf le batch -> [B] en float64."""
    return x.sum(dim=0).flatten(1).sum(dim=1, dtype=torch.float64)


def synaptic_fanout(module, feature_shape):
    """
    Fan-out exact de chaque neurone présynaptique : nombre de synapses de poids non nul
    qu'il possède vers la couche `module`. Retourne un tenseur de forme `feature_shape`.
    """
    weight = module.weight.detach()
    if isinstance(module, nn.Linear):
        return (weight != 0).sum(dim=0).to(torch.float32).expand(feature_shape)

    if module.padding_mode != "zeros":
        raise NotImplementedError(f"padding_mode={module.padding_mode} non supporté")

    # Le gradient d'une convolution à poids binaires par rapport à une entrée de 1
    # donne, pour chaque position d'entrée, le nombre de synapses qui en partent.
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[weight.dim() - 2]
    mask = (weight != 0).to(torch.float32)
    with torch.enable_grad():
        probe = torch.ones(1, *feature_shape, device=weight.device, requires_grad=True)
        out = conv(
            probe,
            mask,
            None,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
        (fanout,) = torch.autograd.grad(out.sum(), probe)
    return fanout[0]


class SynapticActivityMonitor:
    """
    Hooks comptant, par couche et par échantillon, les spikes émis par les neurones
    SpikingJelly et les opérations synaptiques qu'ils déclenchent en aval.
    Suppose des modèles multi-step : tenseurs de forme [T, B, ...].
    """

    def __init__(self, model):
        self.layers = {}
        self._handles = []
        self._fanout_cache = {}
        self._warned = set()
        self._hidden_seen = False
        self._avg_scale = None
        self._batch = None
        self.readout_is_spiking = False

        for name, module in model.named_modules():
            if isinstance(module, neuron.BaseNode):
                hook = self._hook(self._on_spikes, name)
                self._handles.append(module.register_forward_hook(hook))
            elif isinstance(module, SYNAPTIC_TYPES):
                hook = self._hook(self._on_synaptic_input, name)
                self._handles.append(module.register_forward_pre_hook(hook))
            elif isinstance(module, AVG_POOL_TYPES):
                self._handles.append(module.register_forward_hook(self._on_avg_pool))
            elif self._is_identity_skip(name, module):
                hook = self._hook(self._on_identity_skip, name)
                self._handles.append(module.register_forward_pre_hook(hook))

    @staticmethod
    def _hook(callback, name):
        return lambda module, *args: callback(name, module, *args)

    @staticmethod
    def _is_identity_skip(name, module):
        is_identity = isinstance(module, nn.Identity) or (
            isinstance(module, nn.Sequential) and len(module) == 0
        )
        leaf_name = name.rsplit(".", 1)[-1].lower()
        return is_identity and any(hint in leaf_name for hint in SKIP_NAME_HINTS)

    def start_batch(self, batch_size):
        self._hidden_seen = False
        self._avg_scale = None
        self._batch = {
            key: torch.zeros(batch_size, dtype=torch.float64)
            for key in ("hidden_spikes", "input_events", "sops_hidden", "sops_input")
        }

    def end_batch(self):
        batch, self._batch = self._batch, None
        return batch

    def remove(self):
        for handle in self._handles:
            handle.remove()

    def _stats(self, name, kind, module):
        if name not in self.layers:
            self.layers[name] = LayerStats(name, kind, type(module).__name__)
        return self.layers[name]

    def _warn_once(self, name, message):
        if name not in self._warned:
            self._warned.add(name)
            log.warning("    [%s] %s", name, message)

    def _on_spikes(self, name, module, inputs, output):
        spikes = per_sample_sum(output.detach()).cpu()
        stats = self._stats(name, "spiking", module)
        stats.neurons = output.shape[2:].numel()
        stats.events += spikes.sum().item()
        self._batch["hidden_spikes"] += spikes
        self._hidden_seen = True
        self.readout_is_spiking = True

    def _on_avg_pool(self, module, inputs, output):
        self._avg_scale = inputs[0].shape[2:].numel() / output.shape[2:].numel()

    def _event_counts(self, name, x):
        """Nombre d'évènements portés par chaque élément de x, et multiplicité amont."""
        scale, self._avg_scale = self._avg_scale, None
        if scale is not None and scale != 1:
            counts = x * scale
            rounded = counts.round()
            if torch.allclose(counts, rounded, atol=1e-3):
                return rounded, scale
        elif bool((x == x.round()).all()) and bool((x >= 0).all()):
            return x, 1.0

        self._warn_once(
            name, "entrée non entière : 1 évènement compté par élément non nul."
        )
        return (x != 0).to(x.dtype), 1.0

    def _on_synaptic_input(self, name, module, inputs):
        x = inputs[0].detach()
        source = "hidden" if self._hidden_seen else "input"
        counts, multiplicity = self._event_counts(name, x)

        feature_shape = tuple(x.shape[2:])
        key = (name, feature_shape)
        if key not in self._fanout_cache:
            self._fanout_cache[key] = synaptic_fanout(module, feature_shape)
        fanout = self._fanout_cache[key]

        events = per_sample_sum(counts).cpu()
        sops = (counts.sum(dim=0) * fanout).flatten(1).sum(dim=1, dtype=torch.float64)
        sops = sops.cpu()

        stats = self._stats(name, "synaptic", module)
        stats.source = source
        stats.neurons = round(math.prod(feature_shape) * multiplicity)
        stats.synapses = round(fanout.sum(dtype=torch.float64).item() * multiplicity)
        stats.events += events.sum().item()
        stats.sops += sops.sum().item()

        if source == "hidden":
            self._batch["sops_hidden"] += sops
        else:
            self._batch["input_events"] += events
            self._batch["sops_input"] += sops
        self.readout_is_spiking = False

    def _on_identity_skip(self, name, module, inputs):
        if not self._hidden_seen:
            return
        spikes = per_sample_sum(inputs[0].detach()).cpu()
        stats = self._stats(name, "skip", module)
        stats.source = "hidden"
        stats.neurons = stats.synapses = inputs[0].shape[2:].numel()
        stats.events += spikes.sum().item()
        stats.sops += spikes.sum().item()
        self._batch["sops_hidden"] += spikes



def resolve_model_factory():
    try:
        from models import get_model
    except ImportError:
        from src.models import get_model
    return get_model


def load_state_dict(weights_path):
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Checkpoints d'entraînement complets (optimizer, état RNG...) : fichiers locaux.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    for key in ("state_dict", "model_state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    return {k.removeprefix("module."): v for k, v in checkpoint.items()}


def resolve_path(path, plan_dir):
    """Les chemins du plan sont relatifs au dossier d'où il a été généré."""
    candidate = Path(path)
    if not candidate.exists() and not candidate.is_absolute():
        candidate = plan_dir / candidate
    return candidate


def hardware_input_ids(num_inputs, max_input_neurons):
    """Neurones d'entrée conservés par le sous-échantillonnage spatial de SpiNNaker."""
    effective = min(num_inputs, max_input_neurons)
    stride = max(1, num_inputs // effective)
    return {min(i * stride, num_inputs - 1) for i in range(effective)}


def load_batch(sample_paths, time_steps, spatial_shape, time_step_ms, args):
    """
    Reconstruit le tenseur d'entrée [T, B, *spatial_shape] à partir des `spike_times`
    JSON : chaque spike incrémente de 1 le pas de temps correspondant.
    """
    num_inputs = math.prod(spatial_shape)
    x = torch.zeros(time_steps, len(sample_paths), num_inputs)
    labels = []
    kept_ids = None
    if args.max_input_neurons is not None:
        kept_ids = hardware_input_ids(num_inputs, args.max_input_neurons)

    for batch_idx, path in enumerate(sample_paths):
        with open(path, encoding="utf-8") as f:
            sample = json.load(f)

        spike_times = sample["spike_times"]
        if len(spike_times) != num_inputs:
            raise ValueError(
                f"{path} : {len(spike_times)} neurones d'entrée, {num_inputs} attendus."
            )
        labels.append(int(sample["label"]))

        steps, neurons = [], []
        for neuron_idx, times in enumerate(spike_times):
            if not isinstance(times, list) or not times:
                continue
            if kept_ids is not None and neuron_idx not in kept_ids:
                continue
            for spike_time in times[: args.max_spikes_per_neuron]:
                step = round(spike_time / time_step_ms)
                if 0 <= step < time_steps:
                    steps.append(step)
                    neurons.append(neuron_idx)

        if steps:
            x[:, batch_idx].index_put_(
                (torch.tensor(steps), torch.tensor(neurons)),
                torch.tensor(1.0),
                accumulate=True,
            )

    x = x.view(time_steps, len(sample_paths), *spatial_shape)
    return x, torch.tensor(labels)



def run_id_of(entry):
    return f"{entry['dataset']}_{entry['architecture']}_T{entry['T']}_Vth{entry['Vth']}"


def base_row(entry):
    row = dict.fromkeys(DETAIL_COLUMNS, float("nan"))
    row.update(
        run_id=run_id_of(entry),
        status="PENDING",
        error="",
        dataset=entry["dataset"],
        architecture=entry["architecture"],
        Vth=entry["Vth"],
        T=entry["T"],
        energie_reelle_spinnaker=PENDING,
    )
    return row


@torch.no_grad()
def evaluate_run(entry, plan_dir, get_model, device, args):
    """Simule un modèle du plan et retourne (ligne de détails, lignes par couche)."""
    time_steps = int(entry["T"])
    weights_path = resolve_path(entry["weights_path"], plan_dir)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Poids introuvables : {weights_path}")

    with open(resolve_path(entry["metadata_path"], plan_dir), encoding="utf-8") as f:
        metadata = json.load(f)
    spatial_shape = [int(s) for s in metadata["spatial_shape"]]
    time_step_ms = float(metadata.get("time_step_ms", 1.0))
    sample_paths = [resolve_path(p, plan_dir) for p in entry["samples"]]
    if not sample_paths:
        raise ValueError("Aucun échantillon listé dans le plan.")

    model = get_model(
        entry["architecture"], entry["dataset"], v_threshold=float(entry["Vth"])
    )
    model.load_state_dict(load_state_dict(weights_path), strict=True)
    model.to(device).eval()
    functional.reset_net(model)

    monitor = SynapticActivityMonitor(model)
    totals = {}
    n_correct = 0
    output_neurons = 0

    try:
        for start in range(0, len(sample_paths), args.batch_size):
            batch_paths = sample_paths[start : start + args.batch_size]
            x, labels = load_batch(
                batch_paths, time_steps, spatial_shape, time_step_ms, args
            )

            monitor.start_batch(len(batch_paths))
            outputs = model(x.to(device))
            if outputs.dim() == 3:
                outputs = outputs.mean(dim=0)
            functional.reset_net(model)

            n_correct += (outputs.argmax(dim=1).cpu() == labels).sum().item()
            output_neurons = outputs.shape[1]
            for key, values in monitor.end_batch().items():
                totals.setdefault(key, []).append(values)
    finally:
        monitor.remove()

    totals = {key: torch.cat(values) for key, values in totals.items()}
    n_samples = len(sample_paths)
    layers = list(monitor.layers.values())

    hidden_neurons = sum(s.neurons for s in layers if s.kind == "spiking")
    if monitor.readout_is_spiking:
        output_neurons = 0
    sops = totals["sops_hidden"]
    if args.include_input_sops:
        sops = sops + totals["sops_input"]
    mean_sops = sops.mean().item()
    hidden_spikes = totals["hidden_spikes"].mean().item()

    row = base_row(entry)
    row.update(
        status="OK",
        nombre_neurones=hidden_neurons + output_neurons,
        total_synapses=sum(s.synapses for s in layers),
        accuracy=round(100.0 * n_correct / n_samples, 2),
        energie_theorique_sops=round(mean_sops, 3),
        energie_theorique_joules=mean_sops * args.energy_per_sop,
        n_samples=n_samples,
        n_correct=n_correct,
        input_neurons=math.prod(spatial_shape),
        hidden_neurons=hidden_neurons,
        output_neurons=output_neurons,
        input_spikes_per_sample=round(totals["input_events"].mean().item(), 3),
        hidden_spikes_per_sample=round(hidden_spikes, 3),
        hidden_firing_rate=hidden_spikes / max(1, hidden_neurons * time_steps),
        sops_hidden_per_sample=round(totals["sops_hidden"].mean().item(), 3),
        sops_hidden_std=round(totals["sops_hidden"].std().item(), 3)
        if n_samples > 1
        else 0.0,
        sops_input_per_sample=round(totals["sops_input"].mean().item(), 3),
        sops_hidden_all_samples=round(totals["sops_hidden"].sum().item()),
        energy_per_sop_joules=args.energy_per_sop,
        input_sops_included=args.include_input_sops,
    )

    layer_rows = [
        {
            "run_id": row["run_id"],
            "layer": s.name,
            "kind": s.kind,
            "module_type": s.module_type,
            "source": s.source,
            "neurons": s.neurons,
            "synapses": s.synapses,
            "events_per_sample": round(s.events / n_samples, 3),
            "firing_rate": s.events / (n_samples * max(1, s.neurons * time_steps)),
            "sops_per_sample": round(s.sops / n_samples, 3),
        }
        for s in layers
    ]
    return row, layer_rows


def format_cell(value):
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def write_csv(path, columns, rows):
    """Écriture atomique : le CSV reste valide même si le script est interrompu."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: format_cell(row.get(col, "")) for col in columns})
    os.replace(tmp_path, path)


def read_csv(path):
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Énergie théorique (SOPs) des modèles du plan SpiNNaker."
    )
    parser.add_argument("--plan", type=Path, default=Path("spinnaker_plan.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("theoretical_energy_results.csv")
    )
    parser.add_argument("--dataset", default=None, help="Filtrer sur un seul dataset")
    parser.add_argument(
        "--limit", type=int, default=None, help="Nombre maximal de modèles à évaluer"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ne pas réévaluer les runs déjà en statut OK dans le fichier de détails",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, help="cpu, cuda... (auto par défaut)")
    parser.add_argument(
        "--energy-per-sop",
        type=float,
        default=ENERGY_PER_SOP_JOULES,
        help="Énergie d'une opération synaptique en joules",
    )
    parser.add_argument(
        "--include-input-sops",
        action="store_true",
        help="Ajouter les SOPs de la couche d'entrée à energie_theorique_sops",
    )
    parser.add_argument(
        "--max-spikes-per-neuron",
        type=int,
        default=None,
        help="Tronquer les spikes d'entrée comme sur SpiNNaker (MAX_SPIKES_PER_NEURON)",
    )
    parser.add_argument(
        "--max-input-neurons",
        type=int,
        default=None,
        help="Ne garder que les neurones d'entrée déployés sur SpiNNaker "
        "(MAX_INPUT_NEURONS)",
    )
    return parser.parse_args(argv)


def setup_logging(log_path):
    """Logger dédié : le logger racine est déjà configuré par certaines dépendances."""
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    log.setLevel(logging.INFO)
    log.propagate = False
    log.handlers.clear()
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        log.addHandler(handler)


def main(argv=None):
    args = parse_args(argv)
    details_path = args.output.with_name(f"{args.output.stem}_details.csv")
    layers_path = args.output.with_name(f"{args.output.stem}_per_layer.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(args.output.with_suffix(".log"))

    if not args.plan.is_file():
        raise FileNotFoundError(
            f"Plan introuvable : {args.plan}. Lance d'abord prepare_spinnaker_suite.py"
        )
    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)
    plan_dir = args.plan.resolve().parent

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    get_model = resolve_model_factory()

    # Une ligne par run du plan : les runs en échec ou non évalués restent à NaN.
    rows = {run_id_of(entry): base_row(entry) for entry in plan}
    layer_rows = {}
    if args.resume:
        for row in read_csv(details_path):
            if row["run_id"] in rows and row["status"] == "OK":
                rows[row["run_id"]] = row
        for row in read_csv(layers_path):
            if rows.get(row["run_id"], {}).get("status") == "OK":
                layer_rows.setdefault(row["run_id"], []).append(row)

    todo = [
        entry
        for entry in plan
        if (args.dataset is None or entry["dataset"] == args.dataset)
        and rows[run_id_of(entry)]["status"] != "OK"
    ]
    if args.limit is not None:
        todo = todo[: args.limit]

    log.info(
        "%d modèles dans le plan | %d à évaluer | device=%s | %.2e J/SOP",
        len(plan),
        len(todo),
        device,
        args.energy_per_sop,
    )

    def save():
        write_csv(args.output, RESULT_COLUMNS, rows.values())
        write_csv(details_path, DETAIL_COLUMNS, rows.values())
        write_csv(
            layers_path,
            LAYER_COLUMNS,
            [layer for run_layers in layer_rows.values() for layer in run_layers],
        )

    save()
    failures = []
    for index, entry in enumerate(todo, start=1):
        run_id = run_id_of(entry)
        start_time = time.time()
        try:
            row, layer_rows[run_id] = evaluate_run(
                entry, plan_dir, get_model, device, args
            )
            row["wall_time_s"] = round(time.time() - start_time, 2)
            log.info(
                "Modèle %d/%d évalué : %s | acc=%.2f%% | %.4g SOPs | %.4g J | %.1fs",
                index,
                len(todo),
                run_id,
                row["accuracy"],
                row["energie_theorique_sops"],
                row["energie_theorique_joules"],
                row["wall_time_s"],
            )
        except Exception as exc:
            row = base_row(entry)
            row.update(
                status="MISSING_FILE"
                if isinstance(exc, FileNotFoundError)
                else "ERROR",
                error=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300],
                wall_time_s=round(time.time() - start_time, 2),
            )
            layer_rows.pop(run_id, None)
            failures.append(run_id)
            log.error(
                "Modèle %d/%d en échec : %s | %s",
                index,
                len(todo),
                run_id,
                row["error"],
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows[run_id] = row
        save()

    n_ok = sum(1 for row in rows.values() if row["status"] == "OK")
    log.info("Terminé : %d/%d runs évalués avec succès.", n_ok, len(rows))
    if failures:
        log.warning("%d run(s) en échec : %s", len(failures), ", ".join(failures))
    log.info("Tableau de synthèse : %s", args.output)
    log.info("Détails par run     : %s", details_path)
    log.info("Détails par couche  : %s", layers_path)


if __name__ == "__main__":
    main()
