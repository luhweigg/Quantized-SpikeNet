import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theoretical_energy_eval as tee
from spikingjelly.activation_based import functional, neuron

try:
    import spinnaker_energy
except ImportError:
    spinnaker_energy = None

RESULTS_DIR = Path("results_spinnaker_faithful")
SUMMARY_COLUMNS = [
    "run_id",
    "dataset",
    "architecture",
    "T",
    "Vth",
    "status",
    "accuracy_pct",
    "correct_samples",
    "total_samples",
    "sim_time_ms",
    "wall_time_s",
    "total_neurons",
    "total_synapses",
    "error_msg",
    "accuracy_pytorch_pct",
    "accuracy_emulated_pct",
    "accuracy_fixed_point_pct",
]

DT = 1.0
TIME_SCALE_FACTOR = 10  # voir spinnaker_master_eval.py : ne pas dépasser 200
TAU_M = 1.0 / math.log(
    2.0
)  # exp(-DT / TAU_M) = 1/2, comme le LIF tau=2 de SpikingJelly
REST_STEPS = 16  # silence entre deux échantillons : V retombe à 2^-16 de sa valeur
POOL_THRESHOLD = 0.5  # un neurone « OU » décharge dès qu'il reçoit un spike de poids 1
READOUT_THRESHOLD = 30000.0
INPUT_NEURONS_PER_CORE = 256
NEURONS_PER_CORE = 128
WEIGHT_MARGIN = 1.25  # marge sur la somme de poids maximale observée en émulation


class Unsupported(Exception):
    """Le modèle contient une structure que la conversion ne sait pas déplier."""


class TooLarge(Exception):
    """Le réseau déplié ne tient pas dans le budget de la carte."""



def leaf_modules(module):
    for child in module.children():
        if isinstance(child, nn.Sequential):
            yield from leaf_modules(child)
        else:
            yield child


def pair(value, is_1d, fill=1):
    """Paramètre de convolution / pooling ramené en 2D (les couches 1D ont H = 1)."""
    if isinstance(value, (tuple, list)):
        value = tuple(value)
        return (fill, value[0]) if is_1d and len(value) == 1 else value
    return (fill, value) if is_1d else (value, value)


def conv_connections(module, in_shape, max_synapses):
    """Synapses (pré, post, poids) d'une convolution dépliée, et forme de sortie."""
    if module.groups != 1 or module.padding_mode != "zeros":
        raise Unsupported("convolution groupée ou padding non nul")
    is_1d = isinstance(module, nn.Conv1d)
    kernel = pair(module.kernel_size, is_1d)
    stride = pair(module.stride, is_1d)
    dilation = pair(module.dilation, is_1d)
    padding = pair(module.padding, is_1d, fill=0)

    channels, height, width = in_shape
    out_hw = [
        (size + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1
        for i, size in enumerate((height, width))
    ]
    n_out_pos = out_hw[0] * out_hw[1]
    fan_in = channels * kernel[0] * kernel[1]
    if module.out_channels * fan_in * n_out_pos > max_synapses:
        raise TooLarge("trop de synapses dans une convolution")

    # Indice (+1) du neurone d'entrée vu par chaque élément du noyau ; 0 = padding.
    index = torch.arange(1, channels * height * width + 1, dtype=torch.float64)
    index = F.unfold(
        index.reshape(1, channels, height, width), kernel, dilation, padding, stride
    )[0].long()  # [fan_in, n_out_pos]
    weight = module.weight.detach().double().reshape(module.out_channels, fan_in)

    pre = index.unsqueeze(0).expand(module.out_channels, fan_in, n_out_pos)
    post = (
        torch.arange(module.out_channels).reshape(-1, 1, 1) * n_out_pos
        + torch.arange(n_out_pos).reshape(1, 1, -1)
    ).expand_as(pre)
    values = weight.unsqueeze(2).expand_as(pre)
    keep = (pre > 0) & (values != 0)
    conns = np.stack(
        [
            (pre[keep] - 1).numpy().astype(np.float64),
            post[keep].numpy().astype(np.float64),
            values[keep].numpy(),
        ],
        axis=1,
    )
    bias = torch.zeros(module.out_channels, dtype=torch.float64)
    if module.bias is not None:
        bias = module.bias.detach().double()
    out_shape = (module.out_channels, out_hw[0], out_hw[1])
    return conns, bias.repeat_interleave(n_out_pos).numpy(), out_shape


def linear_connections(module, in_shape, averaged, max_synapses):
    """Synapses d'une couche Linear, AvgPool global éventuel replié dans les poids."""
    n_pre = int(np.prod(in_shape))
    weight = module.weight.detach().double()
    if averaged:
        spatial = in_shape[1] * in_shape[2]
        weight = weight.repeat_interleave(spatial, dim=1) / spatial
    if weight.shape[1] != n_pre:
        raise Unsupported(f"Linear attend {weight.shape[1]} entrées, {n_pre} fournies")
    if weight.numel() > max_synapses:
        raise TooLarge("trop de synapses dans une couche Linear")

    post, pre = torch.nonzero(weight, as_tuple=True)
    conns = np.stack(
        [
            pre.numpy().astype(np.float64),
            post.numpy().astype(np.float64),
            weight[post, pre].numpy(),
        ],
        axis=1,
    )
    bias = torch.zeros(weight.shape[0], dtype=torch.float64)
    if module.bias is not None:
        bias = module.bias.detach().double()
    return conns, bias.numpy(), (weight.shape[0], 1, 1), weight.numpy()


def pool_connections(module, in_shape):
    """MaxPool sur des spikes = OU logique : chaque neurone vers sa case de pooling."""
    is_1d = isinstance(module, nn.MaxPool1d)
    kernel = pair(module.kernel_size, is_1d)
    stride = pair(module.stride or module.kernel_size, is_1d)
    if kernel != stride or any(p != 0 for p in pair(module.padding, is_1d, fill=0)):
        raise Unsupported("MaxPool avec recouvrement ou padding")

    channels, height, width = in_shape
    out_h, out_w = height // kernel[0], width // kernel[1]
    c, h, w = np.meshgrid(
        np.arange(channels), np.arange(height), np.arange(width), indexing="ij"
    )
    keep = (h < out_h * kernel[0]) & (w < out_w * kernel[1])
    pre = (c * height + h) * width + w
    post = (c * out_h + h // kernel[0]) * out_w + w // kernel[1]
    conns = np.stack([pre[keep], post[keep], np.ones(int(keep.sum()))], axis=1).astype(
        np.float64
    )
    return conns, (channels, out_h, out_w)


def convert_model(model, input_shape, max_synapses=30_000_000):
    """
    Déplie un modèle séquentiel en populations et projections.
    Retourne un dict : populations (la 0 est l'entrée), projections, readout.
    """
    shape = tuple(int(s) for s in input_shape)
    shape = (int(np.prod(shape)), 1, 1) if len(shape) == 1 else shape
    shape = (shape[0], 1, shape[1]) if len(shape) == 2 else shape

    pops = [
        {"label": "Input", "size": int(np.prod(shape)), "depth": 0, "kind": "input"}
    ]
    projections = []
    source = 0
    pending = None  # (conns, bias, out_shape, dense) en attente du neurone suivant
    averaged = False

    def add_population(kind, size, threshold, conns, bias=None):
        pops.append(
            {
                "label": f"{kind}_{len(pops)}",
                "size": int(size),
                "depth": pops[source]["depth"] + 1,
                "kind": kind,
                "v_thresh": float(threshold),
                "bias": bias,
            }
        )
        projections.append({"pre": source, "post": len(pops) - 1, "conns": conns})
        return len(pops) - 1

    for module in leaf_modules(model.network):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            if pending is not None or averaged:
                raise Unsupported("deux couches synaptiques sans neurone entre elles")
            pending = conv_connections(module, shape, max_synapses) + (None,)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if pending is None:
                raise Unsupported("BatchNorm sans convolution avant")
            conns, bias, out_shape, _ = pending
            scale = module.weight.detach().double() / torch.sqrt(
                module.running_var.detach().double() + module.eps
            )
            shift = module.bias.detach().double() - module.running_mean.double() * scale
            per_channel = out_shape[1] * out_shape[2]
            scale = scale.repeat_interleave(per_channel).numpy()
            shift = shift.repeat_interleave(per_channel).numpy()
            conns[:, 2] *= scale[conns[:, 1].astype(np.int64)]
            pending = (conns, bias * scale + shift, out_shape, None)
        elif isinstance(module, nn.Linear):
            if pending is not None:
                raise Unsupported("deux couches synaptiques sans neurone entre elles")
            pending = linear_connections(module, shape, averaged, max_synapses)
            averaged = False
        elif isinstance(module, neuron.BaseNode):
            if pending is None:
                raise Unsupported("neurone sans couche synaptique avant")
            if getattr(module, "tau", 2.0) != 2.0 or module.v_reset != 0.0:
                raise Unsupported("neurone autre que LIF tau=2, v_reset=0")
            conns, bias, shape, _ = pending
            source = add_population(
                "LIF", np.prod(shape), module.v_threshold, conns, bias
            )
            pending = None
        elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d)):
            if pending is not None:
                raise Unsupported("MaxPool avant le neurone")
            conns, shape = pool_connections(module, shape)
            source = add_population("Pool", np.prod(shape), POOL_THRESHOLD, conns)
        elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
            output_size = module.output_size
            output_size = (
                output_size if isinstance(output_size, (tuple, list)) else [output_size]
            )
            if any(s != 1 for s in output_size) or pending is not None:
                raise Unsupported("AvgPool non global")
            averaged = True
        elif isinstance(module, (nn.Flatten, nn.Dropout, nn.Identity)) or type(
            module
        ).__name__ in ("Flatten", "Dropout"):
            continue
        else:
            raise Unsupported(f"module {type(module).__name__}")

    if pending is None or pending[3] is None:
        raise Unsupported("le modèle ne se termine pas par une couche Linear")
    conns, bias, out_shape, dense = pending
    readout = {"pre": source, "weight": dense, "bias": bias}
    add_population("Readout", out_shape[0], READOUT_THRESHOLD, conns)
    return {"populations": pops, "projections": projections, "readout": readout}


def network_size(spec):
    neurons = sum(p["size"] for p in spec["populations"])
    synapses = sum(len(p["conns"]) for p in spec["projections"])
    cores = sum(
        math.ceil(
            p["size"]
            / (INPUT_NEURONS_PER_CORE if p["kind"] == "input" else NEURONS_PER_CORE)
        )
        for p in spec["populations"]
    )
    return neurons, synapses, cores



def ring_buffer_shift(max_summed_weight):
    """
    Décalage du « ring buffer » de sPyNNaker pour une somme de poids maximale donnée.
    La carte stocke chaque poids sur 16 bits avec une résolution de 2^(shift - 15) et
    sature au-delà de 2^(shift + 1) : trop petit, les entrées saturent ; trop grand, les
    petits poids sont arrondis à zéro.
    """
    if max_summed_weight <= 0:
        return 0
    return max(0, math.ceil(math.log2(max_summed_weight * WEIGHT_MARGIN)))


def emulate(spec, x, shifts=None):
    """
    Rejoue le réseau converti avec la règle de mise à jour de SpiNNaker.
    x : [T, B, n_entrées] (comptes d'évènements).
    shifts : {population: (shift excitateur, shift inhibiteur)} pour reproduire aussi la
    quantification des poids et la saturation de la carte ; None = arithmétique exacte.
    Retourne les logits [B, classes] et, par population, la plus grande somme de poids
    excitatrice et inhibitrice reçue par un neurone en un pas de temps.
    """
    time_steps, batch = x.shape[0], x.shape[1]
    spikes = {0: x.reshape(time_steps, batch, -1).double()}
    peaks = {}
    for proj in spec["projections"]:
        post = proj["post"]
        pop = spec["populations"][post]
        if pop["kind"] == "Readout":
            continue
        conns = proj["conns"]
        bias = np.zeros(pop["size"])
        if pop["bias"] is not None:
            bias = np.asarray(pop["bias"], dtype=np.float64)

        currents = []
        peaks[post] = []
        for receptor, sign in enumerate((1.0, -1.0)):
            keep = conns[:, 2] * sign > 0
            weights = np.abs(conns[keep, 2])
            offset = np.clip(bias * sign, 0.0, None)
            limit = None
            if shifts is not None:
                resolution = 2.0 ** (shifts[post][receptor] - 15)
                weights = np.minimum(np.rint(weights / resolution), 65535) * resolution
                offset = np.minimum(np.rint(offset / resolution), 65535) * resolution
                limit = 65535 * resolution
            matrix = torch.sparse_coo_tensor(
                np.stack([conns[keep, 1], conns[keep, 0]]).astype(np.int64),
                weights,
                (pop["size"], spec["populations"][proj["pre"]]["size"]),
                dtype=torch.float64,
            ).coalesce()
            currents.append((matrix, torch.from_numpy(offset), limit))

        v = torch.zeros(batch, pop["size"], dtype=torch.float64)
        out = torch.zeros(time_steps, batch, pop["size"], dtype=torch.float64)
        peak = [0.0, 0.0]
        for t in range(time_steps):
            pre_spikes = spikes[proj["pre"]][t].T
            total = torch.zeros_like(v)
            for receptor, (matrix, offset, limit) in enumerate(currents):
                received = torch.sparse.mm(matrix, pre_spikes).T + offset
                peak[receptor] = max(peak[receptor], float(received.max()))
                if limit is not None:
                    received = received.clamp(max=limit)
                total += received if receptor == 0 else -received
            v = 0.5 * v + 0.5 * total
            fired = v >= pop["v_thresh"]
            out[t] = fired.double()
            v = torch.where(fired, torch.zeros_like(v), v)
        spikes[post] = out
        peaks[post] = peak

    logits = readout_logits(
        spec, spikes[spec["readout"]["pre"]].sum(dim=0).numpy(), time_steps
    )
    return logits, peaks


def readout_logits(spec, spike_counts, time_steps):
    """Logits = somme sur T de la sortie Linear, à partir des spikes cachés comptés."""
    readout = spec["readout"]
    return spike_counts @ readout["weight"].T + time_steps * readout["bias"]



def split_by_sign(conns, delay):
    """
    Listes excitatrice et inhibitrice (pré, post, |poids|, délai), triées par neurone
    source puis cible : sPyNNaker dimensionne ses lignes synaptiques en supposant cet
    ordre, et les surestime (SynapseRowTooBigException) sur une liste non triée.
    """
    result = []
    for receptor, keep in (
        ("excitatory", conns[:, 2] > 0),
        ("inhibitory", conns[:, 2] < 0),
    ):
        if keep.any():
            rows = conns[keep]
            rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
            rows = np.column_stack(
                [
                    rows[:, 0],
                    rows[:, 1],
                    np.abs(rows[:, 2]),
                    np.full(len(rows), float(delay)),
                ]
            )
            result.append((receptor, rows))
    return result


def delta_neuron_model(sim):
    """Neurone LIF à synapse instantanée ; son nom varie selon la version de sPyNNaker."""
    for owner, name in (
        (sim, "IF_curr_delta"),
        (sim.extra_models, "IFCurrDelta"),
        (sim.extra_models, "IFCurDelta"),
    ):
        if hasattr(owner, name):
            return getattr(owner, name)
    raise RuntimeError(
        "Modèle IF_curr_delta introuvable dans cette version de sPyNNaker"
    )


def deploy(spec, x, time_scale_factor, shifts):
    """Exécute le réseau sur la carte ; retourne les spikes comptés par échantillon."""
    import pyNN.spiNNaker as sim

    time_steps, n_samples = x.shape[0], x.shape[1]
    max_depth = max(p["depth"] for p in spec["populations"])
    window = time_steps + max_depth + REST_STEPS
    run_time = float(n_samples * window)

    # Entrée : un spike répété n fois dans le même pas de temps vaut n évènements.
    frames = x.reshape(time_steps, n_samples, -1)
    spike_times = [[] for _ in range(frames.shape[2])]
    for step, sample, index in torch.nonzero(frames).tolist():
        count = round(frames[step, sample, index].item())
        spike_times[index].extend([float(sample * window + step)] * count)
    for times in spike_times:
        times.sort()
    active_steps = [
        float(s * window + k) for s in range(n_samples) for k in range(time_steps)
    ]

    sim.setup(timestep=DT, time_scale_factor=time_scale_factor)
    sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, INPUT_NEURONS_PER_CORE)
    cell_type = delta_neuron_model(sim)
    sim.set_number_of_neurons_per_core(cell_type, NEURONS_PER_CORE)
    try:
        pops = [
            sim.Population(
                spec["populations"][0]["size"],
                sim.SpikeSourceArray(spike_times=spike_times),
                label="Input",
            )
        ]
        bias_source = sim.Population(
            1, sim.SpikeSourceArray(spike_times=[active_steps]), label="Bias"
        )
        for pop in spec["populations"][1:]:
            cell = cell_type(
                tau_m=TAU_M,
                cm=TAU_M,
                v_rest=0.0,
                v_reset=0.0,
                v_thresh=pop["v_thresh"],
                tau_refrac=0.0,
                i_offset=0.0,
            )
            index = len(pops)
            options = {}
            if index in shifts:
                options["max_expected_summed_weight"] = [
                    0.75 * 2.0**shift for shift in shifts[index]
                ]
            try:
                population = sim.Population(
                    pop["size"],
                    cell,
                    label=pop["label"],
                    initial_values={"v": 0.0},
                    **options,
                )
            except TypeError:
                print(
                    "[!] max_expected_summed_weight refusé par cette version de sPyNNaker."
                )
                population = sim.Population(
                    pop["size"], cell, label=pop["label"], initial_values={"v": 0.0}
                )
            pops.append(population)

        for proj in spec["projections"]:
            for receptor, rows in split_by_sign(proj["conns"], DT):
                sim.Projection(
                    pops[proj["pre"]],
                    pops[proj["post"]],
                    sim.FromListConnector(rows, column_names=["weight", "delay"]),
                    receptor_type=receptor,
                )
        for index, pop in enumerate(spec["populations"]):
            if pop.get("bias") is None or not np.any(pop["bias"]):
                continue
            bias = np.asarray(pop["bias"], dtype=np.float64)
            conns = np.column_stack([np.zeros(len(bias)), np.arange(len(bias)), bias])
            for receptor, rows in split_by_sign(conns, pop["depth"] * DT):
                sim.Projection(
                    bias_source,
                    pops[index],
                    sim.FromListConnector(rows, column_names=["weight", "delay"]),
                    receptor_type=receptor,
                )

        readout_pop = pops[spec["readout"]["pre"]]
        readout_pop.record(["spikes"])
        sim.run(run_time)
        recorded = np.asarray(readout_pop.spinnaker_get_data("spikes"))
    finally:
        sim.end()

    counts = np.zeros((n_samples, spec["populations"][spec["readout"]["pre"]]["size"]))
    if len(recorded):
        samples = np.minimum((recorded[:, 1] // window).astype(np.int64), n_samples - 1)
        np.add.at(counts, (samples, recorded[:, 0].astype(np.int64)), 1.0)
    return counts, run_time


def archive_report(dataset, run_id, results_dir):
    """Archive le dossier de rapports sPyNNaker du run (énergie comprise)."""
    report_dir = Path("reports")
    if not report_dir.is_dir():
        return None
    candidates = [d for d in report_dir.iterdir() if d.is_dir() and d.name[:2] == "20"]
    if not candidates:
        return None
    target = results_dir / "hardware_reports" / dataset / run_id
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    shutil.move(str(latest), str(target))
    latest.mkdir(exist_ok=True)
    return target


def ensure_energy_report_enabled():
    """sPyNNaker n'écrit le rapport d'énergie que si le cfg le demande."""
    import configparser

    config = configparser.ConfigParser(strict=False)
    try:
        config.read([os.path.expanduser("~/.spynnaker.cfg"), "spynnaker.cfg"])
    except configparser.Error:
        pass
    enabled = config.get("Reports", "write_energy_report", fallback="False")
    if enabled.strip().lower() == "true":
        return
    if os.path.exists("spynnaker.cfg"):
        print(
            "[!] Ajoute 'write_energy_report = True' sous [Reports] de spynnaker.cfg."
        )
        return
    with open("spynnaker.cfg", "w", encoding="utf-8") as f:
        f.write("[Reports]\nwrite_energy_report = True\n")
    print("[*] spynnaker.cfg créé : rapport d'énergie activé.")


def evaluate_entry(entry, plan_dir, get_model, args):
    """Convertit un run du plan, l'émule, puis le déploie si demandé."""
    row = dict.fromkeys(SUMMARY_COLUMNS, "")
    row.update(
        run_id=tee.run_id_of(entry),
        dataset=entry["dataset"],
        architecture=entry["architecture"],
        T=entry["T"],
        Vth=entry["Vth"],
        total_samples=len(entry["samples"]),
    )
    start = time.time()
    time_steps = int(entry["T"])
    weights_path = tee.resolve_path(entry["weights_path"], plan_dir)
    with open(
        tee.resolve_path(entry["metadata_path"], plan_dir), encoding="utf-8"
    ) as f:
        metadata = json.load(f)
    spatial_shape = [int(s) for s in metadata["spatial_shape"]]

    args.max_spikes_per_neuron = args.max_input_neurons = None
    x, labels = tee.load_batch(
        [tee.resolve_path(p, plan_dir) for p in entry["samples"]],
        time_steps,
        spatial_shape,
        float(metadata.get("time_step_ms", 1.0)),
        args,
    )
    labels = labels.numpy()

    model = get_model(
        entry["architecture"], entry["dataset"], v_threshold=float(entry["Vth"])
    )
    model.load_state_dict(tee.load_state_dict(weights_path), strict=True)
    model.eval()

    spec = convert_model(model, spatial_shape, args.max_synapses)
    neurons, synapses, cores = network_size(spec)
    row.update(total_neurons=neurons, total_synapses=synapses)
    if cores > args.max_cores or synapses > args.max_synapses:
        raise TooLarge(f"{cores} cœurs et {synapses} synapses nécessaires")

    reference = []
    with torch.no_grad():
        for first in range(0, x.shape[1], 4):
            reference.append(model(x[:, first : first + 4]).argmax(dim=1))
            functional.reset_net(model)
    row["accuracy_pytorch_pct"] = round(
        100.0 * float((torch.cat(reference).numpy() == labels).mean()), 2
    )

    logits, peaks = emulate(spec, x)
    emulated = logits.argmax(axis=1)
    row["accuracy_emulated_pct"] = round(100.0 * float((emulated == labels).mean()), 2)

    shifts = {
        index: tuple(ring_buffer_shift(peak) for peak in peak_pair)
        for index, peak_pair in peaks.items()
    }
    fixed = emulate(spec, x, shifts)[0].argmax(axis=1)
    row["accuracy_fixed_point_pct"] = round(100.0 * float((fixed == labels).mean()), 2)

    if args.emulate:
        predictions, status = fixed, "EMULATED"
    else:
        counts, run_time = deploy(spec, x, args.time_scale_factor, shifts)
        predictions = readout_logits(spec, counts, time_steps).argmax(axis=1)
        row["sim_time_ms"] = run_time
        status = "SUCCESS"

    correct = int((predictions == labels).sum())
    row.update(
        status=status,
        accuracy_pct=round(100.0 * correct / len(labels), 2),
        correct_samples=correct,
        wall_time_s=round(time.time() - start, 2),
    )
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Déploiement fidèle des modèles sur SpiNNaker."
    )
    parser.add_argument("--plan", type=Path, default=Path("spinnaker_plan.json"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--emulate",
        action="store_true",
        help="Ne pas utiliser la carte : émuler le réseau converti en logiciel",
    )
    parser.add_argument("--time-scale-factor", type=int, default=TIME_SCALE_FACTOR)
    parser.add_argument(
        "--max-cores",
        type=int,
        default=700,
        help="Budget de cœurs (une carte SpiNN-5 en offre environ 750 utilisables)",
    )
    parser.add_argument("--max-synapses", type=int, default=30_000_000)
    args = parser.parse_args(argv)
    if not 1 <= args.time_scale_factor <= 200:
        parser.error("--time-scale-factor doit être entre 1 et 200.")

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / (
        "summary_emulation.csv" if args.emulate else "summary_benchmark.csv"
    )
    done = set()
    if summary_path.is_file():
        with open(summary_path, newline="", encoding="utf-8") as f:
            done = {r["run_id"] for r in csv.DictReader(f) if r["status"] == "SUCCESS"}
    else:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS).writeheader()
    if not args.emulate:
        ensure_energy_report_enabled()

    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)
    plan_dir = args.plan.resolve().parent
    get_model = tee.resolve_model_factory()
    entries = [
        e
        for e in plan
        if args.dataset in (None, e["dataset"])
        and args.architecture in (None, e["architecture"])
        and tee.run_id_of(e) not in done
    ][: args.limit]

    for index, entry in enumerate(entries, start=1):
        run_id = tee.run_id_of(entry)
        try:
            row = evaluate_entry(entry, plan_dir, get_model, args)
        except Exception as exc:  # noqa: BLE001 - un run en échec ne doit pas arrêter le plan
            row = dict.fromkeys(SUMMARY_COLUMNS, "")
            status = {Unsupported: "UNSUPPORTED", TooLarge: "TOO_LARGE"}.get(
                type(exc), "EMULATION_ERROR" if args.emulate else "HARDWARE_ERROR"
            )
            row.update(
                run_id=run_id,
                dataset=entry["dataset"],
                architecture=entry["architecture"],
                T=entry["T"],
                Vth=entry["Vth"],
                status=status,
                total_samples=len(entry["samples"]),
                error_msg=f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200],
            )

        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS).writerow(row)
        print(
            f"[{index}/{len(entries)}] {run_id} : {row['status']} | "
            f"pytorch={row['accuracy_pytorch_pct']}% "
            f"émulé={row['accuracy_emulated_pct']}% "
            f"virgule_fixe={row['accuracy_fixed_point_pct']}% "
            f"carte={'-' if args.emulate else row['accuracy_pct']}% "
            f"{row['error_msg']}"
        )

        if row["status"] == "SUCCESS":
            archive_report(entry["dataset"], run_id, results_dir)
            if spinnaker_energy is not None:
                try:
                    spinnaker_energy.collect_to_csv(results_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"[!] Extraction de l'énergie impossible : {exc}")

    print(f"Résumé : {summary_path}")


if __name__ == "__main__":
    main()
