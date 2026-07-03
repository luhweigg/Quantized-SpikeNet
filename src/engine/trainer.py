import torch
from tqdm import tqdm
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.base import MemoryModule


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Epoch", leave=False)
    for events, targets in pbar:
        events, targets = events.to(device, dtype=torch.float32), targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                outputs = model(events)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(events)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        functional.reset_net(model)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    measure_consumption=False,
    energy_per_spike=0.9e-12,
    time_per_inference=0.01,
    static_power=0.0,
):
    """
    Evaluate the model on the test set and estimate sparsity, energy (Joules) and power (Watts).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    firing_rates = []
    hooks = []

    if measure_consumption:

        def hook(module, input, output):
            detached = output.detach()
            firing_rates.append((detached.sum().item(), detached.numel()))

        for m in model.modules():
            if isinstance(m, MemoryModule):
                hooks.append(m.register_forward_hook(hook))

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for events, targets in pbar:
            events, targets = events.to(device, dtype=torch.float32), targets.to(device)
            outputs = model(events)
            loss = criterion(outputs, targets)

            functional.reset_net(model)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    sparsity = 0.0
    power_watts = 0.0
    energy_joules = 0.0

    if measure_consumption:
        for h in hooks:
            h.remove()
        total_spikes = sum(spikes for spikes, _ in firing_rates)
        total_elements = sum(elements for _, elements in firing_rates)

        if total_elements > 0:
            avg_firing_rate = total_spikes / total_elements
            sparsity = (1.0 - avg_firing_rate) * 100

            spikes_per_inference = total_spikes / total if total > 0 else 0
            energy_joules = spikes_per_inference * energy_per_spike
            power_watts = (energy_joules / time_per_inference) + static_power

    return (
        total_loss / len(dataloader),
        100.0 * correct / total,
        sparsity,
        energy_joules,
        power_watts,
    )

def train_step(model, criterion, optimizer, x_i, x_j):
    optimizer.zero_grad()

    x = torch.cat((x_i, x_j), dim=0)
    
    f, z = model(x)
    
    B = z.shape[1]
    b = B // 2
    
    z_i = z[:, :b, ...]
    z_j = z[:, b:, ...]

    loss = criterion(z_i, z_j)
    
    loss.backward()
    optimizer.step()

    return loss.item()