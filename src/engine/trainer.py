import torch
from tqdm import tqdm
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
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(events)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        model.reset_states()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device, measure_sparsity=False):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    firing_rates = []
    hooks = []
    
    if measure_sparsity:
        def hook(module, input, output):
            firing_rates.append(output.detach().mean().item())
        
        for m in model.modules():
            if isinstance(m, MemoryModule):
                hooks.append(m.register_forward_hook(hook))

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for events, targets in pbar:
            events, targets = events.to(device, dtype=torch.float32), targets.to(device)
            outputs = model(events)
            loss = criterion(outputs, targets)
            
            model.reset_states()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    sparsity = 0.0
    if measure_sparsity:
        for h in hooks:
            h.remove()
        avg_firing_rate = sum(firing_rates) / len(firing_rates) if firing_rates else 0.0
        sparsity = (1.0 - avg_firing_rate) * 100

    return total_loss / len(dataloader), 100. * correct / total, sparsity