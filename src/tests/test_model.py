import torch
import torch.nn as nn
from src.models import SpikingMLP, SpikingVGG5
from src.engine import train_one_epoch

def test_overfit_single_batch_NMNIST():
    """
    Verifies that the model SpikingMLP can overfit a random batch,
    validating backpropagation and the SNN computation graph.
    """
    torch.manual_seed(67)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpikingMLP(input_size=2312, hidden_size=256, output_size=10).to(device)

    batch_size = 4
    T = 16
    C, H, W = 2, 34, 34
    
    events = (torch.rand(T, batch_size, C, H, W) > 0.5).float().to(device)
    targets = torch.randint(0, 10, (batch_size,)).to(device)
    
    fake_loader = [(events, targets)]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 200
    
    for _ in range(epochs):
        final_loss, final_acc = train_one_epoch(model, fake_loader, optimizer, criterion, device)
        if final_acc == 100.0 and final_loss < 0.05:
            break
            
    assert final_acc == 100.0
    assert final_loss < 0.05

def test_overfit_single_batch_DVSGesture():
    """
    Verifies that the model SpikingVGG5 can overfit a random batch,
    validating backpropagation and the SNN computation graph.
    """
    torch.manual_seed(67)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpikingVGG5(in_channels=2, out_classes=11).to(device)

    batch_size = 2
    T = 2
    C, H, W = 2, 128, 128

    events = (torch.rand(T, batch_size, C, H, W) > 0.9).float().to(device)
    targets = torch.randint(0, 11, (batch_size,)).to(device)

    fake_loader = [(events, targets)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    initial_loss, _ = train_one_epoch(model, fake_loader, optimizer, criterion, device)

    for _ in range(5):
        final_loss, _ = train_one_epoch(model, fake_loader, optimizer, criterion, device)

    assert final_loss < initial_loss, f"Erreur de propagation ! Init: {initial_loss:.4f}, Final: {final_loss:.4f}"