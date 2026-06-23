import torch
from src.models import SpikingMLP
from src.engine import quantize_weights

def test_quantization_bounds_and_types():
    """
    Verify that weight quantization strictly produces int8 tensors and stays within [-128, 127] bounds.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpikingMLP(input_size=2312, hidden_size=256, output_size=10).to(device)

    quantized_state = quantize_weights(model, num_bits=8)
    
    for name, param in quantized_state.items():
        assert param.dtype == torch.int8
        assert param.min() >= -128
        assert param.max() <= 127