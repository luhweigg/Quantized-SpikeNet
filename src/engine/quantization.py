import torch
import torch.nn as nn

def quantize_weights(model: nn.Module, num_bits: int = 8):
    """
    Apply Fake Quantization for PyTorch evaluation and return the true integer weights (int8) for FPGA export.
    """
    quantized_state_dict = {}
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                scale = param.abs().max() / qmax
                int_weights = torch.clamp(torch.round(param / scale), qmin, qmax)
                quantized_state_dict[name] = int_weights.to(torch.int8)
                param.copy_(int_weights * scale)
                
    return quantized_state_dict