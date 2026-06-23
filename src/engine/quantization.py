import torch
import torch.nn as nn

def quantize_weights(model: nn.Module, num_bits: int = 8, return_metadata: bool = False):
    """
    Apply Fake Quantization for PyTorch evaluation and return the true integer weights (int8) for FPGA export.
    """
    quantized_state_dict = {}
    metadata = {}
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                max_abs = param.abs().max()
                scale = max_abs / qmax if max_abs > 0 else torch.tensor(1.0, device=param.device, dtype=param.dtype)
                int_weights = torch.clamp(torch.round(param / scale), qmin, qmax)
                quantized_state_dict[name] = int_weights.to(torch.int8)
                param.copy_(int_weights * scale)
                metadata[name] = {
                    'scale': float(scale.item()),
                    'zero_point': 0,
                    'num_bits': num_bits,
                    'qmin': qmin,
                    'qmax': qmax,
                }

    if return_metadata:
        return quantized_state_dict, metadata
    return quantized_state_dict