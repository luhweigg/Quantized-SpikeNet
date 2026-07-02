import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode

a = 1.0 # Controls the width and stepness of the surrogate gradient function.

class SpikeFunction(torch.autograd.Function):
    """
    Heaviside step function with a custom surrogate gradient for backpropagation.
    """
    @staticmethod
    def forward(ctx, input, v_threshold):
        ctx.save_for_backward(input)
        ctx.v_threshold = v_threshold
        output = torch.gt(input, v_threshold)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        v_threshold = ctx.v_threshold
        grad_input = grad_output.clone()
        # Rectangular surrogate gradient centered at v_threshold with width controlled by 'a'
        hu = (abs(input - v_threshold) < (a / 2.0)) / a
        return grad_input * hu, None

spikefunc = SpikeFunction.apply

class MixedLIF(nn.Module):
    """
    MixedLIF neuron implementing the dual-path activation (Spiking and Surrogate)
    to stabilize self-supervised learning (SSL).
    """
    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, **kwargs):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.step_mode = 'm'

    def forward(self, x):
        T = x.shape[0]  # Time steps
        B = x.shape[1]  # Batch size
        bs = B // 2     # Split batch in half : Path A (Spiking) and Path B (Surrogate)
        u = torch.zeros((bs,) + x.shape[2:], device=x.device)
        u2 = torch.zeros((bs,) + x.shape[2:], device=x.device)
        o = torch.zeros(x.shape, device=x.device)

        for t in range(T):
            # Path A : Standard LIF dynamics with hard reset
            u = (1.0 / self.tau) * u * (1.0 - spikefunc(u, self.v_threshold).detach()) + x[t, :bs, ...]

            # Path B : Surrogate continuous dynamics
            u2 = (1.0 / self.tau) * u2 * (1.0 - spikefunc(u2, self.v_threshold).detach()) + x[t, bs:, ...]

            # Output path A : Discrete spikes
            o[t, :bs, ...] = spikefunc(u, self.v_threshold)

            # Output path B : Clipped ReLU-like continuous activation (antiderivative of surrogate)
            o[t, bs:, ...] = torch.clamp(u2 - self.v_threshold + 0.5, min=0.0, max=1.0)
        return o

class NeuronFactory:
    """
    Factory to dynamically instantiate the required neuron type.
    Supports LIFNode (Spikingjelly) and MixedLIF (custom).
    """
    @staticmethod
    def build(neuron_type: str, **kwargs):
        if neuron_type == "LIF":
            kwargs.setdefault("step_mode", "m")
            kwargs.setdefault("detach_reset", True)
            return LIFNode(**kwargs)
        elif neuron_type == "MixedLIF":
            return MixedLIF(**kwargs)
        else:
            raise ValueError(f"Unsupported neuron type: {neuron_type}")