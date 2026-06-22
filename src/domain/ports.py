from abc import ABC, abstractmethod
import torch

class ISNNModel(ABC):
    """
    Interface defining the expected behavior of an SNN model regardless of its implementation (PyTorch, SpikingJelly, etc.).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reset_states(self) -> None:
        pass