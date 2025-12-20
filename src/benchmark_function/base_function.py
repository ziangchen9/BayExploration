from abc import ABC, abstractmethod
from typing import List

import torch


class BaseTestFunction(ABC):
    _dim: int = 1
    _optimal: List[float] = []
    _optimal_value: float = 0.0
    _bound: List[List[float, float]] = []
    _default_kwargs = {
        "dtype": torch.float64,
        "device": torch.device("cpu"),
    }

    def __init__(self, noise_level: float = 0.05, **kwargs):
        self.noise_level: float = noise_level
        self.kwargs = {**self._default_kwargs, **kwargs}

    @abstractmethod
    def __call__(self, x: torch.Tensor, noise_level: float = 0.0) -> torch.Tensor:
        return self._evaluate(x) + self._noise(x, noise_level)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def optimal(self) -> torch.Tensor:
        return torch.tensor(self._optimal, **self.kwargs)

    @property
    def bound(self) -> torch.Tensor:
        return torch.tensor(self._bound, **self.kwargs)

    @abstractmethod
    def _evaluate(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _noise(x: torch.Tensor, noise_level: float) -> torch.Tensor:
        return torch.randn_like(x) * noise_level
