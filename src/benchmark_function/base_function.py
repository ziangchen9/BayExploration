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

    def __init__(self, noise_level: float = 0.05):
        self.noise_level: float = noise_level

    def __call__(self, x: torch.Tensor, noise_level: float = 0.0) -> torch.Tensor:
        # 如果没有指定noise_level，使用实例的noise_level
        if noise_level == 0.0:
            noise_level = self.noise_level
        result = self._evaluate(x) + self._noise(x, noise_level)
        # 确保返回形状正确 [..., 1]
        if result.dim() == 0:
            return result.unsqueeze(-1)
        if result.dim() == 1:
            return result.unsqueeze(-1)
        return result

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def optimal(self) -> torch.Tensor:
        return torch.tensor(self._optimal, **self._default_kwargs)

    @property
    def bound(self) -> torch.Tensor:
        return torch.tensor(self._bound, **self._default_kwargs)

    @abstractmethod
    def _evaluate(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _noise(x: torch.Tensor, noise_level: float) -> torch.Tensor:
        return torch.randn_like(x) * noise_level
