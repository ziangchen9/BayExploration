import torch

from src.benchmark_function.base_function import BaseTestFunction


class Easom(BaseTestFunction):
    _dim = 2
    _optimal = [0.0, 0.0]
    _optimal_value = -1.0
    _bound = [[-5.0, -5.0], [5.0, 5.0]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level=noise_level)
        self.kwargs = {**self._default_kwargs, **kwargs}

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        y = (
            -torch.cos(x1)
            * torch.cos(x2)
            * torch.exp(-((x1 - torch.pi) ** 2 + (x2 - torch.pi) ** 2))
        )
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)
