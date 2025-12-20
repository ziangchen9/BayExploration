import torch

from src.benchmark_function.base_function import BaseTestFunction


class ThreeHumpCamel(BaseTestFunction):
    _dim = 2
    _optimal = [0.0, 0.0]
    _optimal_value = 0.0
    _bound = [[-1.0, -1.0], [1.0, 1.0]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level=noise_level)
        self.kwargs = {**self._default_kwargs, **kwargs}

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        y = 2.0 * x1**2 - 1.05 * x1**4 + x1**6 / 6.0 + x1 * x2 + x2**2
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)
