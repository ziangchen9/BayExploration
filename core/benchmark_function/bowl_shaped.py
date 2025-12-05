import torch

from core.benchmark_function.base_function import BaseTestFunction


class Booth(BaseTestFunction):
    _dim = 2
    _optimal = [1.0, 3.0]
    _optimal_value = 0.0
    _bound = [[-10.0, -10.0], [10.0, 10.0]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level, **kwargs)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x_1, x_2 = x[..., 0], x[..., 1]
        y = (x_1 + 2 * x_2 - 7) ** 2 + (2 * x_1 + x_2 - 5) ** 2
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)


class Bohachevsky(BaseTestFunction):
    _dim = 2
    _optimal = [0.0, 0.0]
    _optimal_value = 0.0
    _bound = [[-100.0, -100.0], [100.0, 100.0]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level=noise_level)
        self.kwargs = {**self._default_kwargs, **kwargs}

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        y = (
            x1**2
            + 2 * x2**2
            - 0.3 * torch.cos(3 * torch.pi * x1)
            - 0.4 * torch.cos(4 * torch.pi * x2)
            + 0.7
        )
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)
