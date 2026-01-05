import numpy as np
import torch

from src.benchmark_function.base_function import BaseTestFunction


class Ackley(BaseTestFunction):
    # TODO: add description of Ackley
    _dim = 2
    _optimal = [0.0, 0.0]
    _optimal_value = 0.0
    _bound = [[-32.7680, -32.7680], [32.7680, 32.7680]]

    def __init__(self, noise_level: float = 0.05):
        super().__init__(noise_level)

    def evaluate(self, x: torch.Tensor, noise: float) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        y = (
            torch.exp(-0.2 * torch.sqrt(0.5 * (x1**2 + x2**2)))
            - torch.exp(0.5 * (torch.cos(2 * np.pi * x1) + torch.cos(2 * np.pi * x2)))
            + 1
        )
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)
