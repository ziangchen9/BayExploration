from abc import ABC, abstractmethod
from typing import Any

import torch
from botorch.models import SingleTaskGP

from core.optimizer.gaussian_model.single_guassian import BaseGPModel


class BaseAcquisitionFunction(ABC):
    """Base acquisition functions."""

    OPTIMIZE_KWARGS = {
        "bounds": None,
        "q": 1,
        "num_restarts": 20,
        "raw_samples": 50,
        "options": {"dtype": torch.float64, "with_grad": True},
    }

    def __init__(self):
        self._seed = 0
        self.acquisition_function = None

    def _next_seed(self) -> int:
        seed = self._seed
        self._seed += 1
        return seed

    @abstractmethod
    def _setup(self, pg: SingleTaskGP, **kwargs):
        raise NotImplementedError

    def setup(self, pg: BaseGPModel, **kwargs):
        self.acquisition_function = self._setup(pg=pg.model, **kwargs)
        return self.acquisition_function

    @abstractmethod
    def _optimize(self, seed: int, **kwargs) -> Any:
        raise NotImplementedError

    def optimize(self, **kwargs) -> Any:
        seed = self._next_seed()
        return self._optimize(seed=seed, **kwargs)
