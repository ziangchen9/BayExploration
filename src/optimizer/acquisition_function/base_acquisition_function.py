from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable

import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from src.optimizer.gaussian_model.base_guassian import BaseGPModel


class BaseAcquisitionFunction(ABC):
    """Base acquisition functions."""

    # kwargs for setting up the sampler
    SAMPLER_KWARGS = {"sampler_shape": torch.Size([1024])}

    # kwargs for setting up the acquisition function
    ACQ_FUNC_KWARGS = {}

    # kwargs for optimizing the acquisition function
    OPTIMIZE_KWARGS = {
        "bounds": None,
        "q": 1,
        "num_restarts": 20,
        "raw_samples": 50,
        "options": {"dtype": torch.float64, "with_grad": True},
    }

    def __init__(self):
        self.sampler_builder = None
        self._build_sampler()
        self.acquisition_function_builder = None

    def _build_sampler(self) -> None:
        self.sampler_builder = partial(
            SobolQMCNormalSampler, sample_shape=self.SAMPLER_KWARGS["sampler_shape"]
        )

    @abstractmethod
    def _build_acquisition_function_builder(self) -> Callable:
        raise NotImplementedError

    def setup(self):
        self.acquisition_function_builder = self._build_acquisition_function_builder()

    @abstractmethod
    def _optimize(self, gp_model: BaseGPModel) -> Any:
        raise NotImplementedError

    def optimize(self, seed: int, gp_model: BaseGPModel) -> Any:
        sampler = self.sampler_builder(seed=seed)
        acquisition_function = self.acquisition_function_builder(
            model=gp_model.model, sampler=sampler, **self.ACQ_FUNC_KWARGS
        )
        return self._optimize(acquisition_function=acquisition_function)
