from typing import Any

import torch
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.optim import optimize_acqf

from core.optimizer.acquisition_function.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from core.optimizer.gaussian_model.base_guassian import BaseGPModel


class QPESAcquisitionFunction(BaseAcquisitionFunction):
    def _setup(self, pg: BaseGPModel, **kwargs):
        return qPredictiveEntropySearch(model=pg.model)

    def _optimize(self, seed: int, **kwargs) -> Any:
        bounds = kwargs.get("bounds", None)
        q = kwargs.get("q", 1)
        num_restarts = kwargs.get("num_restarts", 20)
        raw_samples = kwargs.get("raw_samples", 50)
        options = kwargs.get("options", {"dtype": torch.float64, "with_grad": False})
        candidate, acq_value = optimize_acqf(
            self.acquisition_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )
        return candidate, acq_value
