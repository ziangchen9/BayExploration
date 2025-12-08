from typing import Any

import torch
from botorch.acquisition import qKnowledgeGradient
from botorch.optim import optimize_acqf

from core.optimizer.acquisition_function.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from core.optimizer.gaussian_model.single_guassian import BaseGPModel


class QKGAcquisitionFunction(BaseAcquisitionFunction):
    def _setup(self, pg: BaseGPModel, **kwargs):
        num_fantasies = kwargs.get("num_fantasies", 128)
        return qKnowledgeGradient(model=pg.model, num_fantasies=num_fantasies)

    def _optimize(self, seed: int, **kwargs) -> Any:
        bounds = kwargs.get("bounds", None)
        q = kwargs.get("q", 1)
        num_restarts = kwargs.get("num_restarts", 20)
        raw_samples = kwargs.get("raw_samples", 50)
        options = kwargs.get("options", {"dtype": torch.float64, "with_grad": True})
        candidate, acq_value = optimize_acqf(
            self.acquisition_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )
        return candidate, acq_value
