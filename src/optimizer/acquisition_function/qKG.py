from typing import Any

import torch
from botorch.acquisition import qKnowledgeGradient
from botorch.optim import optimize_acqf

from src.optimizer.acquisition_function.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from src.optimizer.gaussian_model.single_guassian import BaseGPModel


class QKGAcquisitionFunction(BaseAcquisitionFunction):

    OPTIMIZE_KWARGS = {
        "q": 1,
        "num_restarts": 20,
        "raw_samples": 50,
        "options": {"dtype": torch.float64, "with_grad": True},
    }

    def _setup(self, pg: BaseGPModel, bounds: torch.Tensor, **kwargs):
        return qKnowledgeGradient(model=pg.model, **kwargs, **self.OPTIMIZE_KWARGS)

    def _optimize(self, seed: int, **kwargs) -> Any:
        candidate, acq_value = optimize_acqf(
            self.acquisition_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )
        return candidate, acq_value
