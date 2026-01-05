from functools import partial
from typing import Any, Callable

import torch
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf

from src.optimizer.acquisition_function.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from src.optimizer.gaussian_model.base_guassian import BaseGPModel


class QUCBAcquisitionFunction(BaseAcquisitionFunction):
    ACQ_FUNC_KWARGS = {"beta": 0.1}

    def _build_acquisition_function_builder(self) -> Callable:
        acq_func_builder = partial(qUpperConfidenceBound, **self.ACQ_FUNC_KWARGS)
        return acq_func_builder

    def _optimize(self, acquisition_function=None, **kwargs) -> Any:
        bounds = self.OPTIMIZE_KWARGS.get("bounds")
        if bounds is None:
            raise ValueError("bounds must be set in OPTIMIZE_KWARGS")
        candidate, acq_value = optimize_acqf(
            acq_function=acquisition_function,
            bounds=bounds,
            **{k: v for k, v in self.OPTIMIZE_KWARGS.items() if k != "bounds"}
        )
        return candidate, acq_value
