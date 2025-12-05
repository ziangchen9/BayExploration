from typing import Any

from core.optimizer.acquisition_function.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from core.optimizer.gaussian_model.base_guassian import BaseGPModel


class BaseBayesianOptimizer:
    def __init__(
        self, model: BaseGPModel, acquisition_function: BaseAcquisitionFunction
    ):
        self.model = model
        self.acquisition_function = acquisition_function
        self.acquisition_function.setup(self.model)

    def optimize(self) -> Any:
        result = self.acquisition_function.optimize()
        return result
