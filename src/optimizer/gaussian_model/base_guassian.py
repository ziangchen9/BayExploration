from abc import ABC, abstractmethod
from functools import partial

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import Kernel
from gpytorch.means import Mean


class BaseGPModel(ABC):

    def __init__(self, mean_module=None, covariance_module=None, target_function_dim=None):
        self.mean_module = mean_module
        self.covariance_module = covariance_module
        self.target_function_dim = target_function_dim
        self.gaussian_model_builder = partial(SingleTaskGP)
        self.model = None

    @abstractmethod
    def _fit(self, model: SingleTaskGP) -> SingleTaskGP:
        raise NotImplementedError

    def fit(self,target_function_dim: int, x: torch.Tensor, y: torch.Tensor) -> None:
        model = self.gaussian_model_builder(
            train_X=x,
            train_Y=y,
            input_transform=Normalize(d=target_function_dim),
            outcome_transform=Standardize(m=1),
        )
        self.model = self._fit(model)
