from abc import ABC, abstractmethod
from functools import partial

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import Kernel
from gpytorch.means import Mean


class BaseGPModel(ABC):

    def __init__(
        self, mean_module: Mean, covariance_module: Kernel, target_function_dim: int
    ):
        self.mean_module = mean_module
        self.covariance_module = covariance_module
        self.gp_builder = partial(
            SingleTaskGP,
            input_transform=Normalize(d=target_function_dim),
            outcome_transform=Standardize(m=1),
            mean_module=mean_module,
            covar_module=covariance_module,
        )

    @abstractmethod
    def _fit(self, model: SingleTaskGP) -> SingleTaskGP:
        raise NotImplementedError

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
        model = self.gp_builder(x, y)
        return self._fit(model)
