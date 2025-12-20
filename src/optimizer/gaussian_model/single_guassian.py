from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import Kernel
from gpytorch.means import Mean

from src.optimizer.gaussian_model.base_guassian import BaseGPModel


class SingleTaskGPModel(BaseGPModel):
    def __init__(
        self, mean_module: Mean, covariance_module: Kernel, target_function_dim: int
    ):
        super().__init__(
            mean_module=mean_module,
            covariance_module=covariance_module,
            target_function_dim=target_function_dim,
        )

    def _fit(self, model: SingleTaskGP) -> SingleTaskGP:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return model
