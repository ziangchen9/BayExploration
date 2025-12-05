from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from core.optimizer.gaussian_model.base_guassian import BaseGPModel
from core.optimizer.gaussian_model.covariance_function.covariance import (
    COVARIANCE_MODULE_MAP,
)
from core.optimizer.gaussian_model.mean_fuctions.means import MEAN_MODULE_MAP


class ConstantMeanSingleTaskGPModel(BaseGPModel):
    """All GP model with constant mean should inherit from this class"""

    def __init__(self, covariance_module: str, **kwargs):
        super().__init__(
            mean_module="constant", covariance_module=covariance_module, **kwargs
        )

    def _fit(self, x: Tensor, y: Tensor) -> SingleTaskGP:
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            mean_module=MEAN_MODULE_MAP[self.mean_module],
            covar_module=COVARIANCE_MODULE_MAP[self.covariance_module],
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class LinearMeanSingleTaskGPModel(BaseGPModel):
    """All GP model with linear mean should inherit from this class"""

    def __init__(self, covariance_module: str, **kwargs):
        super().__init__(
            mean_module="linear", covariance_module=covariance_module, **kwargs
        )

    def _fit(self, x: Tensor, y: Tensor) -> SingleTaskGP:
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            mean_module=MEAN_MODULE_MAP[self.mean_module],
            covar_module=COVARIANCE_MODULE_MAP[self.covariance_module],
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class QuadraticMeanSingleTaskGPModel(BaseGPModel):
    """All GP model with quadratic mean should inherit from this class"""

    def __init__(self, covariance_module: str, **kwargs):
        super().__init__(
            mean_module="quadratic", covariance_module=covariance_module, **kwargs
        )

    def _fit(self, x: Tensor, y: Tensor) -> SingleTaskGP:
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            mean_module=MEAN_MODULE_MAP[self.mean_module],
            covar_module=COVARIANCE_MODULE_MAP[self.covariance_module],
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class RBFConstantMeanSingleTaskGPModel(ConstantMeanSingleTaskGPModel):
    def __init__(self, **kwargs):
        super().__init__(covariance_module="rbf")


class Matern1_5ConstantMeanSingleTaskGPModel(ConstantMeanSingleTaskGPModel):
    def __init__(self, **kwargs):
        super().__init__(covariance_module="matern1_5")


class Matern2_5ConstantMeanSingleTaskGPModel(ConstantMeanSingleTaskGPModel):
    def __init__(self, **kwargs):
        super().__init__(covariance_module="matern2_5")
