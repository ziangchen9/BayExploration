from typing import Dict, Type

from gpytorch.kernels import Kernel
from gpytorch.means import Mean

from src.optimizer.gaussian_model.base_guassian import BaseGPModel
from src.optimizer.gaussian_model.single_guassian import SingleTaskGPModel

GAUSSIAN_MODEL_REGISTRY: Dict[str, Type[BaseGPModel]] = {
    "singletaskgp": SingleTaskGPModel,
    "single_task_gp": SingleTaskGPModel,
}


def get_gaussian_model(
    name: str, mean_module: Mean, covariance_module: Kernel, target_function_dim: int
) -> BaseGPModel:
    name_lower = name.lower().replace("-", "_")
    if name_lower not in GAUSSIAN_MODEL_REGISTRY:
        available = ", ".join(sorted(GAUSSIAN_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown gaussian model name: {name}. Available models: {available}"
        )
    return GAUSSIAN_MODEL_REGISTRY[name_lower](
        mean_module=mean_module,
        covariance_module=covariance_module,
        target_function_dim=target_function_dim,
    )


def list_available_gaussian_models() -> list[str]:
    return sorted(set(GAUSSIAN_MODEL_REGISTRY.keys()))


__all__ = [
    "GAUSSIAN_MODEL_REGISTRY",
    "get_gaussian_model",
    "list_available_gaussian_models",
]
