from typing import Callable, Dict, Type

from gpytorch.means import ConstantMean, LinearMean

from src.optimizer.gaussian_model.mean_fuctions.means import QuadraticMean

MEAN_FUNCTION_REGISTRY: Dict[str, Type | Callable] = {
    "constant": ConstantMean,
    "linear": LinearMean,
    "quadratic": QuadraticMean,
}


def get_mean_function(name: str) -> Callable:
    name_lower = name.lower()
    if name_lower not in MEAN_FUNCTION_REGISTRY:
        available = ", ".join(sorted(MEAN_FUNCTION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown mean function name: {name}. Available functions: {available}"
        )
    return MEAN_FUNCTION_REGISTRY[name_lower]()


def list_available_mean_functions() -> list[str]:
    return sorted(MEAN_FUNCTION_REGISTRY.keys())


__all__ = [
    "MEAN_FUNCTION_REGISTRY",
    "get_mean_function",
    "list_available_mean_functions",
]
