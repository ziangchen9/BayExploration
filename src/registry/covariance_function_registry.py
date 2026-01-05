from typing import Callable, Dict

from gpytorch.kernels import MaternKernel, RBFKernel

COVARIANCE_FUNCTION_REGISTRY: Dict[str, Callable] = {
    "rbf": lambda **kwargs: RBFKernel(**kwargs),
    "matern1_5": lambda **kwargs: MaternKernel(nu=1.5, **kwargs),
    "matern2_5": lambda **kwargs: MaternKernel(nu=2.5, **kwargs),
}


def get_covariance_function(name: str) -> Callable:
    name_lower = name.lower()
    if name_lower not in COVARIANCE_FUNCTION_REGISTRY:
        available = ", ".join(sorted(set(COVARIANCE_FUNCTION_REGISTRY.keys())))
        raise ValueError(
            f"Unknown covariance function name: {name}. Available functions: {available}"
        )
    return COVARIANCE_FUNCTION_REGISTRY[name_lower]()


def list_available_covariance_functions() -> list[str]:
    return sorted(set(COVARIANCE_FUNCTION_REGISTRY.keys()))


__all__ = [
    "COVARIANCE_FUNCTION_REGISTRY",
    "get_covariance_function",
    "list_available_covariance_functions",
]
