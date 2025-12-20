from typing import Dict, Type

from src.benchmark_function.base_function import BaseTestFunction
from src.benchmark_function.bowl_shaped import Bohachevsky, Booth
from src.benchmark_function.many_local_minima import Ackley
from src.benchmark_function.steep_ridges import Easom
from src.benchmark_function.valley_shaped import ThreeHumpCamel

BENCHMARK_FUNCTION_REGISTRY: Dict[str, Type[BaseTestFunction]] = {
    "booth": Booth,
    "bohachevsky": Bohachevsky,
    "threehumpcamel": ThreeHumpCamel,
    "ackley": Ackley,
    "easom": Easom,
}


def get_test_function(name: str, **kwargs) -> BaseTestFunction:
    name_lower = name.lower()
    if name_lower not in BENCHMARK_FUNCTION_REGISTRY:
        available = ", ".join(sorted(BENCHMARK_FUNCTION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown test function name: {name}. Available functions: {available}"
        )
    return BENCHMARK_FUNCTION_REGISTRY[name_lower](**kwargs)


def list_available_functions() -> list[str]:
    return sorted(BENCHMARK_FUNCTION_REGISTRY.keys())


__all__ = [
    "BENCHMARK_FUNCTION_REGISTRY",
    "get_test_function",
    "list_available_functions",
]
