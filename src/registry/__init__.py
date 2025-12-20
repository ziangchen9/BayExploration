"""Registry module for all registered components"""

# 测试函数注册
# 采集函数注册
from src.registry.acquisition_function_registry import (
    ACQUISITION_FUNCTION_REGISTRY,
    get_acquisition_function,
    list_available_acquisition_functions,
)
from src.registry.benchmark_function_registry import (
    BENCHMARK_FUNCTION_REGISTRY,
    get_test_function,
    list_available_functions,
)

# 协方差函数注册
from src.registry.covariance_function_registry import (
    COVARIANCE_FUNCTION_REGISTRY,
    get_covariance_function,
    list_available_covariance_functions,
)

# 均值函数注册
from src.registry.mean_function_registry import (
    MEAN_FUNCTION_REGISTRY,
    get_mean_function,
    list_available_mean_functions,
)

__all__ = [
    # 测试函数
    "BENCHMARK_FUNCTION_REGISTRY",
    "get_test_function",
    "list_available_functions",
    # 均值函数
    "MEAN_FUNCTION_REGISTRY",
    "get_mean_function",
    "list_available_mean_functions",
    # 协方差函数
    "COVARIANCE_FUNCTION_REGISTRY",
    "get_covariance_function",
    "list_available_covariance_functions",
    # 采集函数
    "ACQUISITION_FUNCTION_REGISTRY",
    "get_acquisition_function",
    "list_available_acquisition_functions",
]
