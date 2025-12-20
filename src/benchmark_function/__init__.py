"""Benchmark function module

测试函数定义模块。注册信息请参考 src.registry.benchmark_function_registry
"""

from src.benchmark_function.base_function import BaseTestFunction

# 从 registry 导入注册表和工厂函数
from src.registry.benchmark_function_registry import (
    BENCHMARK_FUNCTION_REGISTRY,
    get_test_function,
)

# 向后兼容：导出 BaseTestFunction 和 get_test_function
__all__ = [
    "BaseTestFunction",
    "BENCHMARK_FUNCTION_REGISTRY",
    "get_test_function",
]
