"""Benchmark function module

测试函数定义模块。注册信息请参考 src.registry.benchmark_function_registry
"""

from src.benchmark_function.base_function import BaseTestFunction

# 注意：为了避免循环导入，这里不直接导入 registry
# 如果需要使用注册表和工厂函数，请从 src.registry 导入：
#   from src.registry import get_target_function, get_test_function, BENCHMARK_FUNCTION_REGISTRY

__all__ = [
    "BaseTestFunction",
]
