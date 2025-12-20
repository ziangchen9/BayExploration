"""采集函数注册表

这个文件集中管理所有采集函数的注册信息。
采集函数用于贝叶斯优化中选择下一个评估点。
"""

from typing import Dict, Type

from src.optimizer.acquisition_function.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from src.optimizer.acquisition_function.qKG import QKGAcquisitionFunction
from src.optimizer.acquisition_function.qLBMVE import QLBMVEAcquisitionFunction
from src.optimizer.acquisition_function.qLogNoisyEI import QLogNoisyEI
from src.optimizer.acquisition_function.qMVE import QMVEAcquisitionFunction
from src.optimizer.acquisition_function.qPES import QPESAcquisitionFunction
from src.optimizer.acquisition_function.qSimpleRegret import (
    QSimpleRegretAcquisitionFunction,
)
from src.optimizer.acquisition_function.qUCB import QUCBAcquisitionFunction
from src.optimizer.acquisition_function.TS import ThompsonSamplingAcquisitionFunction

ACQUISITION_FUNCTION_REGISTRY: Dict[str, Type[BaseAcquisitionFunction]] = {
    "qucb": QUCBAcquisitionFunction,
    "qkg": QKGAcquisitionFunction,
    "qmve": QMVEAcquisitionFunction,
    "qlbmve": QLBMVEAcquisitionFunction,
    "qpes": QPESAcquisitionFunction,
    "qsimpleregret": QSimpleRegretAcquisitionFunction,
    "qlognoisyei": QLogNoisyEI,
    "thompsonsampling": ThompsonSamplingAcquisitionFunction,
    "ts": ThompsonSamplingAcquisitionFunction,  # 简写
}


def get_acquisition_function(name: str, **kwargs) -> BaseAcquisitionFunction:
    if name not in ACQUISITION_FUNCTION_REGISTRY:
        available = ", ".join(list_available_acquisition_functions())
        raise ValueError(
            f"Unknown acquisition function name: {name}. Available functions: {available}"
        )
    return ACQUISITION_FUNCTION_REGISTRY[name](**kwargs)


def list_available_acquisition_functions() -> list[str]:
    """列出所有可用的采集函数名称（去重，优先显示简短名称）

    Returns:
        list[str]: 可用的采集函数名称列表（已排序，去重）
    """
    # 返回优先的简短名称或标准格式（过滤掉下划线格式的别名）
    preferred_names = [
        "qucb",
        "qkg",
        "qmve",
        "qlbmve",
        "qpes",
        "qsimpleregret",
        "qlognoisyei",
        "ts",
    ]
    return sorted(
        [name for name in preferred_names if name in ACQUISITION_FUNCTION_REGISTRY]
    )


__all__ = [
    "ACQUISITION_FUNCTION_REGISTRY",
    "get_acquisition_function",
    "list_available_acquisition_functions",
]
