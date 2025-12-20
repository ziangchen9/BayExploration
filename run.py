from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from core.benchmark_function.base_function import BaseTestFunction
from core.experiment.experiment import Experiment
from core.optimizer.optimizers import BaseBayesianOptimizer
from core.schema import ExperimentRecord


class Experiment:
    """实验运行器，用于运行多次重复实验并记录结果"""

    def __init__(
        self,
        experiment_record: ExperimentRecord,
        config: str | Path | dict[str, Any] | None = None,
        test_function: BaseTestFunction | None = None,
        optimizer: BaseBayesianOptimizer | None = None,
        test_function_factory: (
            Callable[[str, dict[str, Any]], BaseTestFunction] | None
        ) = None,
        optimizer_factory: (
            Callable[[str, dict[str, Any]], BaseBayesianOptimizer] | None
        ) = None,
    ):
        """
        初始化实验运行器

        Args:
            experiment_record: 实验记录对象
            config: 实验配置（文件路径或字典）
            test_function: 测试函数实例（如果提供，将用于所有重复实验）
            optimizer: 优化器实例（如果提供，将用于所有重复实验）
            test_function_factory: 测试函数工厂函数 (func_name, config) -> BaseTestFunction
            optimizer_factory: 优化器工厂函数 (alg_name, config) -> BaseBayesianOptimizer
        """
        self.experiment_record = experiment_record
        self.config = config
        self.test_function = test_function
        self.optimizer = optimizer
        self.test_function_factory = test_function_factory
        self.optimizer_factory = optimizer_factory

        # 设置实验开始时间
        self.experiment_record.start_time = datetime.now()

    def _load_config_if_needed(self) -> dict[str, Any]:
        """如果需要，加载配置"""
        if isinstance(self.config, dict):
            return self.config
        elif self.config is not None:
            return Experiment._load_config(self.config)
        else:
            return {}

    def _get_test_function(
        self, replication_id: int, config: dict[str, Any]
    ) -> BaseTestFunction:
        """获取测试函数实例"""
        if self.test_function is not None:
            return self.test_function

        if self.test_function_factory is not None:
            func_name = self.experiment_record.target_func_name
            return self.test_function_factory(func_name, config)

        raise ValueError(
            "测试函数未设置。请提供test_function参数或test_function_factory参数。"
        )

    def _get_optimizer(
        self, replication_id: int, alg_name: str, config: dict[str, Any]
    ) -> BaseBayesianOptimizer:
        """获取优化器实例"""
        if self.optimizer is not None:
            return self.optimizer

        if self.optimizer_factory is not None:
            return self.optimizer_factory(alg_name, config)

        raise ValueError("优化器未设置。请提供optimizer参数或optimizer_factory参数。")

    def run(self) -> ExperimentRecord:
        """
        运行所有重复实验

        Returns:
            更新后的实验记录
        """
        total_replications = self.experiment_record.total_replications
        total_iterations = self.experiment_record.total_iterations

        # 为每次重复实验运行
        for replication_id in range(total_replications):
            print(f"运行重复实验 {replication_id + 1}/{total_replications}")

            # 为每个算法运行
            for alg_name in self.experiment_record.bo_alg_name:
                print(f"  算法: {alg_name}")

                # 准备配置（为每次重复实验创建新的配置副本，并设置不同的随机种子）
                base_config = self._load_config_if_needed()
                if base_config:
                    config = base_config.copy()
                else:
                    # 如果没有提供config，创建一个基本配置
                    config = {
                        "device": "cpu",
                        "numerical_precision": "float64",
                        "num_init_points": 5,
                        "num_iterations": total_iterations,
                        "noise_level": 0.01,
                    }

                # 为每次重复实验设置不同的随机种子（如果原始配置中有seed）
                if "seed" in config:
                    config["seed"] = config["seed"] + replication_id * 1000
                else:
                    config["seed"] = 42 + replication_id * 1000

                # 创建实验实例
                experiment = Experiment(config)

                # 获取测试函数和优化器
                test_function = self._get_test_function(replication_id, config)
                optimizer = self._get_optimizer(replication_id, alg_name, config)

                # 运行实验
                history = experiment.run(
                    test_function=test_function, optimizer=optimizer
                )

                # 将历史记录添加到实验记录中，并设置replication_id
                for record in history:
                    record.replication_id = replication_id
                    # 添加到记录列表
                    self.experiment_record.records.append(record)
                    # 更新全局最优值
                    if (
                        self.experiment_record.global_best_value is None
                        or record.best_value < self.experiment_record.global_best_value
                    ):
                        self.experiment_record.global_best_value = record.best_value
                        self.experiment_record.global_best_position = (
                            record.best_position.clone()
                        )

        # 设置实验结束时间
        self.experiment_record.end_time = datetime.now()
        self.experiment_record.calculate_duration()

        print(f"实验完成，共运行 {total_replications} 次重复实验")
        print(f"全局最优值: {self.experiment_record.global_best_value}")

        return self.experiment_record


if __name__ == "__main__":
    experiment_record = ExperimentRecord(
        aqc_func_name="qUBC",
        target_func_name="booth",
        records=[],  # 初始记录列表为空
        start_time=datetime.now(),
        end_time=datetime.now(),  # 运行后会更新
        duration=0.0,  # 运行后会自动计算
        total_iterations=30,  # 每次重复实验的迭代次数
        total_replications=10,  # 重复实验次数
        global_best_value=float("inf"),  # 初始全局最优值
        global_best_position=torch.tensor([]),  # 初始全局最优点
        parameters={},  # 算法参数
        metadata={},  # 附加元数据
    )

    # 注意：实际使用时需要提供 test_function 和 optimizer，或者提供对应的工厂函数
    experiment_runner = Experiment(
        experiment_record=experiment_record,
        config=None,  # 可以传入配置路径或字典
        test_function=None,  # 传入测试函数实例
        optimizer=None,  # 传入优化器实例
        # test_function_factory=...,  # 或提供工厂函数
        # optimizer_factory=...,  # 或提供工厂函数
    )

    # 运行实验
    # result = experiment_runner.run()
    # print(f"实验完成，全局最优值: {result.global_best_value}")
    print("示例代码：请提供test_function和optimizer实例或对应的工厂函数")
