import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml
from botorch.models import SingleTaskGP
from torch.quasirandom import SobolEngine

from src.registry import get_test_function, get_mean_function, get_covariance_function, get_acquisition_function
from src.benchmark_function.base_function import BaseTestFunction
from src.optimizer.gaussian_model.base_guassian import BaseGPModel
from src.optimizer.gaussian_model.single_guassian import SingleTaskGPModel
from src.optimizer.optimizers import BaseBayesianOptimizer
from src.schema import CONFIG_SCHEMA, ExperimentRecord, OptimizationRecord


class Experiment:

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        try:
            if isinstance(config, dict):
                self.config = config
            else:
                self.config = self._load_config(config)
            self._validate_config()
        except Exception as e:
            raise ValueError(e)

        self.test_function: BaseTestFunction = self._set_up_target_function()
        self.optimizer: BaseBayesianOptimizer = self._set_up_optimizer()
        self.gp_model: BaseGPModel = self._set_up_gaussian_model()
        self.init_sample_x: torch.Tensor = self._init_sampling()
        self.init_sample_y: torch.Tensor = self._init_sampling()
        self.init_real_y: torch.Tensor = self._init_sampling()
        self.gp: SingleTaskGP = self._fit_gaussian_process()
        self.current_iteration = 0
        self.best_value: float = self._update_history()
        self.best_position: torch.Tensor = self._update_history()
        self.record = None

    @staticmethod
    def _load_config(config: str | Path) -> dict[str, Any]:
        suffix = Path(config).suffix.lower()
        with open(config) as f:
            if suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif suffix == ".json":
                config = json.load(f)
            else:
                raise ValueError
        return config

    def _validate_config(self) -> None:
        def _validate(
            current_config: dict, current_schema: dict, path: str = ""
        ) -> None:
            for key in current_schema:
                current_path = f"{path}.{key}" if path else key
                if key not in current_config:
                    raise ValueError(f"Missing required field: {current_path}")
                if isinstance(current_schema[key], dict) and current_schema[key]:
                    if not isinstance(current_config[key], dict):
                        raise ValueError(f"Field {current_path} should be a dictionary")
                    _validate(current_config[key], current_schema[key], current_path)

        _validate(self.config, CONFIG_SCHEMA)

    def _set_up_target_function(self) -> BaseTestFunction:
        """从配置中设置目标函数"""
        target_func_name = self.config["target_function"]["name"]

        noise_level = self.config.get("execution", {}).get("noise_level")
        env_config = self.config.get("environment", {})
        device = env_config.get("device")
        dtype_str = env_config.get("dtype")

        return get_test_function(
            name=target_func_name,
            noise_level=noise_level,
            device=device,
            dtype=dtype_str,
        )

    # TODO:待完善
    def _set_up_optimizer(self) -> None:
        mean_func_name = self.config["mean_function"]["name"]
        mean_func = get_mean_function(mean_func_name)()
        covar_func_name = self.config["covariance_function"]["name"]
        covar_func = get_covariance_function(covar_func_name)()
        gp_model = SingleTaskGPModel(mean_module=mean_func, covariance_module=covar_func, target_function_dim=self.test_function.dim)
        acq_func_name = self.config["acquisition_function"]["name"]
        acq_func = get_acquisition_function(acq_func_name)()
        

    def update_config(self) -> None:
        """根据测试函数更新配置"""
        if self.test_function is None:
            return

        # 从测试函数获取维度、边界等信息并更新配置
        if "search_space" not in self.config:
            self.config["search_space"] = {}

        # 使用测试函数的边界信息
        bounds = self.test_function.bound.tolist()
        self.config["search_space"]["bounds"] = bounds

    def _execute_single_experiment(
        self, func: BaseTestFunction, optimizer: BaseBayesianOptimizer | None = None
    ) -> None:
        """执行单次实验 [后期改为多进程可并行]

        Args:
            func: 测试函数实例
            optimizer: 优化器实例，如果为None则使用self.optimizer
        """
        # 测试函数与贝叶斯优化器解耦合，测试函数的配置覆盖优化器配置
        self.test_function = func
        self.update_config()

        if optimizer is not None:
            self.optimizer = optimizer

        if self.optimizer is None:
            raise ValueError(
                "优化器未设置，请在_execute_single_experiment中传入或先设置self.optimizer"
            )

        # 从 test function 处获得函数的维度，最小值，边界等信息
        # 初始函数采样
        self._init_sampling()

        # 拟合高斯过程
        self._fit_gaussian_process()

        # 初始化历史记录
        self.best_value = float(self.init_sample_y.min())
        best_idx = int(self.init_sample_y.argmin())
        self.best_position = self.init_sample_x[best_idx].clone()

        # 主优化循环
        num_iterations = self.config.get("num_iterations", 30)
        for iteration in range(num_iterations):
            self.current_iteration = iteration

            # 使用优化器获取下一个候选点
            bounds_tensor = self.test_function.bound.t().to(
                dtype=getattr(torch, self.config["numerical_precision"]),
                device=torch.device(self.config["device"]),
            )

            acq_optimizer_config = self.config.get("acq_optimizer", {})
            candidate, _ = self.optimizer.optimize(
                bounds=bounds_tensor,
                q=1,
                num_restarts=acq_optimizer_config.get("num_restarts", 10),
                raw_samples=acq_optimizer_config.get("raw_samples", 256),
                options={
                    "dtype": getattr(torch, self.config["numerical_precision"]),
                    "with_grad": True,
                },
            )

            # 评估候选点
            candidate_y = self.test_function(
                x=candidate, noise_level=float(self.config.get("noise_level", 0.01))
            )

            # 更新训练数据
            self.init_sample_x = torch.cat([self.init_sample_x, candidate], dim=0)
            self.init_sample_y = torch.cat([self.init_sample_y, candidate_y], dim=0)

            # 更新最优值
            current_value = float(candidate_y.item())
            if self.best_value is None or current_value < self.best_value:
                self.best_value = current_value
                self.best_position = candidate[0].clone()

            # 更新高斯过程模型
            self._fit_gaussian_process()

            # 更新历史记录
            self._update_history(iteration, candidate[0], current_value)

    def _init_sampling(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用Sobol序列进行初始采样

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 初始采样点、带噪声的值、真实值
        """
        sobol_engine = SobolEngine(dimension=self.test_function.dim, scramble=True)
        draw_number = int(self.config.get("num_init_points", 5))
        init_sample_x = sobol_engine.draw(draw_number).to(
            dtype=getattr(torch, self.config["numerical_precision"]),
            device=torch.device(self.config["device"]),
        )

        # 将[0,1]区间的采样点映射到实际边界
        bounds = self.test_function.bound
        for i in range(self.test_function.dim):
            init_sample_x[:, i] = (
                init_sample_x[:, i] * (bounds[i, 1] - bounds[i, 0]) + bounds[i, 0]
            )

        self.init_sample_x = init_sample_x
        self.init_sample_y = self.test_function(
            x=self.init_sample_x,
            noise_level=float(self.config.get("noise_level", 0.01)),
        )
        self.init_real_y = self.test_function(x=self.init_sample_x, noise_level=0.0)
        return self.init_sample_x, self.init_sample_y, self.init_real_y

    def _fit_gaussian_process(self) -> None:
        """拟合高斯过程模型"""
        # 优先使用optimizer中的model，如果没有则使用self.gp_model
        if (
            self.optimizer is not None
            and hasattr(self.optimizer, "model")
            and self.optimizer.model is not None
        ):
            gp_model = self.optimizer.model
            self.gp_model = gp_model
        elif self.gp_model is not None:
            gp_model = self.gp_model
        else:
            # 如果没有设置gp_model，则从配置创建（向后兼容）
            model_config = self.config.get("model", {})
            from core.optimizer.gaussian_model.single_guassian import (
                ConstantMeanSingleTaskGPModel,
            )

            covar_type = (
                model_config.get("covar_module", {}).get("type", "matern2_5").lower()
            )
            if covar_type == "matern":
                nu = model_config.get("covar_module", {}).get("nu", 2.5)
                if abs(nu - 1.5) < 0.1:
                    covar_type = "matern1_5"
                elif abs(nu - 2.5) < 0.1:
                    covar_type = "matern2_5"
                else:
                    covar_type = "matern2_5"  # 默认

            # 创建模型
            gp_model = ConstantMeanSingleTaskGPModel(covariance_module=covar_type)
            self.gp_model = gp_model

        # 拟合模型
        self.gp = gp_model.fit(self.init_sample_x, self.init_sample_y)

        # 更新optimizer的模型引用
        if self.optimizer is not None and hasattr(self.optimizer, "model"):
            self.optimizer.model.model = self.gp
            # 重新设置acquisition function以使用更新后的模型
            self.optimizer.acquisition_function.setup(self.optimizer.model)

    def _update_history(
        self, iteration: int, position: torch.Tensor, value: float
    ) -> None:
        """更新历史记录

        Args:
            iteration: 当前迭代次数
            position: 当前采样位置
            value: 当前函数值
        """
        record = OptimizationRecord(
            iteration_id=iteration,
            replication_id=0,  # 单次实验时默认为0
            current_value=value,
            current_position=position,
            best_value=self.best_value if self.best_value is not None else value,
            best_position=(
                self.best_position if self.best_position is not None else position
            ),
            start_time=datetime.now(),
            end_time=None,
            duration=0.0,
        )
        record.end_time = datetime.now()
        record.duration = (record.end_time - record.start_time).total_seconds()

        self.history.append(record)
        Experiment.RECORDS.append(record)

    def run(
        self,
        test_function: BaseTestFunction | None = None,
        optimizer: BaseBayesianOptimizer | None = None,
    ) -> list[OptimizationRecord]:
        """运行实验的主入口

        Args:
            test_function: 测试函数实例，如果为None则使用self.test_function
            optimizer: 优化器实例，如果为None则使用self.optimizer

        Returns:
            优化历史记录列表
        """
        if test_function is None:
            if self.test_function is None:
                raise ValueError(
                    "测试函数未设置，请在run方法中传入或先设置self.test_function"
                )
        else:
            self.test_function = test_function

        if optimizer is None:
            if self.optimizer is None:
                raise ValueError(
                    "优化器未设置，请在run方法中传入或先设置self.optimizer"
                )
        else:
            self.optimizer = optimizer

        # 执行单次实验
        self._execute_single_experiment(self.test_function, self.optimizer)

        return self.history

    def _record(self, record: OptimizationRecord) -> None:
        """record experiment record"""
        self.RECORDS.append(record)



    def _init_record(self) -> None:
        """init experiment record"""
        self.record = ExperimentRecord(
            aqc_func_name=self.config["acquisition_function"]["name"],
            target_func_name=self.config["target_function"]["name"],
            records=[],
            start_time=datetime.now(),
            end_time=None,
            duration=0.0,
            total_iterations=self.config["num_iterations"],
            total_replications=self.config["num_replications"],
            global_best_value=self.best_value,
            global_best_solution=self.best_position,
        )
