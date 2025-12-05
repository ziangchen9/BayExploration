from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import yaml
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from core.benchmark_function.base_function import BaseTestFunction


class Experiment:
    gp_history: List[SingleTaskGP] = []
    x_history: List[torch.Tensor] = []
    y_noised_history: List[torch.Tensor] = []
    y_real_history: List[torch.Tensor] = []

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        if isinstance(config, str | Path):
            self.config = self.load_config(config)
        else:
            self.config = config
        self.init_sample_x = None
        self.test_function = None

    @classmethod
    def load_config(cls, config: str | Path | Dict) -> Experiment:
        # TODO: 若config是字典则直接使用
        suffix = Path(config).suffix.lower()
        with open(config, "r") as f:
            if suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif suffix == ".json":
                config = json.load(f)
        return cls(config)

    @feild_validator("config")
    @classmethod
    def config_val(cls):
        """
        检查配置是否合法
        Returns:
        """
        pass

    def _execute_single_experiment(self, func: BaseTestFunction) -> None:
        # TODO: 实现单次实验的执行 [后期改为多进程可并行]
        # 测试函数与贝叶斯优化器解耦合，测试函数的配置覆盖优化器配置
        self.update_config()
        # 从 test function 处获得函数的维度，最小值，边界等信息
        self.test_function = func
        # 初始函数采样
        _ = self._init_sampling()
        # 拟合高斯

        # 采集函数
        pass
        """
        
        """

    def _init_sampling(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """initial sampling with sobol engine

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: initial sampling points, noised values, real values
        """
        sobol_engine = SobolEngine(dimension=self.test_function.dim, scramble=True)
        draw_number = self.config["init_scale"] * self.test_function.dim
        init_sample_x = sobol_engine.draw(draw_number).to(
            dtype=getattr(torch, self.config["numerical_precision"]),
            device=torch.device(self.config["device"]),
        )
        for i in range(self.test_function.dim):
            self.init_sample_x[:, i] = (
                init_sample_x[:, i]
                * (self.test_function.bounds[i][1] - self.test_function.bounds[i][0])
                + self.test_function.bounds[i][0]
            )
        self.init_sample_y = self.test_function(
            x=self.init_sample_x, noise_level=float(self.config["noise_level"])
        )
        self.init_real_y = self.test_function(x=init_sample_x, noise_level=0)
        return self.init_sample_x, self.init_sample_y, self.init_real_y

    def _fit_gaussian_process(self) -> None:
        self.gp = SingleTaskGP(
            train_X=self.init_sample_x,
            train_Y=self.init_sample_y,
            input_transform=self.config["Normalize"](d=self.test_function.dim),
            outcome_transform=self.config["Standardize"](m=1),
            mean_module=self.config["gaussian"]["mean_fn"],
            covar_module=self.config["gaussian"]["kernel_fn"],
        )
        # Fit
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.gp.likelihood, self.gp))

    def _update_history(self):
        pass
