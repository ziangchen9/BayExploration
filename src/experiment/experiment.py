import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import torch
import yaml
from botorch.utils.sampling import draw_sobol_samples

from src.optimizer.acquisition_function.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from src.benchmark_function.base_function import BaseTestFunction
from src.optimizer.gaussian_model.base_guassian import BaseGPModel
from src.registry import (
    get_acquisition_function,
    get_covariance_function,
    get_gaussian_model,
    get_mean_function,
    get_target_function,
)
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

        self.budget = self.config.get("execution").get("budget")
        self.num_replications = self.config.get("execution").get("num_replications", 1)
        self.num_init_points = self.config.get("execution").get("num_init_points", 5)
        self.noise_level = self.config.get("execution").get("noise_level", 0.01)
        
        # 设置随机种子和设备
        seed = self.config.get("environment", {}).get("seed", 42)
        device_str = self.config.get("environment", {}).get("device", "cpu")
        dtype_str = self.config.get("environment", {}).get("dtype", "float64")
        
        self.device = torch.device(device_str)
        self.dtype = getattr(torch, dtype_str)
        torch.manual_seed(seed)
        
        # 先设置目标函数（用于获取维度等信息）
        self.target_function: BaseTestFunction = self._set_up_target_function()
        
        # 获取搜索空间边界（需要先有目标函数）
        self.bounds = self._get_bounds()
        self.acquisition_function: BaseAcquisitionFunction = (
            self.get_acquisition_function()
        )

        self.experiment_record: ExperimentRecord = None

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
        """设置目标函数"""
        func_name = self.config["target_function"]["name"]
        func_kwargs = {
            "noise_level": self.noise_level,
        }
        return get_target_function(func_name, **func_kwargs)
    
    def _get_bounds(self) -> torch.Tensor:
        """从配置中获取搜索空间边界，返回 [2, dim] 格式的张量"""
        search_space = self.config.get("search_space", {})
        bounds_dict = search_space.get("bounds", {})
        
        # 如果没有bounds，使用目标函数的默认边界
        if not bounds_dict:
            bounds = self.target_function.bound
            # bound属性返回的是 [2, dim] 格式，第一行是下界，第二行是上界
            # 但我们需要确保格式正确
            if bounds.shape[0] == 2 and bounds.shape[1] == self.target_function.dim:
                return bounds.to(self.device).to(self.dtype)
            else:
                # 如果格式不对，转置
                return bounds.T.to(self.device).to(self.dtype)
        
        # 从配置中构建边界张量
        dim = self.target_function.dim
        
        # 构建边界列表：[[lower1, lower2, ...], [upper1, upper2, ...]]
        lower_bounds = []
        upper_bounds = []
        for i in range(1, dim + 1):
            key = f"x{i}"
            if key in bounds_dict:
                lower_bounds.append(bounds_dict[key][0])
                upper_bounds.append(bounds_dict[key][1])
            else:
                raise ValueError(f"Missing bounds for {key}")
        
        # 转换为 [2, dim] 格式的边界张量
        bounds_tensor = torch.tensor(
            [lower_bounds, upper_bounds], 
            dtype=self.dtype, 
            device=self.device
        )
        return bounds_tensor
    
    def get_gaussian_model(self) -> BaseGPModel:
        """获取高斯过程模型"""
        model_config = self.config.get("model", {})
        mean_name = model_config.get("mean_module", "constant")
        covar_config = model_config.get("covar_module", {})
        covar_type = covar_config.get("type", "Matern-1.5").lower().replace("-", "_").replace("_", "")
        # 处理 "matern1.5" -> "matern1_5"
        if "matern" in covar_type and "." in covar_type:
            covar_type = covar_type.replace(".", "_")
        
        mean_module = get_mean_function(mean_name)
        covariance_module = get_covariance_function(covar_type)
        
        return get_gaussian_model(
            name="singletaskgp",
            mean_module=mean_module,
            covariance_module=covariance_module,
            target_function_dim=self.target_function.dim
        )
    
    def get_acquisition_function(self) -> BaseAcquisitionFunction:
        """获取采集函数"""
        acq_config = self.config.get("acquisition", {})
        acq_name = acq_config.get("acq_function", "qucb").lower()
        return get_acquisition_function(acq_name)
    
    def _initialize_points(self, n_points: int, seed: int = None) -> torch.Tensor:
        """使用Sobol采样初始化点"""
        if seed is not None:
            torch.manual_seed(seed)
        bounds = self.bounds
        X_init = draw_sobol_samples(bounds=bounds, n=n_points, q=1).squeeze(1)
        return X_init.to(self.device).to(self.dtype)
    
    def _execute_single_optimization(self, seed: int = 0) -> list[OptimizationRecord]:
        """执行单次贝叶斯优化"""
        records = []
        
        # 初始化点
        X = self._initialize_points(self.num_init_points, seed=seed)
        # 调用目标函数（使用__call__方法，它会自动处理噪声）
        Y = self.target_function(X, noise_level=self.noise_level)
        
        # 确保Y的形状正确 [n, 1]
        if Y.dim() == 0:
            Y = Y.unsqueeze(0).unsqueeze(-1)
        elif Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        # 如果Y是 [n, m] 形状，取第一列
        if Y.shape[1] > 1:
            Y = Y[:, 0:1]
        
        # 初始化最优值
        best_value = Y.min().item()
        best_idx = Y.argmin()
        best_solution = X[best_idx].clone()
        
        # 记录初始点
        for i in range(self.num_init_points):
            y_val = Y[i, 0].item()  # 取 [i, 0] 元素
            record = OptimizationRecord(
                iteration_id=i,
                replication_id=0,
                current_value=y_val,
                current_position=X[i].clone(),
                best_value_by_now=best_value,
                best_solution_by_now=best_solution.clone(),
                start_time=datetime.now(),
                duration=0.0
            )
            record.end_time = datetime.now()
            records.append(record)
        
        # 贝叶斯优化循环
        for iteration in range(self.num_init_points, self.budget):
            # 拟合GP模型
            gp_model = self.get_gaussian_model()
            gp_model.fit(target_function_dim=self.target_function.dim, x=X, y=Y)
            
            # 设置并优化采集函数
            self.acquisition_function.setup()
            self.acquisition_function.OPTIMIZE_KWARGS["bounds"] = self.bounds
            
            # 优化采集函数获取下一个点
            result = self.acquisition_function.optimize(
                seed=seed + iteration, gp_model=gp_model
            )
            
            # 处理返回结果（可能是元组或单个值）
            if isinstance(result, tuple):
                candidate, _ = result
            else:
                candidate = result
            
            # 确保candidate的形状正确
            if candidate.dim() > 2:
                candidate = candidate.squeeze(0)
            if candidate.dim() == 1:
                candidate = candidate.unsqueeze(0)
            
            # 评估新点（使用__call__方法，它会自动处理噪声）
            y_new = self.target_function(candidate, noise_level=self.noise_level)
            
            # 确保y_new的形状正确 [1, 1]
            if y_new.dim() == 0:
                y_new = y_new.unsqueeze(0).unsqueeze(-1)
            elif y_new.dim() == 1:
                y_new = y_new.unsqueeze(-1)
            # 如果y_new是 [1, m] 形状，取第一列
            if y_new.shape[1] > 1:
                y_new = y_new[:, 0:1]
            
            # 更新数据
            X = torch.cat([X, candidate], dim=0)
            Y = torch.cat([Y, y_new], dim=0)
            
            # 更新最优值
            y_val = y_new[0, 0].item()  # 取 [0, 0] 元素
            if y_val < best_value:
                best_value = y_val
                best_solution = candidate.clone()
            
            # 记录
            record = OptimizationRecord(
                iteration_id=iteration,
                replication_id=0,
                current_value=y_val,
                current_position=candidate.clone(),
                best_value_by_now=best_value,
                best_solution_by_now=best_solution.clone(),
                start_time=datetime.now(),
                duration=0.0
            )
            record.end_time = datetime.now()
            records.append(record)
            
            if (iteration + 1) % 5 == 0:
                print(f"迭代 {iteration + 1}/{self.budget}, 当前最优值: {best_value:.6f}")
        
        return records
    
    def run(self) -> ExperimentRecord:
        """运行完整的实验"""
        start_time = datetime.now()
        
        # 初始化实验记录
        acq_name = self.config.get("acquisition", {}).get("acq_function", "qucb")
        target_name = self.config.get("target_function", {}).get("name", "booth")
        
        self.experiment_record = ExperimentRecord(
            aqc_func_name=acq_name,
            target_func_name=target_name,
            records=[],
            start_time=start_time,
            total_iterations=self.budget,
            total_replications=self.num_replications,
            global_best_value=float('inf'),
            global_best_solution=torch.zeros(self.target_function.dim, dtype=self.dtype, device=self.device)
        )
        
        # 运行多次重复实验
        for rep_id in range(self.num_replications):
            print(f"\n运行重复实验 {rep_id + 1}/{self.num_replications}")
            seed = self.config.get("environment", {}).get("seed", 42) + rep_id * 1000
            torch.manual_seed(seed)
            
            records = self._execute_single_optimization(seed=seed)
            
            # 更新记录中的replication_id并添加到records列表
            rep_records = []
            for record in records:
                record.replication_id = rep_id
                rep_records.append(record)
                # 更新全局最优值
                if record.best_value_by_now < self.experiment_record.global_best_value:
                    self.experiment_record.global_best_value = record.best_value_by_now
                    self.experiment_record.global_best_solution = record.best_solution_by_now.clone()
            
            self.experiment_record.records.append(rep_records)
        
        self.experiment_record.end_time = datetime.now()
        self.experiment_record.calculate_duration()
        
        print(f"\n实验完成！")
        print(f"全局最优值: {self.experiment_record.global_best_value:.6f}")
        print(f"全局最优解: {self.experiment_record.global_best_solution.tolist()}")
        
        return self.experiment_record
