from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

import torch
from pydantic import BaseModel, Field, model_validator

CONFIG_SCHEMA = {
    "target_function": {},
    "acquisition": {},
    "model": {},
    "environment": {
        "seed": {},
        "device": {},
        "dtype": {},
    },
    "search_space": {},
    "execution": {
        "num_init_points": {},
        "budget": {},
        "num_replications": {},
        "checkpoint_file": {},
        "log_interval": {},
        "save_results": {},
        "noise_level": {},
    },
}


class OptimizationRecord(BaseModel):
    """单次贝叶斯优化结果记录"""

    model_config = {"arbitrary_types_allowed": True}

    iteration_id: int = Field(ge=0, description="迭代次数编号")
    replication_id: int = Field(ge=0, description="重复次数编号")
    current_value: float = Field(description="当前迭代的函数值")
    current_position: torch.Tensor = Field(description="当前迭代的采样位置")
    best_value_by_now: float = Field(description="迄今为止的最优函数值")
    best_solution_by_now: torch.Tensor = Field(description="迄今为止的最解")
    start_time: datetime = Field(
        default_factory=datetime.now, description="记录创建时间戳"
    )
    end_time: Optional[datetime] = Field(default=None, description="记录结束时间戳")
    duration: float = Field(description="优化过程持续时间")

    @model_validator(mode="after")
    def calculate_duration(self) -> OptimizationRecord:
        """Auto count the duration of the optimization process"""
        if self.start_time and self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        return self

    def to_dict(self, tensor_to_list: bool = True) -> dict:
        # Turn Pydantic model into a dictionary
        data = self.model_dump()
        if tensor_to_list:
            # Convert torch.Tensor to list
            if isinstance(data.get("current_position"), torch.Tensor):
                data["current_position"] = data["current_position"].tolist()
            if isinstance(data.get("best_position"), torch.Tensor):
                data["best_position"] = data["best_position"].tolist()
        # Convert datetime to string
        if isinstance(data.get("start_time"), datetime):
            data["start_time"] = data["start_time"].isoformat()
        if isinstance(data.get("end_time"), datetime):
            data["end_time"] = data["end_time"].isoformat()
        return data


class ExperimentRecord(BaseModel):
    """All the records of an experiment"""

    model_config = {"arbitrary_types_allowed": True}

    aqc_func_name: str = Field(description="采集函数名称")
    target_func_name: str = Field(description="优化目标函数名称")
    records: List[List[OptimizationRecord]] = (
        Field(default_factory=list, description="优化记录列表"),
    )
    start_time: datetime = Field(
        default_factory=datetime.now, description="实验开始时间"
    )
    end_time: datetime = Field(default_factory=datetime.now, description="实验结束时间")
    duration: float = Field(default=0.0, description="实验持续时间")
    total_iterations: int = Field(default_factory=int, description="单轮迭代次数")
    total_replications: int = Field(default_factory=int, description="总重复轮数")
    global_best_value: float = Field(default_factory=float, description="全局最优值")
    global_best_solution: torch.Tensor = Field(
        default_factory=lambda: torch.tensor([]), description="全局最优解"
    )

    @model_validator(mode="after")
    def calculate_duration(self) -> ExperimentRecord:
        """Auto count the duration of the experiment"""
        if self.start_time and self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        return self

    def add_record(self, record: OptimizationRecord):
        """添加单条记录"""
        self.records.append(record)
        # Update the global best solution and value
        if (
            self.global_best_value is None
            or record.best_value_by_now < self.global_best_value
        ):
            self.global_best_value = record.best_value_by_now
            self.global_best_solution = record.best_solution_by_now.clone()

    def get_best_record(self) -> Optional[OptimizationRecord]:
        """获取最优记录"""
        if not self.records:
            return None
        return min(self.records, key=lambda x: x.best_value_by_now)

    def get_records_by_replication(
        self, replication_id: int
    ) -> list[OptimizationRecord]:
        """按重复实验ID获取记录"""
        if replication_id < len(self.records):
            return self.records[replication_id]
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "aqc_func_name": self.aqc_func_name,
            "target_func_name": self.target_func_name,
            "total_iterations": self.total_iterations,
            "total_replications": self.total_replications,
            "global_best_value": self.global_best_value,
            "global_best_solution": self.global_best_solution.tolist(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
        }
