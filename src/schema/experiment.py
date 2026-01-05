from dataclasses import dataclass

from pydantic import BaseModel

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


@dataclass
class EnvironmentConfig:
    seed: int
    device: str
    dtype: str


@dataclass
class SearchSpaceConfig:
    search_space: dict


@dataclass
class ExecutionConfig:
    num_init_points: int
    budget: int
    num_replications: int
    checkpoint_file: str
    log_interval: int
    save_results: bool
    noise_level: float


class ExperimentConfig(BaseModel):
    target_function: str
    acquisition: str
    model: str
    environment: EnvironmentConfig
    search_space: SearchSpaceConfig
    execution: ExecutionConfig
