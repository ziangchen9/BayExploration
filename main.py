"""贝叶斯优化主入口文件"""

from pathlib import Path

from src.experiment.experiment import Experiment


def main():
    """运行贝叶斯优化实验"""
    # 使用配置文件运行实验
    config_path = Path(__file__).parent / "config" / "example_config.yaml"
    
    # 如果配置文件不存在，使用内联配置
    if not config_path.exists():
        config = {
            "target_function": {
                "name": "booth",
                "params_num": 2,
                "minimize": True,
                "global_optimal": [1.0, 3.0],
                "global_optimal_value": 0.0,
            },
            "model": {
                "gp_type": "SingleTaskGP",
                "mean_module": "constant",
                "covar_module": {
                    "type": "Matern-1.5",
                },
                "input_transform": "normalize",
                "outcome_transform": "standardize",
            },
            "acquisition": {
                "acq_function": "qucb",  # 注意：应该是 "qucb" 而不是 "qubc"
            },
            "environment": {
                "device": "cpu",
                "dtype": "float64",
                "seed": 42,
            },
            "search_space": {
                "bounds": {
                    "x1": [-10.0, 10.0],
                    "x2": [-10.0, 10.0],
                },
                "variable_types": {
                    "x1": "continuous",
                    "x2": "continuous",
                },
            },
            "execution": {
                "noise_level": 0.01,
                "num_init_points": 5,
                "budget": 30,
                "num_replications": 1,
                "checkpoint_file": "bo_checkpoint.json",
                "log_interval": 5,
                "save_results": "results.json",
            },
        }
    else:
        config = config_path
    
    print("=" * 60)
    print("开始运行贝叶斯优化实验")
    print("=" * 60)
    
    # 创建实验实例
    experiment = Experiment(config)
    
    # 运行实验
    result = experiment.run()
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("实验完成！结果摘要：")
    print("=" * 60)
    print(f"目标函数: {result.target_func_name}")
    print(f"采集函数: {result.aqc_func_name}")
    print(f"总迭代次数: {result.total_iterations}")
    print(f"重复实验次数: {result.total_replications}")
    print(f"全局最优值: {result.global_best_value:.6f}")
    print(f"全局最优解: {result.global_best_solution.tolist()}")
    print(f"实验耗时: {result.duration:.2f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    main()
