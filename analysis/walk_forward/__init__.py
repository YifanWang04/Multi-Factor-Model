"""
Walk-Forward Validation Package

滚动窗口验证系统，用于评估量化策略的稳健性。

主要模块：
- walk_forward_config: 配置参数
- rolling_data_processor: 防泄露的数据处理
- walk_forward_engine: 核心验证引擎
- walk_forward_analyzer: 结果分析和报告
- run_walk_forward: 主程序入口

使用方法：
    python analysis/walk_forward/run_walk_forward.py
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"

from . import walk_forward_config

# Lazy imports to avoid dependency issues
def _lazy_import():
    from .rolling_data_processor import process_factors_rolling
    from .walk_forward_engine import WalkForwardEngine
    from .walk_forward_analyzer import WalkForwardAnalyzer
    return process_factors_rolling, WalkForwardEngine, WalkForwardAnalyzer

__all__ = [
    'walk_forward_config',
]
