"""
策略复盘配置文件 (strategy_review_config.py)
============================================
完全自包含：无需前置运行 run_composite_factor，直接配置后运行 run_strategy_review 即可。

用法：修改本文件后运行 python analysis/strategy/run_strategy_review.py

── 1. 五个因子（输入因子编号）────────────────────────────────────────────────
从 factor_processed 目录读取 factor_alpha{编号:03d}_processed.xlsx
"""
SELECTED_FACTOR_INDICES = [32, 42, 20, 95, 73]

# ── 2. 复合因子方式 ─────────────────────────────────────────────────────────
# 可选：ic_m1, ic_m2, ic_m3_N5, ic_m3_N10, ic_m3_N20
#       ols_m1, ols_m2, ols_m3_M5, ols_m3_M10, ols_m3_M20
#       beta_m1, beta_m2, beta_m3_N5/N10/N20
#       rank_ic_m1, rank_ic_m2, rank_ic_m3_N5/N10/N20
#       rank_add, rank_mul
#       pca_pc1, pca_pc2, pca_pc3
COMPOSITE_FACTOR_SHEET = "beta_m1"

# 复合因子计算用的 N/M 窗口（需包含 COMPOSITE_FACTOR_SHEET 中的 N 或 M 值）
N_WINDOWS = [5, 10, 20]
M_WINDOWS = [5, 10, 20]

# ── 3. 策略参数 ─────────────────────────────────────────────────────────────
# 格式：{weight_method}_{N}G_Top{R}_P{D}d
# 例：max_return_5G_Top1_P10d、equal_10G_Top1_P20d
STRATEGY_PARAM = "mvo_10G_Top1_P30d"

# 复合因子调仓周期（交易日数），建议与策略 P{D}d 中的 D 一致
# None 时自动从 STRATEGY_PARAM 解析
REBALANCE_PERIOD: int | None = None

# ── 4. 实盘与基准 ───────────────────────────────────────────────────────────
LIVE_START_DATE: str | None = None
BENCHMARK_TICKER = "QQQ"
BULL_THRESHOLD = 0.03
BEAR_THRESHOLD = -0.03

# ── 5. 券商记录与参数敏感度 ─────────────────────────────────────────────────
BROKER_RECORDS_FILE: str | None = None
RUN_PARAM_SENSITIVITY = True
PARAM_GRID = [
    "max_return_5G_Top1_P5d",
    "max_return_5G_Top1_P7d",
    "max_return_5G_Top1_P10d",
    "max_return_5G_Top1_P15d",
    "max_return_5G_Top1_P20d",
    "equal_5G_Top1_P10d",
    "min_variance_5G_Top1_P10d",
]

# ── 6. 路径（None 时使用默认）────────────────────────────────────────────────
import os
from data.data_config import (
    FACTOR_PROCESSED_DIR as _DEFAULT_FACTOR_PROCESSED_DIR,
)

PROJECT_ROOT = r"D:\qqq"
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output")
FACTOR_DIR: str | None = None
OUTPUT_DIR: str | None = None


def get_factor_processed_dir() -> str:
    if FACTOR_DIR is not None and os.path.isdir(FACTOR_DIR):
        return os.path.abspath(FACTOR_DIR)
    return _DEFAULT_FACTOR_PROCESSED_DIR


def get_output_dir() -> str:
    from datetime import datetime
    if OUTPUT_DIR is not None:
        return os.path.abspath(OUTPUT_DIR)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.join(OUTPUT_BASE, f"strategy_review_{ts}")


def get_rebalance_period() -> int:
    """从 STRATEGY_PARAM 解析调仓周期，或使用 REBALANCE_PERIOD。"""
    if REBALANCE_PERIOD is not None:
        return REBALANCE_PERIOD
    import re
    m = re.search(r"_P(\d+)d$", STRATEGY_PARAM.strip())
    if m:
        return int(m.group(1))
    return 10


def get_selected_factor_names() -> list[str]:
    return [f"alpha{i:03d}" for i in SELECTED_FACTOR_INDICES]
