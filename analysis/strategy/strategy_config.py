"""
策略回测配置文件 (strategy_config.py)
====================================
本文件为 run_strategy.py 的研究回测入口提供配置：输入路径、复合因子选择、
策略参数网格、交易成本/优化窗口，以及 detailed/rebalance-day 使用的 active
profile 默认值。

run_strategy.py 当前会展开的网格参数以 StrategyBacktester._all_combinations()
为准：
  GROUP_NUMS            — 分层数量，例如 [5, 10, 15, 20]
  REBALANCE_PERIODS     — 调仓周期，单位为交易日；每个周期优先读取
                           composite_factors_P{N}_*.xlsx
  TARGET_GROUP_RANKS    — 从最高因子组向下选择，1=最高组，2=第二高组
  WEIGHT_METHODS        — 资产配置方式：equal / factor_score / min_variance /
                           mvo / max_return
  MAX_WEIGHT_GRID       — 单标的权重上限网格；仅对 min_variance、mvo、
                           max_return 三类优化器生效，equal/factor_score
                           使用 MAX_WEIGHT 标量记录默认值
  EXIT_POLICY_GRID      — 出口策略网格：fixed_rebalance / dynamic_tp_sl
  TP_BASE_GRID          — dynamic_tp_sl 下的止盈基准网格；fixed_rebalance
                           不使用该网格
  SL_BASE_GRID          — dynamic_tp_sl 下的止损基准网格；fixed_rebalance
                           不使用该网格

组合数口径：
  fixed_rebalance:
    GROUP_NUMS × REBALANCE_PERIODS × TARGET_GROUP_RANKS × WEIGHT_METHODS
    × MAX_WEIGHT_GRID(仅优化器方法)
  dynamic_tp_sl:
    上述组合 × TP_BASE_GRID × SL_BASE_GRID

不属于 run_strategy.py 网格、但会影响结果的重要配置：
  STRATEGY_RESEARCH_FACTOR_INDICES / STRATEGY_RESEARCH_COMPOSITE_SHEET
    仅用于研究覆盖 active profile 的因子集合和复合因子 sheet；不要用于实盘流程。
  STRATEGY_SELECTED_FACTOR_INDICES / COMPOSITE_FACTOR_SHEET / COMPOSITE_FACTOR_FILE
    决定读取哪个复合因子工作簿和 sheet。
  TP_SL_PROBABILITY
    dynamic_tp_sl 阈值缩放参数，目前是单一标量，不会被网格展开。
  RISK_FREE_RATE / TRANSACTION_COST / OPTIMIZATION_LOOKBACK
    分别影响指标计算、持仓期首日成本扣减和组合优化的历史收益窗口。

时间对齐约定：
  因子信号使用调仓日 T 的收盘截面；持仓收益区间为 (T, T_next]；
  REBALANCE_PERIODS 表示交易日数量，不是自然日。
"""

import os
from qqq_core.paths import ProjectPaths

PROJECT_ROOT = str(ProjectPaths.from_env().root)

# 日频收益率文件：根据 data_config 按 offset 分子目录
from data.data_config import (
    PRICE_FILE,
    STRATEGY_REPORTS_DIR,
    COMPOSITE_FACTOR_OUTPUT_DIR,
    DATA_START_OFFSET_DAYS,
    _offset_dir_suffix,
)
from qqq_config.strategy_profiles import get_active_profile, parse_factor_indices_csv

# ── Active profile 派生配置 ───────────────────────────────────────────────────

ACTIVE_PROFILE = get_active_profile()
ACTIVE_STRATEGY_PROFILE = ACTIVE_PROFILE.name

# Research-only overrides for direct runs of run_strategy.py before a profile is finalized.
# Only set the factor group and composite sheet here; strategy grids stay below.
# Leave values as None to use the active profile.
# Reset them to None before live/rebalance-day runs.
# STRATEGY_RESEARCH_FACTOR_INDICES = [95, 99, 27, 46, 19]          # Example: [95, 99, 27, 46, 19]
# STRATEGY_RESEARCH_COMPOSITE_SHEET = "ic_m3_N10"         # Example: "rank_ic_m3_N20"
# STRATEGY_RESEARCH_FACTOR_INDICES = [95, 101, 62, 65, 32]          # Example: [95, 99, 27, 46, 19]
# STRATEGY_RESEARCH_COMPOSITE_SHEET = "ic_m3_N20"       # Example: "rank_ic_m3_N20"
STRATEGY_RESEARCH_FACTOR_INDICES = [23, 60, 20, 10, 51]  #July 29, 2026
STRATEGY_RESEARCH_COMPOSITE_SHEET = "ic_m1"             #July 29, 2026

def _coerce_factor_indices(value):
    if value is None:
        return []
    if isinstance(value, str):
        return list(parse_factor_indices_csv(value))
    return [int(i) for i in value]


COMPOSITE_FACTOR_SHEET = (
    STRATEGY_RESEARCH_COMPOSITE_SHEET or ACTIVE_PROFILE.composite_sheet
)
STRATEGY_PARAM = ACTIVE_PROFILE.strategy_param
# Profile data_download_start_date 只控制行情下载起点，不固定交易日历相位。
DATA_DOWNLOAD_START_DATE = ACTIVE_PROFILE.data_download_start_date
REBALANCE_ANCHOR_DATE = None
REBALANCE_INTERVAL_WEEKS = ACTIVE_PROFILE.rebalance_interval_weeks
REBALANCE_WEEKDAY = ACTIVE_PROFILE.rebalance_weekday
REBALANCE_WEEK_ANCHOR_DATE = ACTIVE_PROFILE.rebalance_week_anchor_date
FIXED_WEEK_REBALANCE_PERIOD = (
    ACTIVE_PROFILE.rebalance_period
    if ACTIVE_PROFILE.uses_fixed_week_rebalance
    else None
)

# 切换策略时修改 qqq_config/strategy_profiles.py 的 ACTIVE_STRATEGY_PROFILE，
# 或临时设置环境变量 QQQ_STRATEGY_PROFILE；不要在本文件硬编码因子列表。
_research_factor_indices = _coerce_factor_indices(STRATEGY_RESEARCH_FACTOR_INDICES)
STRATEGY_SELECTED_FACTOR_INDICES = (
    _research_factor_indices or list(ACTIVE_PROFILE.factor_indices)
)

STRATEGY_SELECTED_FACTOR_NAMES = [
    f"alpha{i:03d}" for i in STRATEGY_SELECTED_FACTOR_INDICES
]

# 复合因子文件名后缀（如 f95-101-62-65-32）
def build_strategy_factor_suffix(factor_indices=None):
    """基于因子编号列表生成简短后缀，如 f95-101-62-65-32。"""
    if factor_indices is None:
        factor_indices = STRATEGY_SELECTED_FACTOR_INDICES
    return "f" + "-".join(str(int(i)) for i in factor_indices)

# 复合因子 Excel 文件路径（按 offset 分子目录）
def _composite_factor_candidates(period: int | None = None) -> list[str]:
    """
    根据 STRATEGY_SELECTED_FACTOR_INDICES 构建复合因子文件候选路径。
    period 非空时优先使用 period-specific 文件。
    """
    suffix = build_strategy_factor_suffix()
    if period is not None:
        names = [f"composite_factors_P{int(period)}_{suffix}.xlsx"]
    else:
        names = [f"composite_factors_{suffix}.xlsx"]

    offset_dir = COMPOSITE_FACTOR_OUTPUT_DIR  # 来自 data_config，已按 offset 分子目录
    candidates = [os.path.join(offset_dir, name) for name in names]

    legacy_offset_dir = os.path.join(
        PROJECT_ROOT, "output", f"composite_factor_reports{_offset_dir_suffix()}"
    )
    candidates.extend(os.path.join(legacy_offset_dir, name) for name in names)
    if DATA_START_OFFSET_DAYS == 0:
        base_dir = os.path.join(PROJECT_ROOT, "output", "composite_factor_reports")
        candidates.extend(os.path.join(base_dir, name) for name in names)
    return candidates


def get_composite_factor_file(period: int | None = None) -> str:
    """
    根据 STRATEGY_SELECTED_FACTOR_INDICES 构建复合因子文件路径。
    优先使用 offset 子目录；offset 非 0 时不回退到基线目录。
    """
    candidates = _composite_factor_candidates(period)
    for path in candidates:
        if os.path.isfile(path):
            return path
    # 均不存在时返回 offset 路径（让调用方报 FileNotFoundError）
    return candidates[0]

COMPOSITE_FACTOR_FILE = get_composite_factor_file()

RETURN_COLUMN = "Return"

OUTPUT_DIR = STRATEGY_REPORTS_DIR
OUTPUT_EXCEL_NAME = "strategy_backtest_report.xlsx"

# Excel 报表中的 strategy_statistics 会保留全部策略组合；日收益和累计收益
# 宽表默认只导出按 Sharpe 排名前 N 的策略，避免 900+ 组合时 openpyxl 写入
# 百万级单元格导致第 5 步非常慢。设为 None 或 0 可恢复导出全部策略时间序列。
REPORT_TIMESERIES_TOP_N = 10

# ── 网格遍历参数 ──────────────────────────────────────────────────────────────

# 分层数量：88 只股票建议 5 / 10 层；如需 15 / 20 层可自行追加
GROUP_NUMS = [5, 10]
# GROUP_NUMS = [5, 10, 15]

# 调仓周期（交易日数）：相邻调仓日之间至少相隔 N 个交易日
# 每个周期优先读取 matching 的 composite_factors_P{N}_*.xlsx。
# 调仓日从可用复合因子/收益数据的首日自然生成。
# REBALANCE_PERIODS = [10, 20]
REBALANCE_PERIODS = [5, 10, 20]

COMPOSITE_FACTOR_FILES_BY_PERIOD = {
    int(period): get_composite_factor_file(int(period))
    for period in REBALANCE_PERIODS
}

# 目标组排名（从高到低）：1=买最高分组，2=买第二高分组，3=买第三高分组
TARGET_GROUP_RANKS = [1, 2]
# TARGET_GROUP_RANKS = [1]

# 资产配置方式
#   equal         : 等权配置
#   min_variance  : 最小方差组合（用得最多；回撤和波动率小）
#   mvo           : 马科维兹最优（最大化夏普比率；高风险高收益）
#   max_return    : 最大化预期收益
#   factor_score  : 因子值打分加权
WEIGHT_METHODS = ["equal", "min_variance", "mvo", "max_return", "factor_score"]
#WEIGHT_METHODS = ["max_return"]

# Exit policy grid:
# - fixed_rebalance keeps the historical behavior and does not use TP/SL grids.
# - dynamic_tp_sl scans TP_BASE_GRID x SL_BASE_GRID using Adj Close exits.
# EXIT_POLICY_GRID = ["fixed_rebalance"]
EXIT_POLICY_GRID = ["fixed_rebalance", "dynamic_tp_sl"]
TP_BASE_GRID = [0.7, 0.8, 0.9]
SL_BASE_GRID = [0.5, 0.65, 0.8]
TP_SL_PROBABILITY = ACTIVE_PROFILE.tp_sl_probability

# Active-profile defaults used by detailed backtest and rebalance-day reports.
EXIT_POLICY = ACTIVE_PROFILE.exit_policy
TP_BASE = ACTIVE_PROFILE.tp_base
SL_BASE = ACTIVE_PROFILE.sl_base

# ── 回测参数 ──────────────────────────────────────────────────────────────────

RISK_FREE_RATE = 0.02        # 年化无风险利率
TRANSACTION_COST = 0.001     # 单边交易成本（在每个持仓周期首日扣除）

# 组合优化方法（mvo / min_variance / max_return）使用的历史收益率回望窗口（交易日）
OPTIMIZATION_LOOKBACK = 252

# 单只标的最大权重约束（适用于 mvo / min_variance / max_return）
# MAX_WEIGHT 是详细回测/调仓日使用的 active profile 默认标量；run_strategy.py 如需研究扫描，
# 使用 MAX_WEIGHT_GRID 展开网格，避免把 list 传入优化器。
MAX_WEIGHT = ACTIVE_PROFILE.max_weight
# MAX_WEIGHT_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
MAX_WEIGHT_GRID = [0.4, 0.6]
