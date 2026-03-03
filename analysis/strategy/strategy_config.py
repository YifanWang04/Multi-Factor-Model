"""
策略构建配置文件 (strategy_config.py)
=====================================
定义复合因子多头策略网格回测的所有输入输出路径与参数。

运行前检查：
  COMPOSITE_FACTOR_SHEET  — 与 composite_factors.xlsx 中实际 sheet 名一致
  GROUP_NUMS              — 根据标的池数量设定（88 只股票建议 [5, 10]）
  REBALANCE_PERIODS       — 日历天数；系统从因子日期序列中取间隔 ≥ 该天数的节点
  TARGET_GROUP_RANKS      — 从最高组向下：1=最高，2=第二高，3=第三高
  WEIGHT_METHODS          — 选择要遍历的资产配置方式
"""

import os

PROJECT_ROOT = r"D:\qqq"

# ── 输入 ──────────────────────────────────────────────────────────────────────

# 由 run_composite_factor.py 生成的复合因子 Excel
COMPOSITE_FACTOR_FILE = os.path.join(
    PROJECT_ROOT, "output", "composite_factor_reports", "composite_factors.xlsx"
)

# 选定的复合因子方法（Excel sheet 名），对应用户选择的 ols_m3_M5
COMPOSITE_FACTOR_SHEET = "beta_m3_N10"

# 日频收益率数据（与单因子/复合因子流程一致）
PRICE_FILE = os.path.join(PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
RETURN_COLUMN = "Return"

# ── 输出 ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "strategy_reports")
OUTPUT_EXCEL_NAME = "strategy_backtest_report.xlsx"

# ── 网格遍历参数 ──────────────────────────────────────────────────────────────

# 分层数量：88 只股票建议 5 / 10 层；如需 15 / 20 层可自行追加
GROUP_NUMS = [5, 10]

# 调仓周期（日历天数）
# 注意：系统从复合因子已有的调仓日期序列中取样，实际最小精度 = 因子原生周期
# 典型因子原生周期为 10 交易日 ≈ 14 日历天，故 <14 的值等效于"每期必换"
# ⚠️ 建议：使用与 composite_config.REBALANCE_PERIOD 一致的值（10），或其整数倍（20, 30）
# 当前 composite_config.REBALANCE_PERIOD = 10 天
REBALANCE_PERIODS = [10, 20, 30, 60]

# 目标组排名（从高到低）：1=买最高分组，2=买第二高分组，3=买第三高分组
TARGET_GROUP_RANKS = [1, 2, 3]

# 资产配置方式
#   equal         : 等权配置
#   min_variance  : 最小方差组合（用得最多；回撤和波动率小）
#   mvo           : 马科维兹最优（最大化夏普比率；高风险高收益）
#   max_return    : 最大化预期收益
#   factor_score  : 因子值打分加权
WEIGHT_METHODS = ["equal", "min_variance", "mvo", "max_return", "factor_score"]

# ── 回测参数 ──────────────────────────────────────────────────────────────────

RISK_FREE_RATE = 0.02        # 年化无风险利率
TRANSACTION_COST = 0.001     # 单边交易成本（在每个持仓周期首日扣除）

# 组合优化方法（mvo / min_variance / max_return）使用的历史收益率回望窗口（交易日）
OPTIMIZATION_LOOKBACK = 252

# 单只标的最大权重约束（适用于 mvo / min_variance / max_return）
MAX_WEIGHT = 0.4
