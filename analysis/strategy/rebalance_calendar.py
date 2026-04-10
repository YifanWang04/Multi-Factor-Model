"""
调仓日历模块 (rebalance_calendar.py)
===================================
权威实现：从因子日期序列中按交易日间隔选取调仓日，供全项目统一使用。

职责（SRP）：仅负责调仓日历生成，不涉及因子计算、权重分配或回测逻辑。

使用方式：
  from rebalance_calendar import get_rebalance_calendar
  dates = get_rebalance_calendar(factor_index, ret_index, rebalance_period_days)

被以下模块使用：
  - strategy_backtest._select_rebalance_dates  → 本模块导入
  - rebalance_manager.RebalancePeriodManager.get_rebalance_dates → 本模块导入
  - run_rebalance_day._select_rebalance_dates → 从 strategy_backtest 间接使用
"""

import pandas as pd


def get_rebalance_calendar(
    factor_index: pd.DatetimeIndex,
    ret_index: pd.DatetimeIndex,
    rebalance_period_days: int,
) -> list:
    """
    从因子日期序列中，选取交易日间隔 ≥ rebalance_period_days 的节点。

    即相邻调仓日之间至少相隔 rebalance_period_days 个交易日（按 ret_index 计数）。

    Parameters
    ----------
    factor_index : pd.DatetimeIndex
        因子数据的日期索引（可为非日频，如每 10 交易日一次）
    ret_index : pd.DatetimeIndex
        日频收益率/交易日的日期索引，用于正确计数交易日间隔
    rebalance_period_days : int
        调仓周期（交易日数），相邻调仓日之间至少相隔该交易日数

    Returns
    -------
    list
        调仓日列表（均为交易日）
    """
    dates = sorted(factor_index)
    if not dates:
        return []
    ret_sorted = ret_index.sort_values()

    selected = [dates[0]]
    last_selected = dates[0]
    for d in dates[1:]:
        n_trading_days = ((ret_sorted > last_selected) & (ret_sorted <= d)).sum()
        if n_trading_days >= rebalance_period_days:
            selected.append(d)
            last_selected = d

    return selected
