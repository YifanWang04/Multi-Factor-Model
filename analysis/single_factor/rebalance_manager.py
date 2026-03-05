"""
调仓周期管理 (rebalance_manager.py)
=====================================
本模块根据「调仓周期（日历天数）」将日频因子与收益率对齐为调仓期数据，供 IC 分析、分组与回测使用。

主要类：RebalancePeriodManager(factor, ret, rebalance_period)

主要方法：
- get_rebalance_dates()：从因子日期索引中选取日历间隔 ≥ rebalance_period 天的调仓日，
  与 strategy_backtest.py 的 _select_rebalance_dates 逻辑保持一致。
- align_factor_return_by_period()：对每个调仓期，取「调仓日当天」的因子值（EOD 可得），
  以及「从当前调仓日次日到下一调仓日」的累计收益率；返回 factor_periods 与 ret_periods。

时序约定：
  - 因子：调仓日 T 当天的 EOD 截面值（T 日收盘前已可获得）
  - 收益：从 T+1 开始持有到下一调仓日 T_next（含），即 (T, T_next] 区间
  - 交易：T 日收盘执行，T 日当日收益不计入持仓
"""

import pandas as pd
import numpy as np

class RebalancePeriodManager:
    """
    管理调仓周期，生成调仓日期
    """
    
    def __init__(self, factor, ret, rebalance_period):
        """
        Parameters:
        -----------
        factor: DataFrame, 因子数据（日频）
        ret: DataFrame, 收益率数据（日频）
        rebalance_period: int, 调仓周期（日历天数），与 strategy_backtest 保持一致
        """
        self.factor = factor
        self.ret = ret
        self.rebalance_period = rebalance_period
        
    def get_rebalance_dates(self):
        """
        从因子日期索引中选取日历间隔 ≥ rebalance_period 的调仓日。
        使用日历天数间隔（与 strategy_backtest._select_rebalance_dates 逻辑一致），
        而非简单的等间隔下标切片（[::N] 是交易日计数，非日历天数）。

        Returns:
        --------
        list of datetime: 调仓日期列表
        """
        dates = sorted(self.factor.index)
        if not dates:
            return []
        selected = [dates[0]]
        for d in dates[1:]:
            if (d - selected[-1]).days >= self.rebalance_period:
                selected.append(d)
        return selected
    
    def align_factor_return_by_period(self):
        """
        按调仓周期对齐因子和收益率。

        时序约定（无前瞻）：
          - 因子值：调仓日 T 当天的截面值（EOD 数据，T 日已可获得）
          - 累计收益率：(T, T_next] 区间，即 T+1 到下一调仓日（含）
          - T 当日收益不计入持仓（当日收盘执行调仓）

        Returns:
        --------
        factor_periods: DataFrame, index=调仓日, columns=股票代码, 值=T 日因子截面
        ret_periods: DataFrame, index=调仓日, columns=股票代码, 值=持仓期间累计收益率
        """
        rebalance_dates = self.get_rebalance_dates()
        
        factor_periods = []
        ret_periods = []
        period_dates = []
        
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]
            
            # 调仓日当天的因子值（EOD 可得，无前瞻）
            available = self.factor.index[self.factor.index <= current_date]
            if len(available) == 0:
                continue
            factor_date = available[-1]
            factor_value = self.factor.loc[factor_date]
            
            # 持仓收益：(current_date, next_date]，跳过 current_date 当天（current_date 可能不在 ret.index）
            mask = (self.ret.index > current_date) & (self.ret.index <= next_date)
            period_ret = self.ret.loc[mask]
            if len(period_ret) == 0:
                continue
            cumulative_ret = (1 + period_ret).prod() - 1
            
            factor_periods.append(factor_value)
            ret_periods.append(cumulative_ret)
            period_dates.append(current_date)
        
        factor_periods = pd.DataFrame(factor_periods, index=period_dates)
        ret_periods = pd.DataFrame(ret_periods, index=period_dates)
        
        return factor_periods, ret_periods