"""
调仓周期管理 (rebalance_manager.py)
=====================================
本模块根据「调仓周期（天数）」将日频因子与收益率对齐为调仓期数据，供 IC 分析、分组与回测使用。

主要类：RebalancePeriodManager(factor, ret, rebalance_period)

主要方法：
- get_rebalance_dates()：从因子日期索引中等间隔取调仓日（每隔 rebalance_period 天）。
- align_factor_return_by_period()：对每个调仓期，取「调仓日前一交易日」的因子值，以及「从当前调仓日到下一调仓日」的累计收益率；返回 factor_periods（DataFrame，每行一个调仓期的截面因子）与 ret_periods（同结构的累计收益）。
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
        rebalance_period: int, 调仓周期（天）
        """
        self.factor = factor
        self.ret = ret
        self.rebalance_period = rebalance_period
        
    def get_rebalance_dates(self):
        """
        获取调仓日期
        
        Returns:
        --------
        list of datetime: 调仓日期列表
        """
        dates = self.factor.index
        rebalance_dates = dates[::self.rebalance_period]
        return rebalance_dates.tolist()
    
    def align_factor_return_by_period(self):
        """
        按调仓周期对齐因子和收益率
        
        返回每个调仓期的：
        - 因子值（调仓日前一天）
        - 期间累计收益率（调仓日到下一个调仓日）
        
        Returns:
        --------
        factor_periods: DataFrame, 每个调仓期的因子值
        ret_periods: DataFrame, 每个调仓期的累计收益率
        """
        rebalance_dates = self.get_rebalance_dates()
        
        factor_periods = []
        ret_periods = []
        period_dates = []
        
        for i in range(len(rebalance_dates) - 1):
            # 当前调仓日
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]
            
            # 调仓日前一天的因子值
            factor_date = self.factor.index[self.factor.index < current_date]
            if len(factor_date) == 0:
                continue
            factor_date = factor_date[-1]
            
            # 本期因子值
            factor_value = self.factor.loc[factor_date]
            
            # 本期收益率：持有期从调仓日次日到下一调仓日（不含调仓日当天，避免前瞻）
            # 调仓日在 T 日收盘执行，T 日收益不应计入持仓收益
            period_ret = self.ret.loc[current_date:next_date]
            if len(period_ret) == 0:
                continue
            if len(period_ret) >= 2:
                period_ret = period_ret.iloc[1:]
            else:
                period_ret = pd.DataFrame()
            if len(period_ret) > 0:
                cumulative_ret = (1 + period_ret).prod() - 1
            else:
                cumulative_ret = pd.Series(0.0, index=self.ret.columns)
            
            factor_periods.append(factor_value)
            ret_periods.append(cumulative_ret)
            period_dates.append(current_date)
        
        # 转换为 DataFrame
        factor_periods = pd.DataFrame(factor_periods, index=period_dates)
        ret_periods = pd.DataFrame(ret_periods, index=period_dates)
        
        return factor_periods, ret_periods