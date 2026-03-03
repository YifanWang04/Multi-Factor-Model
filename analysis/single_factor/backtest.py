"""
多空/多头/空头回测模块 (backtest.py)
=====================================
本模块提供基于分组收益率的回测引擎，不直接依赖调仓明细，适用于单因子测试中的多空、纯多头、纯空头策略。

主要类：
- LongShortBacktestEnhanced：多空回测。给定多头组号与空头组号，计算 (多头收益 - 空头收益) 并扣除双边交易成本，输出净值与收益率序列；支持 run_multiple_pairs 批量多组多空组合。
- LongOnlyBacktest：纯多头。对指定组做多，扣除单边交易成本，可对全部组 run_all_groups 得到各组净值与收益。
- ShortOnlyBacktest：纯空头。对指定组做空（收益 = -该组收益），扣除交易成本。

输入：group_returns (DataFrame，index=日期，columns=组号)，以及可选的 transaction_cost。
"""

import pandas as pd
import numpy as np


class LongShortBacktestEnhanced:
    """
    多空回测 - 增强版
    支持多种多空组合，考虑交易成本
    """

    def __init__(self, group_returns, transaction_cost=0.001):
        """
        Parameters:
        -----------
        group_returns: DataFrame, 每组的收益率
        transaction_cost: float, 交易成本（单边）
        """
        self.group_returns = group_returns
        self.transaction_cost = transaction_cost

    def run(self, long_group, short_group):
        """
        运行多空回测
        
        Parameters:
        -----------
        long_group: int, 多头组号
        short_group: int, 空头组号
        
        Returns:
        --------
        nav: Series, 净值曲线
        returns: Series, 收益率序列
        """
        long_ret = self.group_returns[long_group]
        short_ret = self.group_returns[short_group]
        ls_returns = long_ret - short_ret
        ls_returns = ls_returns - 2 * self.transaction_cost
        nav = (1 + ls_returns).cumprod()
        return nav, ls_returns

    def run_multiple_pairs(self, long_short_pairs):
        """
        运行多个多空组合
        
        Parameters:
        -----------
        long_short_pairs: list of tuples, [(long_group, short_group, name), ...]
        
        Returns:
        --------
        dict: {name: {'nav': nav, 'returns': returns}}
        """
        results = {}
        for long_group, short_group, name in long_short_pairs:
            nav, returns = self.run(long_group, short_group)
            results[name] = {'nav': nav, 'returns': returns}
        return results


class LongOnlyBacktest:
    """
    纯多头回测
    """
    
    def __init__(self, group_returns, transaction_cost=0.001):
        """
        Parameters:
        -----------
        group_returns: DataFrame, 每组的收益率
        transaction_cost: float, 交易成本（单边）
        """
        self.group_returns = group_returns
        self.transaction_cost = transaction_cost
    
    def run(self, group_num):
        """
        运行纯多头回测
        
        Parameters:
        -----------
        group_num: int, 组号
        
        Returns:
        --------
        nav: Series, 净值曲线
        returns: Series, 收益率序列
        """
        # 获取该组收益率
        returns = self.group_returns[group_num]
        
        # 扣除交易成本
        returns = returns - self.transaction_cost
        
        # 计算净值
        nav = (1 + returns).cumprod()
        
        return nav, returns
    
    def run_all_groups(self):
        """
        运行所有组的纯多头回测
        
        Returns:
        --------
        dict: {group_num: {'nav': nav, 'returns': returns}}
        """
        results = {}
        
        for group_num in self.group_returns.columns:
            nav, returns = self.run(group_num)
            results[group_num] = {
                'nav': nav,
                'returns': returns
            }
        
        return results


class ShortOnlyBacktest:
    """
    纯空头回测：对每一组做空（收益 = -该组收益）。
    Group 10 = 因子值最高组。对动量类因子该组表现最好，做空 Group 10 会亏损属正常；
    做空 Group 1（因子值最低）在动量因子下可能盈利。
    """
    
    def __init__(self, group_returns, transaction_cost=0.001):
        """
        Parameters:
        -----------
        group_returns: DataFrame, 每组的收益率
        transaction_cost: float, 交易成本（单边）
        """
        self.group_returns = group_returns
        self.transaction_cost = transaction_cost
    
    def run(self, group_num):
        """
        运行纯空头回测（做空该组股票）
        
        Parameters:
        -----------
        group_num: int, 组号
        
        Returns:
        --------
        nav: Series, 净值曲线
        returns: Series, 收益率序列
        """
        # 获取该组收益率（取负：做空收益 = 卖出价-买入价 对应 -多头收益）
        returns = -self.group_returns[group_num]
        # #region agent log
        if group_num == 1:
            try:
                _d = self.group_returns.index[0]
                r_long = float(self.group_returns.loc[_d, group_num])
                r_short = float(returns.loc[_d])
                with open(r"d:\qqq\.cursor\debug.log", "a", encoding="utf-8") as _f:
                    _f.write('{"id":"short_formula","location":"ShortOnlyBacktest.run","message":"Short return = -long","data":{"group":1,"R_long":r_long,"R_short":r_short,"neg_ok":abs(r_short+r_long)<1e-6},"hypothesisId":"H3"}\n')
            except Exception:
                pass
        # #endregion
        # 扣除交易成本
        returns = returns - self.transaction_cost
        
        # 计算净值
        nav = (1 + returns).cumprod()
        
        return nav, returns
    
    def run_all_groups(self):
        """
        运行所有组的纯空头回测
        
        Returns:
        --------
        dict: {group_num: {'nav': nav, 'returns': returns}}
        """
        results = {}
        
        for group_num in self.group_returns.columns:
            nav, returns = self.run(group_num)
            results[group_num] = {
                'nav': nav,
                'returns': returns
            }
        
        return results