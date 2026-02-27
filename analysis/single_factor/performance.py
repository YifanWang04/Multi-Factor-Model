"""
绩效分析模块 (performance.py)
=====================================
本模块根据净值曲线与收益率序列计算常用绩效指标，供多空/多头/空头回测结果评估使用。

主要类：PerformanceAnalyzer(nav, returns, rf=0.02, periods_per_year=252)

计算的指标包括：
- 收益：总收益、年化收益
- 风险：年化波动率、最大回撤、卡玛比率
- 风险调整：夏普比率（扣除无风险利率）
- 交易统计：胜率、盈亏比
- 时间范围：起始/结束日期、总期数

方法：
- calculate_metrics()：返回上述指标字典。
- get_monthly_returns()：重采样为月收益（兼容 pandas 2.2+ 的 'ME' 与旧版 'M'）。
"""

import numpy as np
import pandas as pd

class PerformanceAnalyzer:
    """
    绩效分析器 - 增强版
    计算详细的绩效指标
    """

    def __init__(self, nav, returns, rf=0.02, periods_per_year=252):
        """
        Parameters:
        -----------
        nav: Series, 净值曲线
        returns: Series, 收益率序列
        rf: float, 无风险利率（年化）
        periods_per_year: int, 每年的期数（日频=252，周频=52，月频=12）
        """
        self.nav = nav
        self.returns = returns
        self.rf = rf
        self.periods_per_year = periods_per_year

    def calculate_metrics(self):
        """
        计算所有绩效指标
        
        Returns:
        --------
        dict: 绩效指标字典
        """
        if len(self.nav) == 0 or len(self.returns) == 0:
            return self._empty_metrics()
        
        metrics = {}
        
        # 基础指标
        metrics['Total_Return'] = self.nav.iloc[-1] - 1
        metrics['Annual_Return'] = self._annualized_return()
        metrics['Volatility'] = self._annualized_volatility()
        metrics['Sharpe'] = self._sharpe_ratio()
        
        # 风险指标
        metrics['Max_Drawdown'] = self._max_drawdown()
        metrics['Calmar'] = self._calmar_ratio()
        
        # 收益指标
        metrics['Win_Rate'] = self._win_rate()
        metrics['Profit_Loss_Ratio'] = self._profit_loss_ratio()
        
        # 其他指标
        metrics['Total_Periods'] = len(self.returns)
        metrics['Start_Date'] = self.returns.index[0]
        metrics['End_Date'] = self.returns.index[-1]
        
        return metrics
    
    def _annualized_return(self):
        """年化收益率"""
        total_return = self.nav.iloc[-1] / self.nav.iloc[0] - 1
        n_periods = len(self.nav)
        n_years = n_periods / self.periods_per_year
        
        if n_years > 0:
            annual_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annual_return = 0
        
        return annual_return
    
    def _annualized_volatility(self):
        """年化波动率"""
        vol = self.returns.std() * np.sqrt(self.periods_per_year)
        return vol
    
    def _sharpe_ratio(self):
        """夏普比率；波动率过小（近似常数收益）时返回 nan，避免除零爆炸"""
        annual_return = self._annualized_return()
        vol = self._annualized_volatility()
        vol_min = 1e-10  # 避免常数序列导致 Sharpe 数值爆炸
        if vol > vol_min:
            sharpe = (annual_return - self.rf) / vol
        else:
            sharpe = np.nan
        return sharpe
    
    def _max_drawdown(self):
        """最大回撤"""
        cummax = self.nav.cummax()
        drawdown = (self.nav - cummax) / cummax
        max_dd = drawdown.min()
        return max_dd
    
    def _calmar_ratio(self):
        """卡玛比率 (年化收益率 / 最大回撤)"""
        annual_return = self._annualized_return()
        max_dd = self._max_drawdown()
        
        if max_dd < 0:
            calmar = -annual_return / max_dd
        else:
            calmar = np.nan
        
        return calmar
    
    def _win_rate(self):
        """胜率"""
        win_rate = (self.returns > 0).mean()
        return win_rate
    
    def _profit_loss_ratio(self):
        """盈亏比（平均盈利/平均亏损）"""
        profits = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        
        if len(losses) > 0:
            avg_profit = profits.mean() if len(profits) > 0 else 0
            avg_loss = -losses.mean()  # 取绝对值
            
            if avg_loss > 0:
                pl_ratio = avg_profit / avg_loss
            else:
                pl_ratio = np.nan
        else:
            pl_ratio = np.inf if len(profits) > 0 else np.nan
        
        return pl_ratio
    
    def get_monthly_returns(self):
        """
        获取月度收益率
        
        Returns:
        --------
        Series: 月度收益率
        """
        # 使用 'ME' 代替 'M'（pandas 2.2+）
        try:
            monthly_returns = self.returns.resample('ME').apply(
                lambda x: (1 + x).prod() - 1
            )
        except ValueError:
            # 兼容旧版pandas
            monthly_returns = self.returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
        return monthly_returns
    
    def _empty_metrics(self):
        """返回空的指标字典"""
        return {
            'Total_Return': np.nan,
            'Annual_Return': np.nan,
            'Volatility': np.nan,
            'Sharpe': np.nan,
            'Max_Drawdown': np.nan,
            'Calmar': np.nan,
            'Win_Rate': np.nan,
            'Profit_Loss_Ratio': np.nan,
            'Total_Periods': 0,
            'Start_Date': None,
            'End_Date': None
        }