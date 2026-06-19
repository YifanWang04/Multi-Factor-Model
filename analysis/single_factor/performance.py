"""Performance metrics for single-factor period-return backtests."""

import numpy as np
import pandas as pd

from qqq_core.performance_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    profit_loss_ratio,
    sharpe_ratio,
    total_return,
    win_rate,
)


class PerformanceAnalyzer:
    """Calculate performance metrics for long/short, long-only and short-only tests.

    The return series is the source of truth. ``nav`` is accepted for backward
    compatibility with existing callers, but metrics are rebuilt from returns so
    the first period is not accidentally dropped.
    """

    def __init__(self, nav, returns, rf=0.02, periods_per_year=252):
        self.nav = nav
        self.returns = returns
        self.rf = rf
        self.periods_per_year = periods_per_year

    def calculate_metrics(self):
        if len(self.returns) == 0:
            return self._empty_metrics()

        return {
            "Total_Return": total_return(self.returns),
            "Annual_Return": self._annualized_return(),
            "Volatility": self._annualized_volatility(),
            "Sharpe": self._sharpe_ratio(),
            "Max_Drawdown": self._max_drawdown(),
            "Calmar": self._calmar_ratio(),
            "Win_Rate": self._win_rate(),
            "Profit_Loss_Ratio": self._profit_loss_ratio(),
            "Total_Periods": len(self.returns),
            "Start_Date": self.returns.index[0],
            "End_Date": self.returns.index[-1],
        }

    def _annualized_return(self):
        return annualized_return(self.returns, self.periods_per_year)

    def _annualized_volatility(self):
        return annualized_volatility(self.returns, self.periods_per_year)

    def _sharpe_ratio(self):
        return sharpe_ratio(self.returns, self.rf, self.periods_per_year)

    def _max_drawdown(self):
        return max_drawdown(self.returns)

    def _calmar_ratio(self):
        return calmar_ratio(self.returns, self.periods_per_year)

    def _win_rate(self):
        return win_rate(self.returns)

    def _profit_loss_ratio(self):
        return profit_loss_ratio(self.returns)

    def get_monthly_returns(self):
        try:
            return self.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        except ValueError:
            return self.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

    def _empty_metrics(self):
        return {
            "Total_Return": np.nan,
            "Annual_Return": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "Max_Drawdown": np.nan,
            "Calmar": np.nan,
            "Win_Rate": np.nan,
            "Profit_Loss_Ratio": np.nan,
            "Total_Periods": 0,
            "Start_Date": None,
            "End_Date": None,
        }
