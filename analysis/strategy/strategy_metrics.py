"""Strategy performance metrics.

Inputs:
- ``daily_returns``: daily portfolio returns.
- ``rebalance_returns``: per-opening-period total returns.

The daily return series is the source of truth for full-period metrics.
Opening statistics remain based on ``rebalance_returns``.
"""

import numpy as np
import pandas as pd

from qqq_core.performance_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    max_drawdown_info,
    loss_duration_stats,
    nav_from_returns,
    sharpe_ratio,
    worst_period_drawdown,
)


class StrategyMetrics:
    def __init__(
        self,
        daily_returns: pd.Series,
        rebalance_returns: pd.Series,
        rf: float = 0.02,
        periods_per_year: int = 252,
    ):
        self.rets = daily_returns.dropna()
        self.rb_rets = rebalance_returns.dropna()
        self.rf = rf
        self.ppy = periods_per_year
        self._nav: pd.Series | None = None

    def compute_all(self) -> dict:
        if len(self.rets) == 0:
            return self._empty()

        m: dict = {}
        m["ret_1d"] = self._tail_ret(1)
        m["ret_1w"] = self._tail_ret(5)
        m["ret_1m"] = self._tail_ret(21)
        m["ret_3m"] = self._tail_ret(63)
        m["ret_6m"] = self._tail_ret(126)
        m["ret_1y"] = self._tail_ret(252)
        m["ret_last_year"] = self._last_full_year_ret()

        m["annual_return"] = self._annual_return()
        m["annual_vol"] = self._annual_vol()
        m["sharpe"] = self._sharpe()

        m["open_win_rate"] = self._open_win_rate()
        m["open_pl_ratio"] = self._open_pl_ratio()
        m["annual_open_count"] = self._annual_open_count()
        m["annual_profit_count"] = self._annual_profit_count()

        m["max_drawdown"] = self._max_drawdown()
        m["max_loss_duration"], m["avg_loss_duration"] = loss_duration_stats(self.rets)
        m["calmar"] = self._calmar()
        dd_start, dd_end = self._max_dd_dates()
        m["max_dd_start"] = dd_start
        m["max_dd_end"] = dd_end
        wp_dd, wp_start, wp_end = self._worst_period_drawdown()
        m["worst_period_drawdown"] = wp_dd
        m["worst_period_dd_start"] = wp_start
        m["worst_period_dd_end"] = wp_end

        return m

    @property
    def nav(self) -> pd.Series:
        if self._nav is None:
            self._nav = nav_from_returns(self.rets)
        return self._nav

    def _tail_ret(self, n_days: int) -> float:
        tail = self.rets.iloc[-n_days:]
        if len(tail) == 0:
            return np.nan
        return float((1.0 + tail).prod() - 1.0)

    def _last_full_year_ret(self) -> float:
        if len(self.rets) == 0:
            return np.nan
        last_year = self.rets.index[-1].year - 1
        yr = self.rets[self.rets.index.year == last_year]
        if len(yr) == 0:
            return np.nan
        return float((1.0 + yr).prod() - 1.0)

    def _annual_return(self) -> float:
        return annualized_return(self.rets, self.ppy)

    def _annual_vol(self) -> float:
        return annualized_volatility(self.rets, self.ppy)

    def _sharpe(self) -> float:
        return sharpe_ratio(self.rets, self.rf, self.ppy)

    def _open_win_rate(self) -> float:
        if len(self.rb_rets) == 0:
            return np.nan
        return float((self.rb_rets > 0).mean())

    def _open_pl_ratio(self) -> float:
        if len(self.rb_rets) == 0:
            return np.nan
        profits = self.rb_rets[self.rb_rets > 0]
        losses = self.rb_rets[self.rb_rets < 0]
        if len(losses) == 0:
            return np.inf if len(profits) > 0 else np.nan
        avg_profit = float(profits.mean()) if len(profits) > 0 else 0.0
        avg_loss = float(-losses.mean())
        return float(avg_profit / avg_loss) if avg_loss > 1e-12 else np.nan

    def _annual_open_count(self) -> float:
        if len(self.rb_rets) == 0 or len(self.rets) == 0:
            return np.nan
        n_years = len(self.rets) / self.ppy
        return float(len(self.rb_rets) / n_years) if n_years > 0 else np.nan

    def _annual_profit_count(self) -> float:
        if len(self.rb_rets) == 0 or len(self.rets) == 0:
            return np.nan
        n_years = len(self.rets) / self.ppy
        n_profit = int((self.rb_rets > 0).sum())
        return float(n_profit / n_years) if n_years > 0 else np.nan

    def _max_drawdown(self) -> float:
        return max_drawdown(self.rets)

    def _calmar(self) -> float:
        return calmar_ratio(self.rets, self.ppy)

    def _max_dd_dates(self) -> tuple:
        _, dd_start, dd_end = max_drawdown_info(self.rets)
        return dd_start, dd_end

    def _worst_period_drawdown(self) -> tuple:
        return worst_period_drawdown(self.rets, self.rb_rets)

    @staticmethod
    def _empty() -> dict:
        keys = [
            "ret_1d", "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_last_year",
            "annual_return", "annual_vol", "sharpe",
            "open_win_rate", "open_pl_ratio", "annual_open_count", "annual_profit_count",
            "max_drawdown", "max_loss_duration", "avg_loss_duration",
            "calmar", "max_dd_start", "max_dd_end",
            "worst_period_drawdown", "worst_period_dd_start", "worst_period_dd_end",
        ]
        return {k: np.nan for k in keys}


def compute_all_metrics(results: dict, rf: float = 0.02) -> dict:
    all_metrics = {}
    for name, res in results.items():
        calc = StrategyMetrics(
            daily_returns=res.get("daily_returns", pd.Series(dtype=float)),
            rebalance_returns=res.get("rebalance_returns", pd.Series(dtype=float)),
            rf=rf,
        )
        m = calc.compute_all()
        m.update(res.get("params", {}))
        all_metrics[name] = m
    return all_metrics
