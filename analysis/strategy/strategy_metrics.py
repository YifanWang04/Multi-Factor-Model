"""
策略绩效指标计算 (strategy_metrics.py)
=====================================
输入：日频组合收益率序列（daily_returns）+ 每持仓周期收益率序列（rebalance_returns）
输出：完整绩效指标字典，供回测报表使用。

计算指标清单：
  近期收益  : 近1日、近1周、近1月、近3月、近半年、近1年、上一年整年
  全周期    : 年化收益、年化波动、夏普比率
  开仓统计  : 开仓胜率、开仓盈亏比、年化开仓次数、年化盈利次数
  回撤      : 最大回撤、Calmar 比率、最大回撤起始日、最大回撤结束日
"""

import numpy as np
import pandas as pd


class StrategyMetrics:
    """
    绩效指标计算器。

    Parameters
    ----------
    daily_returns     : pd.Series（index=日期，values=日收益率）
    rebalance_returns : pd.Series（index=调仓日，values=该持仓周期总收益率）
    rf                : float，年化无风险利率，默认 0.02
    periods_per_year  : int，日频=252
    """

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

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        if len(self.rets) == 0:
            return self._empty()

        m: dict = {}

        # ── 近期收益 ──────────────────────────────────────────────────
        m["ret_1d"] = self._tail_ret(1)
        m["ret_1w"] = self._tail_ret(5)
        m["ret_1m"] = self._tail_ret(21)
        m["ret_3m"] = self._tail_ret(63)
        m["ret_6m"] = self._tail_ret(126)
        m["ret_1y"] = self._tail_ret(252)
        m["ret_last_year"] = self._last_full_year_ret()

        # ── 全周期指标 ─────────────────────────────────────────────────
        m["annual_return"] = self._annual_return()
        m["annual_vol"] = self._annual_vol()
        m["sharpe"] = self._sharpe()

        # ── 开仓统计（基于调仓周期收益） ──────────────────────────────
        m["open_win_rate"] = self._open_win_rate()
        m["open_pl_ratio"] = self._open_pl_ratio()
        m["annual_open_count"] = self._annual_open_count()
        m["annual_profit_count"] = self._annual_profit_count()

        # ── 回撤 ──────────────────────────────────────────────────────
        m["max_drawdown"] = self._max_drawdown()
        m["calmar"] = self._calmar()
        dd_start, dd_end = self._max_dd_dates()
        m["max_dd_start"] = dd_start
        m["max_dd_end"] = dd_end
        # 单持仓周期最坏回撤（所有调仓周期内各自的最大回撤，取最差值）
        wp_dd, wp_start, wp_end = self._worst_period_drawdown()
        m["worst_period_drawdown"] = wp_dd
        m["worst_period_dd_start"] = wp_start
        m["worst_period_dd_end"] = wp_end

        return m

    # ------------------------------------------------------------------
    # 净值曲线（懒加载）
    # ------------------------------------------------------------------

    @property
    def nav(self) -> pd.Series:
        if self._nav is None:
            self._nav = (1.0 + self.rets).cumprod()
        return self._nav

    # ------------------------------------------------------------------
    # 近期收益
    # ------------------------------------------------------------------

    def _tail_ret(self, n_days: int) -> float:
        tail = self.rets.iloc[-n_days:]
        if len(tail) == 0:
            return np.nan
        return float((1.0 + tail).prod() - 1.0)

    def _last_full_year_ret(self) -> float:
        """上一个完整日历年（例如 2025-01-01 ~ 2025-12-31）的收益率。"""
        if len(self.rets) == 0:
            return np.nan
        last_year = self.rets.index[-1].year - 1
        yr = self.rets[self.rets.index.year == last_year]
        if len(yr) == 0:
            return np.nan
        return float((1.0 + yr).prod() - 1.0)

    # ------------------------------------------------------------------
    # 全周期指标
    # ------------------------------------------------------------------

    def _annual_return(self) -> float:
        n = len(self.nav)
        if n < 2:
            return np.nan
        total = float(self.nav.iloc[-1] / self.nav.iloc[0])
        n_years = n / self.ppy
        if n_years <= 0:
            return np.nan
        return float(total ** (1.0 / n_years) - 1.0)

    def _annual_vol(self) -> float:
        if len(self.rets) < 2:
            return np.nan
        return float(self.rets.std() * np.sqrt(self.ppy))

    def _sharpe(self) -> float:
        ar = self._annual_return()
        av = self._annual_vol()
        if np.isnan(ar) or np.isnan(av) or av < 1e-10:
            return np.nan
        return float((ar - self.rf) / av)

    # ------------------------------------------------------------------
    # 开仓统计
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 回撤
    # ------------------------------------------------------------------

    def _max_drawdown(self) -> float:
        if len(self.nav) == 0:
            return np.nan
        cummax = self.nav.cummax()
        dd = (self.nav - cummax) / cummax
        return float(dd.min())

    def _calmar(self) -> float:
        ar = self._annual_return()
        mdd = self._max_drawdown()
        if np.isnan(ar) or np.isnan(mdd) or mdd >= 0.0:
            return np.nan
        return float(-ar / mdd)

    def _max_dd_dates(self) -> tuple:
        """返回 (最大回撤起始日, 最大回撤结束日)。"""
        if len(self.nav) < 2:
            return np.nan, np.nan
        cummax = self.nav.cummax()
        dd = (self.nav - cummax) / cummax
        dd_end = dd.idxmin()
        dd_start = self.nav.loc[:dd_end].idxmax()
        return dd_start, dd_end

    def _worst_period_drawdown(self) -> tuple:
        """
        返回 (单周期最坏回撤, 该回撤起始日, 该回撤结束日)。
        对每个持仓区间（调仓日 → 下一调仓日）分别计算期间最大回撤，
        取所有区间中最差（最负）的一次。
        """
        if len(self.rets) == 0 or len(self.rb_rets) == 0:
            return np.nan, np.nan, np.nan

        # rebalance_returns 的 index 是调仓日，values 是区间总收益率
        rb_dates = self.rb_rets.index.tolist()

        worst_dd = 0.0       # 越大越好的初始值（回撤 ≤ 0）
        worst_start = np.nan
        worst_end = np.nan

        for i, rb_start in enumerate(rb_dates):
            # 该区间的日频收益率切片
            if i + 1 < len(rb_dates):
                rb_end = rb_dates[i + 1]
            else:
                # 最后一个区间：到 nav 最后一个日期
                if len(self.nav) == 0:
                    continue
                rb_end = self.nav.index[-1]

            period_ret = self.rets[self.rets.index > rb_start]
            if i + 1 < len(rb_dates):
                period_ret = period_ret[period_ret.index <= rb_end]

            if len(period_ret) == 0:
                continue

            # 从买入时刻（rb_start 的净值）起计算期间最大回撤
            if rb_start in self.nav.index:
                base_nav = self.nav.loc[rb_start]
            else:
                # 找不到精确 rb_start，用最近的前序日期净值
                valid = self.nav.index[self.nav.index <= rb_start]
                if len(valid) == 0:
                    continue
                base_nav = self.nav.loc[valid[-1]]

            period_nav = (1.0 + period_ret).cumprod() * base_nav
            if len(period_nav) == 0:
                continue

            cummax = period_nav.cummax()
            dd = (period_nav - cummax) / cummax
            dd_min = dd.min()

            if dd_min < worst_dd:
                worst_dd = dd_min
                worst_end = dd.idxmin()
                worst_start = period_nav.loc[:worst_end].idxmax() if len(period_nav.loc[:worst_end]) > 0 else np.nan

        if worst_dd == 0.0 and np.isnan(worst_end):
            return 0.0, np.nan, np.nan
        return float(worst_dd), worst_start, worst_end

    # ------------------------------------------------------------------
    # 空指标字典（无收益数据时的占位）
    # ------------------------------------------------------------------

    @staticmethod
    def _empty() -> dict:
        keys = [
            "ret_1d", "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_last_year",
            "annual_return", "annual_vol", "sharpe",
            "open_win_rate", "open_pl_ratio", "annual_open_count", "annual_profit_count",
            "max_drawdown", "calmar", "max_dd_start", "max_dd_end",
            "worst_period_drawdown", "worst_period_dd_start", "worst_period_dd_end",
        ]
        return {k: np.nan for k in keys}


# ---------------------------------------------------------------------------
# 便捷函数：从回测结果字典批量计算指标
# ---------------------------------------------------------------------------

def compute_all_metrics(results: dict, rf: float = 0.02) -> dict:
    """
    输入 {strategy_name: result_dict}（来自 StrategyBacktester.run_grid()），
    返回 {strategy_name: metrics_dict}。
    """
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
