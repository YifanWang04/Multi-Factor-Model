"""Shared portfolio performance metric helpers.

The helpers in this module treat the return series as the source of truth.
Net asset value is reconstructed as ``(1 + returns).cumprod()`` with an
implicit initial wealth of 1.0 for drawdown calculations.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


VOL_MIN: float = 1e-10


def clean_returns(returns: pd.Series | list | np.ndarray | None) -> pd.Series:
    """Return a numeric Series with NaN/inf values removed."""

    if returns is None:
        return pd.Series(dtype=float)
    series = pd.Series(returns).copy()
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series.astype(float)


def nav_from_returns(returns: pd.Series | list | np.ndarray | None) -> pd.Series:
    """Build a NAV series from returns, without the initial 1.0 anchor."""

    rets = clean_returns(returns)
    nav = (1.0 + rets).cumprod()
    nav.name = "NAV"
    return nav


def total_return(returns: pd.Series | list | np.ndarray | None) -> float:
    """Compound total return over the full return series."""

    rets = clean_returns(returns)
    if len(rets) == 0:
        return float("nan")
    terminal_wealth = float((1.0 + rets).prod())
    return terminal_wealth - 1.0


def annualized_return(
    returns: pd.Series | list | np.ndarray | None,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized compounded return.

    Returns NaN when terminal wealth is non-positive because fractional powers
    would be undefined or misleading.
    """

    rets = clean_returns(returns)
    n = len(rets)
    if n == 0 or periods_per_year <= 0:
        return float("nan")
    terminal_wealth = float((1.0 + rets).prod())
    if terminal_wealth <= 0:
        return float("nan")
    return float(terminal_wealth ** (float(periods_per_year) / n) - 1.0)


def annualized_volatility(
    returns: pd.Series | list | np.ndarray | None,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized sample volatility."""

    rets = clean_returns(returns)
    if len(rets) < 2 or periods_per_year <= 0:
        return float("nan")
    return float(rets.std(ddof=1) * math.sqrt(float(periods_per_year)))


def sharpe_ratio(
    returns: pd.Series | list | np.ndarray | None,
    rf: float = 0.02,
    periods_per_year: float = 252.0,
) -> float:
    """Sharpe ratio using annualized compounded return and annualized vol."""

    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if np.isnan(ann_ret) or np.isnan(ann_vol) or ann_vol <= VOL_MIN:
        return float("nan")
    return float((ann_ret - float(rf)) / ann_vol)


def _wealth_with_initial(
    returns: pd.Series | list | np.ndarray | None,
    initial_index: Any | None = None,
) -> pd.Series:
    rets = clean_returns(returns)
    wealth = nav_from_returns(rets)
    if len(rets) == 0:
        return pd.Series([1.0], index=[initial_index], dtype=float)
    idx0 = rets.index[0] if initial_index is None else initial_index
    initial = pd.Series([1.0], index=[idx0], dtype=float)
    return pd.concat([initial, wealth])


def drawdown_series(returns: pd.Series | list | np.ndarray | None) -> pd.Series:
    """Drawdown series with an implicit initial wealth anchor included."""

    wealth = _wealth_with_initial(returns)
    cummax = wealth.cummax()
    dd = wealth / cummax - 1.0
    return dd


def max_drawdown(returns: pd.Series | list | np.ndarray | None) -> float:
    """Maximum drawdown, including drawdown from the initial wealth of 1.0."""

    rets = clean_returns(returns)
    if len(rets) == 0:
        return float("nan")
    dd = drawdown_series(rets)
    return float(dd.min())


def max_drawdown_info(returns: pd.Series | list | np.ndarray | None) -> tuple[float, Any, Any]:
    """Return maximum drawdown plus peak and trough dates/labels."""

    rets = clean_returns(returns)
    if len(rets) == 0:
        return float("nan"), np.nan, np.nan

    wealth = _wealth_with_initial(rets)
    values = wealth.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(values)
    dd = values / peaks - 1.0
    end_pos = int(np.nanargmin(dd))
    start_pos = int(np.nanargmax(values[: end_pos + 1]))
    return float(dd[end_pos]), wealth.index[start_pos], wealth.index[end_pos]


def calmar_ratio(
    returns: pd.Series | list | np.ndarray | None,
    periods_per_year: float = 252.0,
) -> float:
    """Calmar ratio using annualized return divided by absolute max drawdown."""

    ann_ret = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if np.isnan(ann_ret) or np.isnan(mdd) or mdd >= 0.0:
        return float("nan")
    return float(ann_ret / abs(mdd))


def win_rate(returns: pd.Series | list | np.ndarray | None) -> float:
    rets = clean_returns(returns)
    if len(rets) == 0:
        return float("nan")
    return float((rets > 0).mean())


def profit_loss_ratio(returns: pd.Series | list | np.ndarray | None) -> float:
    rets = clean_returns(returns)
    if len(rets) == 0:
        return float("nan")
    profits = rets[rets > 0]
    losses = rets[rets < 0]
    if len(losses) == 0:
        return float("inf") if len(profits) > 0 else float("nan")
    avg_profit = float(profits.mean()) if len(profits) > 0 else 0.0
    avg_loss = float(-losses.mean())
    return float(avg_profit / avg_loss) if avg_loss > VOL_MIN else float("nan")


def performance_summary(
    returns: pd.Series | list | np.ndarray | None,
    rf: float = 0.02,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Return the standard metric set for one return series."""

    return {
        "total_return": total_return(returns),
        "annual_return": annualized_return(returns, periods_per_year),
        "annual_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, rf, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_loss_ratio": profit_loss_ratio(returns),
    }


def worst_period_drawdown(
    daily_returns: pd.Series | list | np.ndarray | None,
    rebalance_returns: pd.Series | list | np.ndarray | None,
) -> tuple[float, Any, Any]:
    """Worst drawdown inside any rebalance holding interval.

    ``rebalance_returns.index`` is interpreted as rebalance dates. Each interval
    is ``(rb_date, next_rb_date]``; the last interval extends to the end of the
    daily return series. The first interval uses an initial base wealth of 1.0.
    """

    rets = clean_returns(daily_returns)
    rb_rets = clean_returns(rebalance_returns)
    if len(rets) == 0 or len(rb_rets) == 0:
        return float("nan"), np.nan, np.nan

    full_nav = nav_from_returns(rets)
    rb_dates = list(rb_rets.index)
    worst_dd = 0.0
    worst_start: Any = np.nan
    worst_end: Any = np.nan

    for i, rb_start in enumerate(rb_dates):
        if i + 1 < len(rb_dates):
            rb_end = rb_dates[i + 1]
            period_ret = rets[(rets.index > rb_start) & (rets.index <= rb_end)]
        else:
            period_ret = rets[rets.index > rb_start]

        if len(period_ret) == 0:
            continue

        prior_nav = full_nav[full_nav.index <= rb_start]
        base_nav = float(prior_nav.iloc[-1]) if len(prior_nav) > 0 else 1.0

        period_nav = (1.0 + period_ret).cumprod() * base_nav
        wealth = pd.concat([pd.Series([base_nav], index=[rb_start]), period_nav])
        cummax = wealth.cummax()
        dd = wealth / cummax - 1.0
        dd_min = float(dd.min())
        if dd_min < worst_dd:
            values = wealth.to_numpy(dtype=float)
            peaks = np.maximum.accumulate(values)
            dd_values = values / peaks - 1.0
            end_pos = int(np.nanargmin(dd_values))
            start_pos = int(np.nanargmax(values[: end_pos + 1]))
            worst_dd = dd_min
            worst_start = wealth.index[start_pos]
            worst_end = wealth.index[end_pos]

    if worst_dd == 0.0 and pd.isna(worst_end):
        return 0.0, np.nan, np.nan
    return float(worst_dd), worst_start, worst_end
