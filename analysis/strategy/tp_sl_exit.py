"""Dynamic take-profit / stop-loss helpers for strategy backtests.

The module checks exits with adjusted close prices only. It does not rebalance
cash from exited positions back into still-open positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EXIT_FIXED_REBALANCE = "fixed_rebalance"
EXIT_DYNAMIC_TP_SL = "dynamic_tp_sl"
EXIT_TAKE_PROFIT = "take_profit"
EXIT_STOP_LOSS = "stop_loss"
EXIT_FORCED_REBALANCE = "forced_rebalance"


@dataclass(frozen=True)
class ExitEvent:
    symbol: str
    exit_date: pd.Timestamp
    exit_reason: str
    exit_td: int
    exit_price: float
    exit_return: float
    tp_threshold: float
    sl_threshold: float
    signal_probability: float


def normalize_exit_policy(value: object) -> str:
    """Return a supported exit policy string."""

    text = str(value or EXIT_FIXED_REBALANCE).strip()
    if text not in {EXIT_FIXED_REBALANCE, EXIT_DYNAMIC_TP_SL}:
        raise ValueError(
            f"Unsupported exit_policy {text!r}; expected "
            f"{EXIT_FIXED_REBALANCE!r} or {EXIT_DYNAMIC_TP_SL!r}"
        )
    return text


def threshold_for_day(
    base: float,
    rebalance_period: int,
    td: int,
    probability: float = 1.0,
) -> float:
    """Return the time-decayed threshold for one holding day."""

    if rebalance_period <= 0:
        raise ValueError("rebalance_period must be positive")
    td = int(td)
    scale = max(float(rebalance_period - td), 0.0) / float(rebalance_period)
    return float(base) * scale * float(probability)


def thresholds_for_day(
    tp_base: float,
    sl_base: float,
    rebalance_period: int,
    td: int,
    probability: float = 1.0,
) -> tuple[float, float]:
    """Return (take_profit_threshold, stop_loss_threshold)."""

    return (
        threshold_for_day(tp_base, rebalance_period, td, probability),
        threshold_for_day(sl_base, rebalance_period, td, probability),
    )


def find_exit_event(
    symbol: str,
    entry_price: float,
    price_series: pd.Series,
    rebalance_period: int,
    tp_base: float,
    sl_base: float,
    probability: float = 1.0,
    force_exit_date: pd.Timestamp | None = None,
    force_exit_price: float | None = None,
) -> ExitEvent | None:
    """Find the first TP/SL trigger, or return a forced rebalance exit."""

    if entry_price is None or not np.isfinite(entry_price) or entry_price <= 0:
        return None

    p = float(probability)
    series = price_series.dropna().sort_index()
    last_td = 0
    last_tp = np.nan
    last_sl = np.nan
    last_date = None
    last_price = np.nan

    for td, (date, price) in enumerate(series.items(), start=1):
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        ts = pd.Timestamp(date)
        pnl = float(price) / float(entry_price) - 1.0
        tp, sl = thresholds_for_day(tp_base, sl_base, rebalance_period, td, p)
        last_td = td
        last_tp = tp
        last_sl = sl
        last_date = ts
        last_price = float(price)
        if td < rebalance_period and pnl >= tp:
            return ExitEvent(symbol, ts, EXIT_TAKE_PROFIT, td, float(price), pnl, tp, sl, p)
        if td < rebalance_period and pnl <= -sl:
            return ExitEvent(symbol, ts, EXIT_STOP_LOSS, td, float(price), pnl, tp, sl, p)

    if force_exit_date is not None and force_exit_price is not None:
        if np.isfinite(force_exit_price) and force_exit_price > 0:
            td = int(last_td) if last_td else int(rebalance_period)
            tp, sl = thresholds_for_day(tp_base, sl_base, rebalance_period, td, p)
            pnl = float(force_exit_price) / float(entry_price) - 1.0
            return ExitEvent(
                symbol,
                pd.Timestamp(force_exit_date),
                EXIT_FORCED_REBALANCE,
                td,
                float(force_exit_price),
                pnl,
                tp,
                sl,
                p,
            )

    if last_date is None or not np.isfinite(last_price):
        return None
    pnl = last_price / float(entry_price) - 1.0
    return ExitEvent(
        symbol,
        last_date,
        EXIT_FORCED_REBALANCE,
        int(last_td),
        float(last_price),
        pnl,
        float(last_tp),
        float(last_sl),
        p,
    )


def build_exit_events(
    price_df: pd.DataFrame,
    entry_prices: pd.Series,
    rb_date: pd.Timestamp,
    exit_end_date: pd.Timestamp,
    rebalance_period: int,
    tp_base: float,
    sl_base: float,
    probability: float = 1.0,
) -> dict[str, ExitEvent]:
    """Build exit events for all positions in one holding period."""

    events: dict[str, ExitEvent] = {}
    if price_df is None or price_df.empty:
        return events

    holding_prices = price_df.loc[
        (price_df.index > rb_date) & (price_df.index <= exit_end_date)
    ]
    for symbol, entry_price in entry_prices.dropna().items():
        if symbol not in holding_prices.columns:
            continue
        price_series = holding_prices[symbol]
        force_price = np.nan
        force_date = exit_end_date
        available = price_series.dropna()
        if len(available) > 0:
            force_date = pd.Timestamp(available.index[-1])
            force_price = float(available.iloc[-1])
        event = find_exit_event(
            symbol=str(symbol),
            entry_price=float(entry_price),
            price_series=price_series,
            rebalance_period=rebalance_period,
            tp_base=tp_base,
            sl_base=sl_base,
            probability=probability,
            force_exit_date=force_date,
            force_exit_price=force_price,
        )
        if event is not None:
            events[str(symbol)] = event
    return events


def event_counts(events: dict[str, ExitEvent]) -> dict[str, int]:
    """Count exit reasons in a holding-period event dict."""

    counts = {
        "tp_count": 0,
        "sl_count": 0,
        "forced_close_count": 0,
    }
    for event in events.values():
        if event.exit_reason == EXIT_TAKE_PROFIT:
            counts["tp_count"] += 1
        elif event.exit_reason == EXIT_STOP_LOSS:
            counts["sl_count"] += 1
        elif event.exit_reason == EXIT_FORCED_REBALANCE:
            counts["forced_close_count"] += 1
    return counts
