"""
调仓日操作计算模块 (rebalance_operations.py)
============================================
职责：
  - 调仓日判定与未来调仓日推算
  - 当前调仓日操作明细（卖出 + 买入）
  - 实时价格获取与市值重估（MTM Round 1 & 2）

导出：
  get_rebalance_day_status   — 调仓日状态判定
  get_current_rebalance_ops  — 当前调仓日操作明细
  apply_live_prices_to_ops   — 对调仓日操作应用实时价格
  collect_live_prices_for_mtm — 收集实时价格用于 MTM
  fetch_live_prices          — yfinance 实时价格获取
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ── 路径注册（strategy_utils / strategy_backtest / portfolio_optimizer 位于同级目录）───
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from strategy_utils import _get_price_on_date
from strategy_backtest import _build_groups, _select_rebalance_dates
from portfolio_optimizer import compute_weights


# ---------------------------------------------------------------------------
# 全局常量（消除魔法数字）
# ---------------------------------------------------------------------------

WEIGHT_FILTER_THRESHOLD: float = 0.0001

REBALANCE_EXTRAPOLATE_MAX_PERIODS: int = 24
REBALANCE_EXTRAPOLATE_FUTURE_MIN: int = 12

LIVE_PRICE_MAX_RETRIES: int = 3
LIVE_PRICE_RETRY_DELAY_BASE: float = 0.5
LIVE_PRICE_RETRY_DELAY_MULT: float = 2.0
LIVE_PRICE_TIMEOUT: float = 10.0


# ---------------------------------------------------------------------------
# 调仓日判定
# ---------------------------------------------------------------------------

def get_rebalance_day_status(
    rebalance_dates: list,
    rebalance_period: int,
    as_of_date: pd.Timestamp,
    last_factor_date: pd.Timestamp,
    trading_dates: Optional[list] = None,
) -> dict:
    """
    判定调仓日状态。
    rebalance_period: 调仓周期（交易日数）。
    trading_dates: 可选，交易日序列；用于推算未来调仓日。
    有 trading_dates 时优先用实际交易日序列外推（保持与回测日历一致），
    无 trading_dates 时用 pd.bdate_range 工作日序列兜底。
    """
    rebalance_dates = sorted(rebalance_dates)
    if not rebalance_dates:
        return {
            "is_rebalance_today": False,
            "current_rebalance_date": None,
            "next_rebalance_date": None,
            "future_rebalance_dates": [],
            "all_rebalance_dates": [],
        }

    anchor = rebalance_dates[-1]
    extrapolated = []
    current_date = anchor
    sorted_td = sorted(trading_dates) if trading_dates else []

    for _ in range(REBALANCE_EXTRAPOLATE_MAX_PERIODS):
        if sorted_td:
            try:
                idx = next(i for i, x in enumerate(sorted_td) if x >= current_date)
            except StopIteration:
                idx = len(sorted_td)
            next_idx = idx + rebalance_period
            if next_idx < len(sorted_td):
                current_date = sorted_td[next_idx]
            else:
                bdate_range = pd.bdate_range(start=current_date, periods=rebalance_period + 1, freq="B")
                current_date = pd.Timestamp(bdate_range[-1])
        else:
            bdate_range = pd.bdate_range(start=current_date, periods=rebalance_period + 1, freq="B")
            current_date = pd.Timestamp(bdate_range[-1])
        extrapolated.append(current_date)
        future_so_far = [x for x in extrapolated if x > as_of_date]
        if len(future_so_far) >= REBALANCE_EXTRAPOLATE_FUTURE_MIN:
            break

    all_dates = sorted(set(rebalance_dates) | set(extrapolated))
    past_all = [x for x in all_dates if x <= as_of_date]
    current_rebalance_date = past_all[-1] if past_all else None
    future_all = [x for x in all_dates if x > as_of_date]
    next_rebalance_date = future_all[0] if future_all else None

    is_rebalance_today = (
        current_rebalance_date is not None
        and current_rebalance_date.date() == as_of_date.date()
    )

    future_rebalance_dates = future_all[:REBALANCE_EXTRAPOLATE_FUTURE_MIN]

    return {
        "is_rebalance_today": is_rebalance_today,
        "current_rebalance_date": current_rebalance_date,
        "next_rebalance_date": next_rebalance_date,
        "future_rebalance_dates": future_rebalance_dates,
        "all_rebalance_dates": rebalance_dates,
    }


# ---------------------------------------------------------------------------
# 实时价格获取
# ---------------------------------------------------------------------------

def fetch_live_prices(symbols: list[str]) -> dict[str, dict[str, float]]:
    """
    通过 yfinance 获取当日开盘价和现价（带重试机制）。
    返回 {symbol: {"open": float, "current": float}}，失败的标的不包含在结果中。
    异常处理：捕获网络错误（HTTPError）、API 属性变化（AttributeError）等，
    重试失败后打印错误信息（不再静默吞掉）。
    """
    import requests as _req

    result: dict[str, dict[str, float]] = {}
    delay = LIVE_PRICE_RETRY_DELAY_BASE

    for sym in symbols:
        success = False
        last_error = ""
        for attempt in range(LIVE_PRICE_MAX_RETRIES):
            err_kind = "未知"
            try:
                ticker = yf.Ticker(sym)
                fi = ticker.fast_info
                open_p = getattr(fi, "open", None)
                current_p = getattr(fi, "last_price", None)
                if open_p is not None and current_p is not None:
                    result[sym] = {"open": float(open_p), "current": float(current_p)}
                    success = True
                    break
            except _req.HTTPError as e:
                err_kind = "HTTPError"
                last_error = f"[{err_kind}] {e}"
            except AttributeError as e:
                err_kind = "AttributeError"
                last_error = f"[{err_kind}] 字段不存在: {e}"
            except (ValueError, KeyError) as e:
                err_kind = type(e).__name__
                last_error = f"[{err_kind}] 解析错误: {e}"
            except _req.RequestException as e:
                err_kind = "RequestException"
                last_error = f"[{err_kind}] 网络错误: {e}"
            except Exception as e:
                err_kind = type(e).__name__
                last_error = f"[{err_kind}] {e}"

            if attempt < LIVE_PRICE_MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= LIVE_PRICE_RETRY_DELAY_MULT

        if not success:
            print(f"  ⚠️ 获取 {sym} 实时价失败（已重试 {LIVE_PRICE_MAX_RETRIES} 次）: {last_error}")

    return result


# ---------------------------------------------------------------------------
# MTM 辅助：向量化价格获取
# ---------------------------------------------------------------------------

def _get_price_for_symbols_vectorized(
    price_df: pd.DataFrame,
    as_of: pd.Timestamp,
    symbols: list[str],
) -> dict[str, float]:
    """
    向量化获取指定日期各标的收盘价（返回 dict，替代逐行查 price_df）。
    取 as_of 及之前最近可用的 Adj Close；若无列或全缺失则返回 NaN。
    """
    result: dict[str, float] = {}
    if price_df.empty or not symbols:
        return result

    valid_syms = [s for s in symbols if s in price_df.columns]
    if not valid_syms:
        return result

    available_idx = price_df.index[price_df.index <= as_of]
    if len(available_idx) == 0:
        return result

    prices = price_df.loc[available_idx[-1], valid_syms].dropna()
    for sym in valid_syms:
        if sym in prices.index:
            v = prices[sym]
            result[sym] = float(v) if pd.notna(v) else float("nan")
        else:
            result[sym] = float("nan")
    return result


def collect_live_prices_for_mtm(
    ops_df: pd.DataFrame,
    price_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> dict[str, dict[str, float]]:
    """
    对需要按市值计价、且本地行情表在 as_of 仍无有效价的标的，批量拉取 yfinance 实时价。

    向量化实现（替代逐行 iterrows）：先用条件过滤，再用 price_df 向量查本地价格。
    """
    if ops_df.empty:
        return {}
    if "Next_Rebalance_Date" not in ops_df.columns:
        return {}

    as_of = pd.Timestamp(as_of_date).normalize()
    ops = ops_df.copy()
    ops["Next_Rebalance_Date"] = pd.to_datetime(ops["Next_Rebalance_Date"], errors="coerce")

    valid_next = ops["Next_Rebalance_Date"].notna()
    open_period = ops["Next_Rebalance_Date"] > as_of
    missing_sell = ops["Sell_Price_Close"].isna()
    need_mask = valid_next & (open_period | missing_sell)
    need_ops = ops[need_mask]

    if need_ops.empty:
        return {}

    syms = need_ops["Symbol"].unique().tolist()
    local_prices = _get_price_for_symbols_vectorized(price_df, as_of, syms)
    missing_syms = [s for s in syms if pd.isna(local_prices.get(s))]
    if not missing_syms:
        return {}

    print(f"  MTM：对 {len(missing_syms)} 只标的拉取实时价（本地无 as_of 收盘价）")
    return fetch_live_prices(missing_syms)


# ---------------------------------------------------------------------------
# 对调仓日操作应用实时价格
# ---------------------------------------------------------------------------

def apply_live_prices_to_operations(
    current_ops: pd.DataFrame,
    price_df: pd.DataFrame,
    current_rebalance_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, bool]:
    """
    当调仓日且数据中无当日收盘价时，用 yfinance 获取当日开盘价和现价，
    以开盘价作为 Today_Open，以现价作为收盘价替代（仅 Buy_Price_Close）。
    未到期持仓的假设卖出价由 apply_mark_to_market_operations_df 单独处理（不在此覆盖）。
    返回 (更新后的 current_ops, 是否使用了实时价)。
    """
    if current_ops.empty:
        return current_ops, False

    if current_rebalance_date.date() != as_of_date.date():
        return current_ops, False
    if price_df.empty or price_df.index.max().date() >= as_of_date.date():
        return current_ops, False

    symbols = current_ops["Symbol"].unique().tolist()
    live_prices = fetch_live_prices(symbols)
    if len(live_prices) == 0:
        print("  无法获取实时价格，继续使用历史数据")
        return current_ops, False

    print(f"  调仓日且未收盘：使用实时价（开盘价+现价替代收盘价），成功获取 {len(live_prices)}/{len(symbols)} 只")
    ops = current_ops.copy()

    if "Today_Open" not in ops.columns:
        ops.insert(ops.columns.get_loc("Symbol") + 1, "Today_Open", np.nan)

    for idx, row in ops.iterrows():
        sym = row["Symbol"]
        if sym not in live_prices:
            continue
        o, c = live_prices[sym]["open"], live_prices[sym]["current"]
        ops.at[idx, "Today_Open"] = o
        if row["Action"] == "Buy":
            ops.at[idx, "Buy_Price_Close"] = c
            bv = row.get("Buy_Value", np.nan)
            if not (np.isnan(bv) if isinstance(bv, float) else False) and bv is not None and c > 0:
                ops.at[idx, "Shares"] = float(bv) / c

    return ops, True


# ---------------------------------------------------------------------------
# 当前调仓日操作明细
# ---------------------------------------------------------------------------

def get_current_rebalance_operations(
    result: dict,
    current_rebalance_date: pd.Timestamp,
    next_rebalance_date: Optional[pd.Timestamp] = None,
    rebalance_dates: Optional[list] = None,
    strategy_params: Optional[dict] = None,
    config=None,
) -> pd.DataFrame:
    """
    获取当前调仓日操作：卖出上一期持仓 + 买入本期标的。
    next_rebalance_date: 外推的下一调仓日，用于填充 Next_Rebalance_Date 字段。
    rebalance_dates: 历史调仓日列表，用于在无卖出记录时回退到前一个调仓日的卖出。

    卖出逻辑：
    - 取 Rebalance_Date < current_rebalance_date 且 Next_Rebalance_Date == current_rebalance_date 的记录
    - 若无匹配（current_rebalance_date 还未发生），回退到前一调仓日的卖出记录
    - 过滤掉 Next_Rebalance_Date 为空的记录（最后一期持仓，无实际卖出）

    返回含 Action 列（Sell/Buy）的 DataFrame，先卖后买。
    """
    if "error" in result:
        return pd.DataFrame()

    ops_df = result.get("operations_df", pd.DataFrame())
    sell_ops = pd.DataFrame()
    buy_ops = pd.DataFrame()

    rb_ts = pd.Timestamp(current_rebalance_date)

    # 1. 卖出操作：买入日在 current_rb 之前、卖出日在 current_rb 当日
    if not ops_df.empty and "Next_Rebalance_Date" in ops_df.columns:
        ops = ops_df.copy()
        ops["Rebalance_Date"] = pd.to_datetime(ops["Rebalance_Date"], errors="coerce")
        ops["Next_Rebalance_Date"] = pd.to_datetime(ops["Next_Rebalance_Date"], errors="coerce")

        mask_sell = (
            (ops["Rebalance_Date"] < rb_ts)
            & (ops["Next_Rebalance_Date"] == rb_ts)
            & (ops["Sell_Price_Close"].notna())
        )
        sell_ops = ops.loc[mask_sell].copy()

        # 若无卖出记录（current_rb 还未发生），回退到前一调仓日的卖出
        if sell_ops.empty and rebalance_dates:
            sorted_rb = sorted([pd.Timestamp(d) for d in rebalance_dates if pd.notna(d)])
            prev_rb = None
            for d in reversed(sorted_rb):
                if d < rb_ts:
                    prev_rb = d
                    break
            if prev_rb is not None:
                mask_prev = (
                    (ops["Rebalance_Date"] < prev_rb)
                    & (ops["Next_Rebalance_Date"] == prev_rb)
                    & (ops["Sell_Price_Close"].notna())
                )
                sell_ops = ops.loc[mask_prev].copy()

        if not sell_ops.empty:
            sell_ops.insert(0, "Action", "Sell")

    # 2. 买入操作：今日买入的标的
    if not ops_df.empty and "Rebalance_Date" in ops_df.columns:
        ops = ops_df.copy()
        ops["Rebalance_Date"] = pd.to_datetime(ops["Rebalance_Date"], errors="coerce")
        mask_buy = ops["Rebalance_Date"] == rb_ts
        buy_ops = ops.loc[mask_buy].copy()
        if not buy_ops.empty:
            buy_ops.insert(0, "Action", "Buy")

    if buy_ops.empty:
        computed = _compute_last_rebalance_ops(
            factor_df=result.get("_factor_df"),
            ret_df=result.get("_ret_df"),
            price_df=result.get("_price_df"),
            rb_date=current_rebalance_date,
            config=result.get("_config"),
            next_rb_date=next_rebalance_date,
            strategy_params=strategy_params,
        )
        if not computed.empty:
            buy_ops = computed.copy()
            buy_ops.insert(0, "Action", "Buy")

    # 合并：先卖后买
    if sell_ops.empty and buy_ops.empty:
        return pd.DataFrame()
    if sell_ops.empty:
        return buy_ops
    if buy_ops.empty:
        return sell_ops
    return pd.concat([sell_ops, buy_ops], ignore_index=True)


def _compute_last_rebalance_ops(
    factor_df,
    ret_df,
    price_df,
    rb_date: pd.Timestamp,
    config,
    next_rb_date: Optional[pd.Timestamp] = None,
    strategy_params: Optional[dict] = None,
) -> pd.DataFrame:
    """对最后一个调仓日（无 next_rb）计算买入操作明细。next_rb_date 为外推的下一调仓日。"""
    if factor_df is None or ret_df is None or price_df is None or config is None:
        print(f"  ⚠️ _compute_last_rebalance_ops：缺少必要数据，跳过")
        return pd.DataFrame()

    if strategy_params is None:
        strategy_params = {}

    group_num = strategy_params.get("group_num", 5)
    target_rank = strategy_params.get("target_rank", 1)
    weight_method = strategy_params.get("weight_method", "equal")
    target_group = group_num - (target_rank - 1)
    lookback = getattr(config, "OPTIMIZATION_LOOKBACK", 252)
    rf = getattr(config, "RISK_FREE_RATE", 0.02)
    max_weight = getattr(config, "MAX_WEIGHT", 0.4)

    if rb_date in factor_df.index:
        signal_date = rb_date
    else:
        avail = factor_df.index[factor_df.index <= rb_date]
        if len(avail) == 0:
            print(f"  ⚠️ _compute_last_rebalance_ops：调仓日 {rb_date.date()} 前无可用因子，跳过")
            return pd.DataFrame()
        signal_date = avail[-1]

    factor_signal = factor_df.loc[signal_date]
    groups = _build_groups(factor_signal, group_num)
    if target_group not in groups or len(groups[target_group]) == 0:
        print(f"  ⚠️ _compute_last_rebalance_ops：目标分组 {target_group} 为空，跳过")
        return pd.DataFrame()
    group_stocks = groups[target_group]

    hist_ret = ret_df.loc[ret_df.index < rb_date, :].tail(lookback)
    weights = compute_weights(
        method=weight_method,
        stocks=group_stocks,
        factor_values=factor_signal,
        hist_returns=hist_ret,
        lookback=lookback,
        rf=rf,
        max_weight=max_weight,
    )

    buy_prices = _get_price_on_date(price_df, rb_date, group_stocks)
    valid_stocks = list(set(weights.index) & set(buy_prices.index))
    if len(valid_stocks) == 0:
        print(f"  ⚠️ _compute_last_rebalance_ops：权重标的与买入价格标的无交集，跳过")
        return pd.DataFrame()

    w = weights.reindex(valid_stocks).fillna(0)
    w = w / w.sum()
    buy_p = buy_prices.reindex(valid_stocks).dropna()
    common = w.index.intersection(buy_p.index)
    if len(common) == 0:
        return pd.DataFrame()
    w = w[common] / w[common].sum()

    _next_rb = pd.Timestamp(next_rb_date) if next_rb_date is not None else pd.NaT
    _holding_days = (
        (pd.Timestamp(next_rb_date) - rb_date).days
        if next_rb_date is not None
        else float("nan")
    )

    records = []
    for sym in common:
        bp = float(buy_p[sym])
        factor_val = float(factor_signal[sym]) if sym in factor_signal.index else float("nan")
        wt = float(w[sym])
        buy_value = wt * 1.0
        shares = buy_value / bp if bp > 0 else float("nan")
        records.append({
            "Rebalance_Date": rb_date,
            "Next_Rebalance_Date": _next_rb,
            "Holding_Days": _holding_days,
            "Symbol": sym,
            "Weight": wt,
            "Buy_Price_Close": bp,
            "Sell_Price_Close": float("nan"),
            "Period_Return": float("nan"),
            "Buy_Value": buy_value,
            "Sell_Value": float("nan"),
            "Shares": shares,
            "Factor_Value": factor_val,
        })
    return pd.DataFrame(records)
