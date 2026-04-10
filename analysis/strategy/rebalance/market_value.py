"""
市值重估模块 (market_value.py)
================================
职责：
  - MarkToMarket 类：未到期持仓的市值重估
  - patch_period_summary_from_mtm：用 MTM 结果同步更新 period_summary_df

导出：
  MarkToMarket
  patch_period_summary_from_mtm
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class MarkToMarket:
    """
    对未到期持仓进行市值重估（MTM）。

    使用方式：
        mtm = MarkToMarket(ops_df, price_df, as_of_date)
        mtm.apply(live_prices=...)  # live_prices: {sym: {"current": float, ...}}
        marked_ops = mtm.operations_df
        patch_period_summary_from_mtm(result, marked_ops, as_of_date)  # 同步更新
    """

    # 卖出价来源标注
    SOURCE_OPEN_PERIOD = "假设市价(未到期)"   # 下一调仓日尚未到期
    SOURCE_FILL_MISSING = "假设市价(补全)"     # 卖出价历史缺失
    SOURCE_MATURED = "到期收盘"               # 已到期，有历史卖出价

    def __init__(
        self,
        ops_df: pd.DataFrame,
        price_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ):
        self._ops_df = ops_df.copy()
        self._price_df = price_df
        self._as_of = pd.Timestamp(as_of_date).normalize()
        self._live_prices: dict = {}
        self._source_col = "Sell_Price_Source"

        self._ops_df["Next_Rebalance_Date"] = pd.to_datetime(
            self._ops_df.get("Next_Rebalance_Date", pd.NaT), errors="coerce"
        )
        self._ops_df["Rebalance_Date"] = pd.to_datetime(
            self._ops_df.get("Rebalance_Date", pd.NaT), errors="coerce"
        )
        if self._source_col not in self._ops_df.columns:
            self._ops_df[self._source_col] = ""

    @property
    def operations_df(self) -> pd.DataFrame:
        return self._ops_df

    def _mark_price_for_symbol(self, symbol: str) -> float:
        """取 as_of 及之前最近可用的 Adj Close；若无则用 live_prices。"""
        if symbol not in self._price_df.columns:
            lp = self._live_prices.get(symbol, {})
            cur = lp.get("current")
            return float(cur) if cur is not None else float("nan")

        series = self._price_df[symbol].dropna()
        if len(series) == 0:
            lp = self._live_prices.get(symbol, {})
            cur = lp.get("current")
            return float(cur) if cur is not None else float("nan")

        valid = series[series.index <= self._as_of]
        if len(valid) > 0:
            return float(valid.iloc[-1])

        lp = self._live_prices.get(symbol, {})
        cur = lp.get("current")
        return float(cur) if cur is not None else float("nan")

    def apply(self, live_prices: Optional[dict] = None) -> "MarkToMarket":
        """
        执行市值重估。传入 live_prices: {symbol: {"current": float}}，
        用于 price_df 在 as_of 无有效价时的实时回退。
        返回 self（支持链式调用）。
        """
        if self._ops_df.empty:
            return self

        if live_prices:
            self._live_prices = live_prices

        for idx, row in self._ops_df.iterrows():
            next_rb = row["Next_Rebalance_Date"]
            if pd.isna(next_rb):
                continue
            next_rb = pd.Timestamp(next_rb)

            sell_was_raw = row.get("Sell_Price_Close", np.nan)
            sell_was = float(sell_was_raw) if pd.notna(sell_was_raw) else np.nan

            # 条件：下一调仓日未到，或卖出价仍缺失
            need_mtm = (next_rb > self._as_of) or (pd.isna(sell_was))
            if not need_mtm:
                if str(self._ops_df.at[idx, self._source_col] or "").strip() == "":
                    self._ops_df.at[idx, self._source_col] = self.SOURCE_MATURED
                continue

            mark = self._mark_price_for_symbol(row["Symbol"])
            if pd.isna(mark) or mark <= 0:
                continue

            bp_raw = pd.to_numeric(row.get("Buy_Price_Close"), errors="coerce")
            bp = float(bp_raw) if pd.notna(bp_raw) else np.nan
            if pd.isna(bp) or bp <= 0:
                continue

            wt = pd.to_numeric(row.get("Weight"), errors="coerce")
            buy_value_raw = pd.to_numeric(row.get("Buy_Value"), errors="coerce")
            if pd.isna(buy_value_raw) and pd.notna(wt):
                buy_value = float(wt)  # 虚拟资金基准：组合规模=1
            elif pd.notna(buy_value_raw):
                buy_value = float(buy_value_raw)
            else:
                continue  # Buy_Value 与 Weight 均为 NaN，跳过

            stk_ret = mark / bp - 1.0
            self._ops_df.at[idx, "Sell_Price_Close"] = mark
            self._ops_df.at[idx, "Period_Return"] = stk_ret
            self._ops_df.at[idx, "Sell_Value"] = buy_value * (1.0 + stk_ret)
            self._ops_df.at[idx, "Shares"] = (
                buy_value / bp if buy_value > 0 else np.nan
            )
            self._ops_df.at[idx, self._source_col] = (
                self.SOURCE_OPEN_PERIOD if next_rb > self._as_of
                else self.SOURCE_FILL_MISSING
            )

        return self

    def was_applied(self) -> bool:
        """检查是否实际执行了 MTM（而非仅标注到期收盘）。"""
        if self._ops_df.empty or self._source_col not in self._ops_df.columns:
            return False
        return self._ops_df[self._source_col].isin(
            [self.SOURCE_OPEN_PERIOD, self.SOURCE_FILL_MISSING]
        ).any()


def patch_period_summary_from_mtm(
    result: dict,
    mtm_ops: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> None:
    """
    用 MTM 后的 ops 数据同步更新 result["period_summary_df"]。
    对尚未到期的持仓期，用 MTM 后的个股收益加权更新 Period_Return 与 Holding_Days。
    """
    ps = result.get("period_summary_df")
    if ps is None or ps.empty:
        return

    ps = ps.copy()
    mtm_ops = mtm_ops.copy()
    as_of = pd.Timestamp(as_of_date).normalize()

    ps["Rebalance_Date"] = pd.to_datetime(ps["Rebalance_Date"], errors="coerce")
    ps["Next_Rebalance_Date"] = pd.to_datetime(ps["Next_Rebalance_Date"], errors="coerce")
    mtm_ops["Rebalance_Date"] = pd.to_datetime(mtm_ops["Rebalance_Date"], errors="coerce")
    mtm_ops["Next_Rebalance_Date"] = pd.to_datetime(mtm_ops["Next_Rebalance_Date"], errors="coerce")

    for i, prow in ps.iterrows():
        nr = prow["Next_Rebalance_Date"]
        if pd.isna(nr):
            continue
        nr = pd.Timestamp(nr)
        if nr <= as_of:
            continue

        rb = pd.Timestamp(prow["Rebalance_Date"])
        sub = mtm_ops[
            (mtm_ops["Rebalance_Date"] == rb)
            & (mtm_ops["Next_Rebalance_Date"] == nr)
        ]
        if sub.empty:
            continue

        w = pd.to_numeric(sub["Weight"], errors="coerce").fillna(0.0)
        r = pd.to_numeric(sub["Period_Return"], errors="coerce")
        if w.sum() > 0 and r.notna().any():
            port_ret = float((w * r.fillna(0)).sum() / w.sum())
            ps.at[i, "Period_Return"] = port_ret
        ps.at[i, "Holding_Days"] = max(0, (as_of - rb).days)

    result["period_summary_df"] = ps
