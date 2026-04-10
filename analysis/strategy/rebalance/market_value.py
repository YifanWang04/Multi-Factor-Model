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

        向量化实现（替代逐行 iterrows）：
          Phase 1: 批量获取所有标的的 mark 价格（本地向量查 + yfinance 补缺）
          Phase 2: 向量条件过滤 + 批量赋值
        """
        if self._ops_df.empty:
            return self

        if live_prices:
            self._live_prices = live_prices

        # ── Phase 1: 批量获取所有标的的 mark 价格 ──────────────────────
        syms = self._ops_df["Symbol"].unique().tolist()
        mark_prices = {
            sym: self._mark_price_for_symbol(sym)
            for sym in syms
        }

        # ── Phase 2: 向量条件过滤 + 批量赋值 ───────────────────────────
        ops = self._ops_df
        next_rb_col = ops["Next_Rebalance_Date"]
        nr_ts = pd.to_datetime(next_rb_col, errors="coerce")
        sell_was = pd.to_numeric(ops["Sell_Price_Close"], errors="coerce")
        need_mtm = nr_ts.notna() & (
            (nr_ts > self._as_of) | sell_was.isna()
        )

        if not need_mtm.any():
            # 全部到期且有卖出价：仅补填 SOURCE_MATURED
            empty_src = ops[self._source_col].replace("", np.nan).isna()
            if empty_src.any():
                ops.loc[empty_src, self._source_col] = self.SOURCE_MATURED
            return self

        target = ops[need_mtm].copy()
        sym_col = target["Symbol"]
        marks = sym_col.map(mark_prices)          # Series: idx → mark_price
        valid_mark = marks.notna() & (marks > 0)

        bp = pd.to_numeric(target["Buy_Price_Close"], errors="coerce")
        valid_bp = bp.notna() & (bp > 0)

        wt = pd.to_numeric(target["Weight"], errors="coerce")
        buy_value_raw = pd.to_numeric(target["Buy_Value"], errors="coerce")
        buy_value = buy_value_raw.copy()
        use_weight = buy_value_raw.isna() & wt.notna()
        buy_value.loc[use_weight] = wt.loc[use_weight]

        valid_buy = buy_value.notna() & (buy_value > 0)
        final_mask = valid_mark & valid_bp & valid_buy

        idx_final = target.index[final_mask]

        if len(idx_final) > 0:
            bp_v = bp.loc[idx_final]
            mk_v = marks.loc[idx_final]
            bv_v = buy_value.loc[idx_final]
            nr_v = nr_ts.loc[idx_final]

            ops.loc[idx_final, "Sell_Price_Close"] = mk_v
            ops.loc[idx_final, "Period_Return"] = mk_v / bp_v - 1.0
            ops.loc[idx_final, "Sell_Value"] = bv_v * (1.0 + (mk_v / bp_v - 1.0))
            ops.loc[idx_final, "Shares"] = bv_v / bp_v
            ops.loc[idx_final, self._source_col] = np.where(
                nr_v > self._as_of,
                self.SOURCE_OPEN_PERIOD,
                self.SOURCE_FILL_MISSING,
            )

        # 补填不需要 MTM 但无来源标注的行
        already_filled = ops[self._source_col].replace("", np.nan).notna()
        still_empty = ops[self._source_col].replace("", np.nan).isna()
        has_sell_price = sell_was.notna()
        matured_mask = need_mtm & ~need_mtm & still_empty & has_sell_price
        ops.loc[still_empty & ~need_mtm, self._source_col] = self.SOURCE_MATURED

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

    向量化实现（替代逐行 iterrows）。
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

    # 筛选未到期的持仓期
    nr_col = ps["Next_Rebalance_Date"]
    future = nr_col.notna() & (nr_col > as_of)
    if not future.any():
        result["period_summary_df"] = ps
        return

    future_idx = ps.index[future]
    rb_col = ps["Rebalance_Date"]

    w = pd.to_numeric(mtm_ops["Weight"], errors="coerce").fillna(0.0)
    r = pd.to_numeric(mtm_ops["Period_Return"], errors="coerce")

    new_rets = []
    new_days = []
    for i in future_idx:
        rb = pd.Timestamp(rb_col[i])
        nr = pd.Timestamp(nr_col[i])
        sub = mtm_ops[
            (mtm_ops["Rebalance_Date"] == rb)
            & (mtm_ops["Next_Rebalance_Date"] == nr)
        ]
        if sub.empty:
            new_rets.append(np.nan)
            new_days.append(ps.at[i, "Holding_Days"])
            continue
        sub_w = w.loc[sub.index]
        sub_r = r.loc[sub.index]
        ws = sub_w.sum()
        if ws > 0 and sub_r.notna().any():
            port_ret = float((sub_w * sub_r.fillna(0)).sum() / ws)
        else:
            port_ret = np.nan
        new_rets.append(port_ret)
        new_days.append(max(0, (as_of - rb).days))

    ps.loc[future_idx, "Period_Return"] = new_rets
    ps.loc[future_idx, "Holding_Days"] = new_days
    result["period_summary_df"] = ps
