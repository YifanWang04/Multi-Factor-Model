"""
Excel 报表生成模块 (rebalance_report.py)
======================================
职责：
  write_rebalance_day_report — 将调仓日报表写入单文件 Excel（含全部 Sheet）

导出：
  write_rebalance_day_report
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── 路径注册（strategy_utils 位于同级目录）────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from strategy_utils import filter_weight_lt
from .discord_notifier import compute_extended_metrics as _compute_extended_metrics


WEIGHT_FILTER_THRESHOLD: float = 0.0001


def write_rebalance_day_report(
    result: dict,
    status: dict,
    current_ops: pd.DataFrame,
    output_path: str,
    used_live_prices: bool = False,
    mtm_applied: bool = False,
    # 以下参数从调用方传入，避免直接引用模块级配置变量
    strategy_params: Optional[dict] = None,
    selected_factor_indices: Optional[list] = None,
    selected_factor_names: Optional[list[str]] = None,
    composite_factor_sheet: str = "ic_m3_N20",
    strategy_param: str = "",
    rebalance_period: int = 20,
    data_start_offset_days: int = 0,
    rf_rate: float = 0.02,
) -> None:
    """
    写入合并后的调仓日报表（单文件，含全部 sheet）。

    Parameters
    ----------
    result : dict
        回测结果（来自 run_detailed_backtest）
    status : dict
        调仓日状态（来自 get_rebalance_day_status）
    current_ops : pd.DataFrame
        当前调仓日操作明细
    output_path : str
        输出 Excel 路径
    used_live_prices : bool
        是否使用了实时价格
    mtm_applied : bool
        是否执行了 MTM 市值重估
    strategy_params : dict
        策略参数字典
    selected_factor_indices : list
        选定因子索引
    selected_factor_names : list[str]
        选定因子名称
    composite_factor_sheet : str
        复合因子方法
    strategy_param : str
        策略参数字符串
    rebalance_period : int
        调仓周期
    data_start_offset_days : int
        数据起始日偏移
    rf_rate : float
        无风险利率
    """
    if strategy_params is None:
        strategy_params = {}
    if selected_factor_indices is None:
        selected_factor_indices = []
    if selected_factor_names is None:
        selected_factor_names = []

    if "error" in result:
        raise ValueError(result["error"])

    params = result.get("params", {})
    as_of = pd.Timestamp(datetime.now().date())

    daily_returns = result["daily_returns"]
    nav = result["nav"]
    rebalance_returns = result.get("rebalance_returns", pd.Series(dtype=float))

    # 计算绩效指标
    total_ret = float(nav.iloc[-1]) - 1.0 if len(nav) > 0 else float("nan")
    ann_ret = (1 + total_ret) ** (252 / max(1, len(daily_returns))) - 1 if len(daily_returns) > 0 else float("nan")
    vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else float("nan")
    sharpe = (ann_ret - rf_rate) / vol if vol and vol > 0 else float("nan")
    max_dd = float((nav / nav.cummax() - 1).min()) if len(nav) > 0 else float("nan")
    max_dd_pct = max_dd * 100
    calmar = ann_ret / abs(max_dd) if max_dd and max_dd != 0 else float("nan")

    # 单周期最坏回撤（复用 discord_notifier 中的逻辑，避免重复代码）
    wp_dd = np.nan
    wp_dd_pct = np.nan
    if len(rebalance_returns) > 0:
        rb_dates = rebalance_returns.index.tolist()
        worst_val = 0.0
        for i, rb_start in enumerate(rb_dates):
            if i + 1 < len(rb_dates):
                rb_end = rb_dates[i + 1]
            else:
                if len(nav) == 0:
                    continue
                rb_end = nav.index[-1]
            period_ret = daily_returns[daily_returns.index > rb_start]
            if i + 1 < len(rb_dates):
                period_ret = period_ret[period_ret.index <= rb_end]
            if len(period_ret) == 0:
                continue
            if rb_start in nav.index:
                base_nav = nav.loc[rb_start]
            else:
                valid = nav.index[nav.index <= rb_start]
                if len(valid) == 0:
                    continue
                base_nav = nav.loc[valid[-1]]
            period_nav = (1.0 + period_ret).cumprod() * base_nav
            cummax = period_nav.cummax()
            dd_s = (period_nav - cummax) / cummax
            dd_min = dd_s.min()
            if dd_min < worst_val:
                worst_val = dd_min
        if worst_val < 0:
            wp_dd = float(worst_val)
            wp_dd_pct = wp_dd * 100
    win_days = int((daily_returns > 0).sum())
    total_days = len(daily_returns)
    win_rate = win_days / total_days if total_days > 0 else float("nan")
    avg_win = float(daily_returns[daily_returns > 0].mean()) if win_days > 0 else 0.0
    loss_days = int((daily_returns < 0).sum())
    avg_loss = float(daily_returns[daily_returns < 0].mean()) if loss_days > 0 else 0.0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")

    def _fmt(v: float, f: str) -> str:
        if isinstance(v, float) and np.isnan(v):
            return "-"
        return f.format(v)

    price_conv_parts = []
    if mtm_applied:
        price_conv_parts.append(
            "未到期持仓：Sell_Price_Close 为 As_Of 日收盘或实时价（假设卖出），见 Sell_Price_Source 列"
        )
    if used_live_prices:
        price_conv_parts.append(
            "调仓日且未收盘：Today_Open=开盘价，Buy_Price_Close=现价（买入估计）"
        )
    if not price_conv_parts:
        price_conv = "Adj Close（收盘价）；T 日收盘执行；未到期持仓已按市值计价列示"
    else:
        price_conv = "；".join(price_conv_parts)

    status_rows = [
        ["Parameter", "Value"],
        ["As_Of_Date", str(as_of.date())],
        ["Is_Rebalance_Today", "是" if status["is_rebalance_today"] else "否"],
        ["Current_Rebalance_Date", str(status["current_rebalance_date"].date()) if status["current_rebalance_date"] else "-"],
        ["Next_Rebalance_Date", str(status["next_rebalance_date"].date()) if status["next_rebalance_date"] else "-"],
        ["Price_Convention", price_conv],
        ["Rebalance_Period_TradingDays", strategy_params.get("rebalance_period", rebalance_period)],
        ["Data_Start_Offset_TradingDays", data_start_offset_days],
        ["---", "---"],
        ["Factor_Indices", str(selected_factor_indices)],
        ["Selected_Factors", ", ".join(selected_factor_names)],
        ["Composite_Factor", composite_factor_sheet],
        ["Composite_Method", f"IC加权 {composite_factor_sheet} (M=3月, N=20日)"],
        ["Strategy_Param", strategy_param],
        ["Weight_Method", params.get("weight_method", strategy_params.get("weight_method", ""))],
        ["Group_Num", params.get("group_num", strategy_params.get("group_num", ""))],
        ["Target_Rank", params.get("target_rank", strategy_params.get("target_rank", ""))],
        ["---", "---"],
        ["Total_Return", _fmt(total_ret, "{:.4f}")],
        ["Annual_Return", _fmt(ann_ret, "{:.4f}")],
        ["Annual_Volatility_Pct", _fmt(vol * 100 if not np.isnan(vol) else float("nan"), "{:.2f}")],
        ["Sharpe_Ratio", _fmt(sharpe, "{:.2f}")],
        ["Max_Drawdown_Pct", _fmt(max_dd_pct, "{:.2f}")],
        ["Worst_Period_Drawdown_Pct", _fmt(wp_dd_pct, "{:.2f}")],
        ["Calmar_Ratio", _fmt(calmar, "{:.2f}")],
        ["Win_Rate", _fmt(win_rate, "{:.2%}")],
        ["Profit_Loss_Ratio", _fmt(pl_ratio, "{:.2f}")],
    ]

    # 过滤低权重操作
    filtered_ops = filter_weight_lt(current_ops, WEIGHT_FILTER_THRESHOLD, logger=print)
    df_ops_raw = result["operations_df"]
    df_ops_filtered = filter_weight_lt(df_ops_raw, WEIGHT_FILTER_THRESHOLD, logger=print)

    def _nan_to_dash(df: pd.DataFrame) -> pd.DataFrame:
        return df.replace({np.nan: "-"}, inplace=False)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(status_rows[1:], columns=status_rows[0]).to_excel(
            writer, sheet_name="Rebalance_Config_Status", index=False
        )

        if not filtered_ops.empty:
            _nan_to_dash(filtered_ops).to_excel(writer, sheet_name="Current_Operations", index=False)
        else:
            pd.DataFrame({"Note": ["无当前调仓日操作（今日非调仓日或数据不足）"]}).to_excel(
                writer, sheet_name="Current_Operations", index=False
            )

        future_rb = status.get("future_rebalance_dates", [])
        if future_rb:
            pd.DataFrame({"Future_Rebalance_Date": future_rb}).to_excel(
                writer, sheet_name="Future_Rebalance_Dates", index=False
            )
        else:
            pd.DataFrame({"Note": ["暂无未来调仓日数据"]}).to_excel(
                writer, sheet_name="Future_Rebalance_Dates", index=False
            )

        # ---- 新增：包含所有权重的 sheet（不过滤）----
        # Current_Operations_All：当前调仓日所有操作（含 weight < 0.0001）
        if not current_ops.empty:
            _nan_to_dash(current_ops).to_excel(writer, sheet_name="Current_Operations_All", index=False)
        else:
            pd.DataFrame({"Note": ["无当前调仓日操作（今日非调仓日或数据不足）"]}).to_excel(
                writer, sheet_name="Current_Operations_All", index=False
            )

        # All_Operations_All：历史所有操作（含 weight < 0.0001）
        if len(df_ops_raw) > 0:
            _nan_to_dash(df_ops_raw).to_excel(writer, sheet_name="All_Operations_All", index=False)

        df_period = result["period_summary_df"]
        if len(df_period) > 0:
            df_period.to_excel(writer, sheet_name="Period_Summary", index=False)

        df_dr = daily_returns.reset_index()
        df_dr.columns = ["Date", "Daily_Return"]
        df_dr.to_excel(writer, sheet_name="Daily_Returns", index=False)

        df_nav = nav.reset_index()
        df_nav.columns = ["Date", "NAV"]
        df_nav["Cumulative_Return"] = df_nav["NAV"] - 1.0
        df_nav.to_excel(writer, sheet_name="Cumulative_Returns", index=False)

    print(f"调仓日报表已写入: {output_path}")
