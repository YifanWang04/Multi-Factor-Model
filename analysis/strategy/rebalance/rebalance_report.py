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
import pandas_market_calendars as pmc

# ── 路径注册（strategy_utils 位于同级目录）────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from strategy_utils import filter_weight_lt
from tp_sl_exit import (
    EXIT_DYNAMIC_TP_SL,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT,
    thresholds_for_day,
)


WEIGHT_FILTER_THRESHOLD: float = 0.0001
ALL_OPERATIONS_WEIGHT_FILTER_THRESHOLD: float = 0.01
TP_SL_ACTION_LOOKAHEAD_DATES: int = 5


def _describe_composite_method(sheet_name: str) -> str:
    """给报告用的复合方法说明；m1 明确标注全样本前瞻偏误。"""
    if "_m1" in sheet_name:
        return f"{sheet_name}（全样本 oracle baseline，含前瞻偏误，仅供研究对比）"
    if sheet_name.startswith("ic_"):
        return f"{sheet_name}（IC 加权）"
    if sheet_name.startswith("rank_ic_"):
        return f"{sheet_name}（Rank IC 加权）"
    if sheet_name.startswith("beta_"):
        return f"{sheet_name}（Beta 加权）"
    if sheet_name.startswith("ols_"):
        return f"{sheet_name}（OLS 多元回归加权）"
    if sheet_name.startswith("pca_"):
        return f"{sheet_name}（PCA 主成分）"
    if sheet_name.startswith("rank_"):
        return f"{sheet_name}（截面排名复合）"
    return sheet_name


def _as_naive_trading_day(value) -> pd.Timestamp:
    """Normalize NYSE calendar values to timezone-naive midnight timestamps."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _future_nyse_trading_days(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None,
    rebalance_period: int,
) -> list[pd.Timestamp]:
    """Return holding trading days after start_date and before end_date."""

    start = pd.Timestamp(start_date).normalize()
    if pd.isna(start):
        return []

    if end_date is not None and pd.notna(end_date):
        end = pd.Timestamp(end_date).normalize()
    else:
        end = start + pd.Timedelta(days=max(30, int(rebalance_period) * 4))

    if end <= start:
        return []

    cal = pmc.get_calendar("NYSE")
    valid = cal.valid_days(start, end)
    days = [_as_naive_trading_day(x) for x in valid]
    return [d for d in days if d > start and d < end]


def _active_dynamic_buy_rows(current_ops: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Select live buy rows eligible for a forward TP/SL schedule."""

    if current_ops is None or current_ops.empty:
        return pd.DataFrame()
    if "Action" not in current_ops.columns:
        return pd.DataFrame()

    ops = current_ops.copy()
    buy_rows = ops[ops["Action"].astype(str).str.lower() == "buy"].copy()
    if buy_rows.empty:
        return pd.DataFrame()

    if "Next_Rebalance_Date" in buy_rows.columns:
        buy_rows["Next_Rebalance_Date"] = pd.to_datetime(
            buy_rows["Next_Rebalance_Date"], errors="coerce"
        )
        buy_rows = buy_rows[
            buy_rows["Next_Rebalance_Date"].isna()
            | (buy_rows["Next_Rebalance_Date"] > as_of)
        ]

    if "Exit_Reason" in buy_rows.columns:
        exit_reason = buy_rows["Exit_Reason"].fillna("").astype(str)
        buy_rows = buy_rows[~exit_reason.isin([EXIT_TAKE_PROFIT, EXIT_STOP_LOSS])]

    return buy_rows


def build_tp_sl_schedule(
    current_ops: pd.DataFrame,
    as_of_date: pd.Timestamp,
    exit_policy: str,
    rebalance_period: int,
    tp_base: float,
    sl_base: float,
    probability: float = 1.0,
) -> pd.DataFrame:
    """Build the forward dynamic TP/SL price schedule for current buys."""

    columns = [
        "Symbol",
        "Rebalance_Date",
        "Schedule_Date",
        "TD",
        "Next_Rebalance_Date",
        "Buy_Price_Close",
        "TP_Return_Threshold",
        "SL_Return_Threshold",
        "TP_Price",
        "SL_Price",
        "Weight",
        "Shares",
    ]

    if exit_policy != EXIT_DYNAMIC_TP_SL:
        return pd.DataFrame(columns=columns)
    if rebalance_period <= 1:
        return pd.DataFrame(columns=columns)

    as_of = pd.Timestamp(as_of_date).normalize()
    active = _active_dynamic_buy_rows(current_ops, as_of)
    if active.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for _, row in active.iterrows():
        symbol = row.get("Symbol")
        rb_date = pd.to_datetime(row.get("Rebalance_Date"), errors="coerce")
        next_rb = pd.to_datetime(row.get("Next_Rebalance_Date"), errors="coerce")
        buy_price = pd.to_numeric(row.get("Buy_Price_Close"), errors="coerce")
        if pd.isna(symbol) or pd.isna(rb_date) or pd.isna(buy_price) or buy_price <= 0:
            continue

        trading_days = _future_nyse_trading_days(rb_date, next_rb, rebalance_period)
        for td, schedule_date in enumerate(trading_days, start=1):
            if td >= rebalance_period:
                break
            tp_ret, sl_ret = thresholds_for_day(
                tp_base,
                sl_base,
                rebalance_period,
                td,
                probability,
            )
            records.append(
                {
                    "Symbol": symbol,
                    "Rebalance_Date": pd.Timestamp(rb_date).normalize(),
                    "Schedule_Date": schedule_date,
                    "TD": td,
                    "Next_Rebalance_Date": (
                        pd.Timestamp(next_rb).normalize() if pd.notna(next_rb) else pd.NaT
                    ),
                    "Buy_Price_Close": float(buy_price),
                    "TP_Return_Threshold": tp_ret,
                    "SL_Return_Threshold": sl_ret,
                    "TP_Price": float(buy_price) * (1.0 + tp_ret),
                    "SL_Price": float(buy_price) * (1.0 - sl_ret),
                    "Weight": row.get("Weight", np.nan),
                    "Shares": row.get("Shares", np.nan),
                }
            )

    return pd.DataFrame(records, columns=columns)


def build_tp_sl_action_checklist(
    schedule_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookahead_dates: int = TP_SL_ACTION_LOOKAHEAD_DATES,
) -> pd.DataFrame:
    """Return the nearest schedule rows for manual alert/check setup."""

    columns = [
        "Date",
        "Symbol",
        "TP_Price",
        "SL_Price",
        "Buy_Price_Close",
        "Weight",
        "Suggested_Check",
    ]
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(columns=columns)

    as_of = pd.Timestamp(as_of_date).normalize()
    sched = schedule_df.copy()
    sched["Schedule_Date"] = pd.to_datetime(sched["Schedule_Date"], errors="coerce")
    future = sched[sched["Schedule_Date"] >= as_of].copy()
    if future.empty:
        return pd.DataFrame(columns=columns)

    keep_dates = sorted(future["Schedule_Date"].dropna().unique())[:lookahead_dates]
    checklist = future[future["Schedule_Date"].isin(keep_dates)].copy()
    checklist["Date"] = checklist["Schedule_Date"]
    checklist["Suggested_Check"] = "Set price alert / Check near close"
    return checklist[columns].sort_values(["Date", "Symbol"]).reset_index(drop=True)


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
        ["Composite_Method", _describe_composite_method(composite_factor_sheet)],
        ["Strategy_Param", strategy_param],
        ["Weight_Method", params.get("weight_method", strategy_params.get("weight_method", ""))],
        ["Group_Num", params.get("group_num", strategy_params.get("group_num", ""))],
        ["Target_Rank", params.get("target_rank", strategy_params.get("target_rank", ""))],
        ["Exit_Policy", params.get("exit_policy", strategy_params.get("exit_policy", ""))],
        ["TP_Base", params.get("tp_base", "")],
        ["SL_Base", params.get("sl_base", "")],
        ["Signal_Probability", params.get("probability", "")],
        ["TP_Count", params.get("tp_count", "")],
        ["SL_Count", params.get("sl_count", "")],
        ["Forced_Close_Count", params.get("forced_close_count", "")],
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
    df_ops_all_filtered = filter_weight_lt(
        df_ops_raw,
        ALL_OPERATIONS_WEIGHT_FILTER_THRESHOLD,
        logger=print,
    )

    exit_policy = params.get("exit_policy", strategy_params.get("exit_policy", ""))

    def _as_float(value, default: float) -> float:
        try:
            if value == "":
                return default
            parsed = float(value)
            if np.isfinite(parsed):
                return parsed
        except (TypeError, ValueError):
            pass
        return default

    tp_base = _as_float(params.get("tp_base", strategy_params.get("tp_base", np.nan)), np.nan)
    sl_base = _as_float(params.get("sl_base", strategy_params.get("sl_base", np.nan)), np.nan)
    probability = _as_float(
        params.get("probability", strategy_params.get("probability", strategy_params.get("tp_sl_probability", 1.0))),
        1.0,
    )
    report_rebalance_period = int(strategy_params.get("rebalance_period", rebalance_period))
    if not np.isfinite(tp_base):
        tp_base = 0.0
    if not np.isfinite(sl_base):
        sl_base = 0.0

    tp_sl_schedule = build_tp_sl_schedule(
        current_ops=filtered_ops,
        as_of_date=as_of,
        exit_policy=str(exit_policy),
        rebalance_period=report_rebalance_period,
        tp_base=tp_base,
        sl_base=sl_base,
        probability=probability,
    )
    tp_sl_checklist = build_tp_sl_action_checklist(tp_sl_schedule, as_of)

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

        if str(exit_policy) == EXIT_DYNAMIC_TP_SL:
            if not tp_sl_schedule.empty:
                _nan_to_dash(tp_sl_schedule).to_excel(writer, sheet_name="TP_SL_Schedule", index=False)
            else:
                pd.DataFrame({"Note": ["动态 TP/SL 策略当前无可生成 schedule 的持仓"]}).to_excel(
                    writer, sheet_name="TP_SL_Schedule", index=False
                )

            if not tp_sl_checklist.empty:
                _nan_to_dash(tp_sl_checklist).to_excel(
                    writer, sheet_name="TP_SL_Action_Checklist", index=False
                )
            else:
                pd.DataFrame({"Note": ["当前无今日或近期 TP/SL 人工检查项"]}).to_excel(
                    writer, sheet_name="TP_SL_Action_Checklist", index=False
                )
        else:
            pd.DataFrame({"Note": [f"Exit_Policy={exit_policy}; TP/SL schedule not applicable"]}).to_excel(
                writer, sheet_name="TP_SL_Schedule", index=False
            )
            pd.DataFrame({"Note": [f"Exit_Policy={exit_policy}; TP/SL checklist not applicable"]}).to_excel(
                writer, sheet_name="TP_SL_Action_Checklist", index=False
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

        # All_Operations_All：历史操作明细，过滤 1% 以下的小权重噪音。
        if len(df_ops_all_filtered) > 0:
            _nan_to_dash(df_ops_all_filtered).to_excel(writer, sheet_name="All_Operations_All", index=False)

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
