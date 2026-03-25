"""
策略回测详细报表生成 (run_detailed_backtest_report.py)
=====================================================
基于 strategy 模块的回测逻辑，生成单策略的明细报表。

配置：
  - 因子选择：来自 strategy_config.STRATEGY_SELECTED_FACTOR_INDICES（与 composite_config 独立）
  - 复合因子方法：由 COMPOSITE_FACTOR_SHEET 指定（如 ic_m3_N20）
  - 策略参数：整串配置，如 max_return_5G_Top1_P10d
    格式 {weight_method}_{N}G_Top{R}_P{D}d

输出 Excel（多 Sheet）：
  - 调仓操作明细：每期调仓的股票、买卖价格、权重、收益率等
  - 日收益率序列：日期 × 日收益率
  - 累计收益率：日期 × 净值
  - 每期调仓汇总：调仓日、持仓数、期间收益率等
  - 持仓明细：每期调仓日各标的权重与因子值

用法（项目根目录）：
  python analysis/strategy/run_detailed_backtest_report.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

# ── 路径注册 ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SF_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "analysis", "single_factor")
_ROOT = os.path.dirname(os.path.dirname(_HERE))

for _p in [_HERE, _SF_DIR, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_strategy import load_composite_factor, load_return_data
from analysis.strategy.strategy_config import (
    STRATEGY_SELECTED_FACTOR_INDICES,
    COMPOSITE_FACTOR_FILE,
)
from strategy_backtest import (
    StrategyBacktester,
    _build_groups,
    _select_rebalance_dates,
)
from portfolio_optimizer import compute_weights
import strategy_config as cfg


# ---------------------------------------------------------------------------
# 配置（本脚本专用）
# ---------------------------------------------------------------------------

PROJECT_ROOT = r"D:\qqq"
from data.data_config import PRICE_FILE, DATA_START_OFFSET_DAYS, STRATEGY_REPORTS_DIR
COMPOSITE_FACTOR_SHEET = "ic_m3_N20"  # beta_m3 方法，N=10 窗口

def _get_data_offset():
    return DATA_START_OFFSET_DAYS
OUTPUT_DIR = STRATEGY_REPORTS_DIR
OUTPUT_EXCEL_NAME = "strategy_detailed_backtest_report.xlsx"

# 策略参数：整串配置，格式 {weight_method}_{N}G_Top{R}_P{D}d
# 例：max_return_5G_Top1_P10d、mvo_10G_Top2_P30d、min_variance_5G_Top3_P20d
STRATEGY_PARAM = "max_return_5G_Top1_P10d"

# 因子索引来自 strategy_config.STRATEGY_SELECTED_FACTOR_INDICES
# ⚠️ 切换因子后需先运行 run_composite_factor.py 确保 composite_factors.xlsx 存在且含指定 sheet


def _safe_tag(s: str) -> str:
    """将字符串转成适合文件名的 tag（尽量保持可读性）。"""
    s = str(s)
    s = s.strip().replace(" ", "")
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in s)


def build_detailed_report_filename(
    base_name: str,
    composite_sheet: str,
    strategy_param: str,
    data_start_offset_days: int = 0,
) -> str:
    """
    在文件名里追加：复合因子方法（sheet）+ 策略参数 + 数据起始日偏移。
    例：strategy_detailed_backtest_report__ic_m3_N20__max_return_5G_Top1_P10d__dataoffset0.xlsx
    """
    root, ext = os.path.splitext(base_name)
    ext = ext or ".xlsx"
    sheet_tag = _safe_tag(composite_sheet)
    strat_tag = _safe_tag(strategy_param)
    offset_tag = f"dataoffset{int(data_start_offset_days)}"
    return f"{root}__{sheet_tag}__{strat_tag}__{offset_tag}{ext}"


def parse_strategy_param(param: str) -> tuple:
    """
    解析策略参数字符串，格式：{weight_method}_{N}G_Top{R}_P{D}d
    例：max_return_5G_Top1_P10d -> (weight_method, group_num, target_rank, rebalance_days)

    Returns
    -------
    tuple : (weight_method, group_num, target_rank, rebalance_days)
    """
    m = re.match(r"^(.+)_(\d+)G_Top(\d+)_P(\d+)d$", param.strip())
    if not m:
        raise ValueError(
            f"策略参数格式错误: '{param}'，应为 {{weight_method}}_{{N}}G_Top{{R}}_P{{D}}d，"
            "例：max_return_5G_Top1_P10d"
        )
    weight_method = m.group(1)
    group_num = int(m.group(2))
    target_rank = int(m.group(3))
    rebalance_days = int(m.group(4))
    return weight_method, group_num, target_rank, rebalance_days


def _strategy_param_from_params(params: dict) -> str:
    """从 params 字典还原策略参数字符串。"""
    w = params.get("weight_method", "")
    g = params.get("group_num", "")
    r = params.get("target_rank", "")
    p = params.get("rebalance_period", "")
    if w != "" and g != "" and r != "" and p != "":
        return f"{w}_{g}G_Top{r}_P{p}d"
    return ""


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_price_data(price_file: str, price_column: str = "Adj Close") -> pd.DataFrame:
    """
    加载日频价格数据，返回宽表 DataFrame(index=日期, columns=股票代码)。
    使用 pd.concat 一次性构建，避免 frame.insert 循环导致的 fragmentation 告警。
    """
    if not os.path.isfile(price_file):
        raise FileNotFoundError(f"价格文件不存在: {price_file}")

    price_data = pd.read_excel(price_file, sheet_name=None)
    columns_dict = {}
    for ticker, df in price_data.items():
        if "Date" not in df.columns or price_column not in df.columns:
            continue
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        columns_dict[ticker] = df[price_column]
    if not columns_dict:
        return pd.DataFrame()
    price_df = pd.concat(columns_dict, axis=1)
    price_df = price_df.apply(pd.to_numeric, errors="coerce")
    price_df.sort_index(inplace=True)
    return price_df


def _get_price_on_date(price_df: pd.DataFrame, date: pd.Timestamp, stocks: list) -> pd.Series:
    """获取指定日期各标的收盘价，缺失则前向填充。"""
    if date not in price_df.index:
        # 取不超过该日期的最近一天
        idx = price_df.index[price_df.index <= date]
        if len(idx) == 0:
            return pd.Series(dtype=float)
        date = idx[-1]
    row = price_df.loc[date]
    return row.reindex(stocks).dropna()


# ---------------------------------------------------------------------------
# 详细回测：记录每期调仓明细
# ---------------------------------------------------------------------------

def run_detailed_backtest(
    factor_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    price_df: pd.DataFrame,
    group_num: int,
    target_rank: int,
    rebalance_period: int,
    weight_method: str,
    config,
) -> dict:
    """
    运行单策略详细回测，返回含调仓操作、日收益、累计收益等完整数据。
    调仓日历由数据起始日（DATA_START_OFFSET_DAYS）控制。
    """
    target_group = group_num - (target_rank - 1)
    rebalance_dates = _select_rebalance_dates(
        factor_df.index,
        ret_df.index,
        rebalance_period,
    )
    if len(rebalance_dates) < 2:
        return {"error": "调仓日不足 2 个"}

    # 将最后一期持仓延伸到下一调仓日（按调仓周期外推）。
    # 使用外推的真实下一调仓日（如 3.27），而非价格数据截止日（如 3.17），
    # 确保 Next_Rebalance_Date 显示正确的调仓计划，不被当日数据截止日误导。
    last_rb = pd.Timestamp(rebalance_dates[-1])
    _after = ret_df.index[ret_df.index > last_rb].sort_values()
    if len(_after) >= rebalance_period:
        # 用实际交易日计数外推（与 _select_rebalance_dates 逻辑一致）
        next_rb_date = pd.Timestamp(_after[rebalance_period - 1])
    else:
        # 数据不足时退化为业务日近似
        _bd = pd.bdate_range(start=last_rb, periods=rebalance_period + 1, freq="B")
        next_rb_date = pd.Timestamp(_bd[-1])
    rebalance_dates = list(rebalance_dates) + [next_rb_date]

    all_daily_rets = []
    all_dates = []
    period_rets = []
    period_dates = []
    operations_records = []
    period_summary_records = []
    period_cum_ret = 1.0  # 用于计算 period cumulative return

    trans_cost = getattr(config, "TRANSACTION_COST", 0.001)
    lookback = getattr(config, "OPTIMIZATION_LOOKBACK", 252)
    rf = getattr(config, "RISK_FREE_RATE", 0.02)
    max_weight = getattr(config, "MAX_WEIGHT", 0.4)

    for i in range(len(rebalance_dates) - 1):
        rb_date = rebalance_dates[i]
        next_rb = rebalance_dates[i + 1]

        # 因子信号
        if rb_date in factor_df.index:
            signal_date = rb_date
        else:
            avail = factor_df.index[factor_df.index <= rb_date]
            if len(avail) == 0:
                continue
            signal_date = avail[-1]

        factor_signal = factor_df.loc[signal_date]

        # 分组
        groups = _build_groups(factor_signal, group_num)
        if target_group not in groups or len(groups[target_group]) == 0:
            continue
        group_stocks = groups[target_group]

        # 权重
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

        # 买卖价格
        buy_prices = _get_price_on_date(price_df, rb_date, group_stocks)
        sell_prices = _get_price_on_date(price_df, next_rb, group_stocks)

        # 持仓期收益
        holding_mask = (ret_df.index > rb_date) & (ret_df.index <= next_rb)
        period_df = ret_df.loc[holding_mask, :]
        if len(period_df) == 0:
            continue

        # 有效标的（有买卖价的）
        valid_stocks = list(
            set(weights.index) & set(buy_prices.index) & set(sell_prices.index)
        )
        if len(valid_stocks) == 0:
            continue

        w = weights.reindex(valid_stocks).fillna(0)
        w = w / w.sum()
        buy_p = buy_prices.reindex(valid_stocks).dropna()
        sell_p = sell_prices.reindex(valid_stocks).dropna()
        common = w.index.intersection(buy_p.index).intersection(sell_p.index)
        if len(common) == 0:
            continue
        w = w[common] / w[common].sum()

        period_daily = []
        for j, (date, row) in enumerate(period_df.iterrows()):
            valid = w.index[w.index.isin(row.dropna().index)]
            if len(valid) == 0:
                port_ret = 0.0
            else:
                ww = w[valid] / w[valid].sum()
                port_ret = float((row[valid] * ww).sum())
            if j == 0:
                port_ret -= 2 * trans_cost
            period_daily.append(port_ret)
            all_daily_rets.append(port_ret)
            all_dates.append(date)

        if not period_daily:
            continue

        period_cum = float(pd.Series(period_daily).add(1.0).prod() - 1.0)
        period_rets.append(period_cum)
        period_dates.append(rb_date)
        period_cum_ret *= 1.0 + period_cum

        # 操作明细：每只股票一行（组合规模=1 的虚拟资金）
        period_days = (next_rb - rb_date).days
        for sym in common:
            bp = buy_prices[sym] if sym in buy_prices.index else np.nan
            sp = sell_prices[sym] if sym in sell_prices.index else np.nan
            stk_ret = (sp / bp - 1.0) if (not np.isnan(bp) and not np.isnan(sp) and bp > 0) else np.nan
            factor_val = factor_signal[sym] if sym in factor_signal.index else np.nan
            wt = w[sym]
            buy_value = wt * 1.0  # 虚拟买入金额
            sell_value = buy_value * (1 + stk_ret) if not np.isnan(stk_ret) else np.nan
            shares = buy_value / bp if (not np.isnan(bp) and bp > 0) else np.nan
            operations_records.append({
                "Rebalance_Date": rb_date,
                "Next_Rebalance_Date": next_rb,
                "Holding_Days": period_days,
                "Symbol": sym,
                "Weight": wt,
                "Buy_Price_Close": bp,
                "Sell_Price_Close": sp,
                "Period_Return": stk_ret,
                "Buy_Value": buy_value,
                "Sell_Value": sell_value,
                "Shares": shares,
                "Factor_Value": factor_val,
            })

        # Symbols: 仅保留 weight > 0.01 的股票，并标注权重（格式：SYM:weight%）
        symbols_with_weight = [
            f"{sym}:{w[sym] * 100:.1f}%"
            for sym in sorted(common)
            if w[sym] > 0.01
        ]
        symbols_str = ", ".join(symbols_with_weight)

        period_summary_records.append({
            "Rebalance_Date": rb_date,
            "Next_Rebalance_Date": next_rb,
            "Holding_Days": (next_rb - rb_date).days,
            "Position_Count": len(common),
            "Period_Return": period_cum,
            "Period_Cumulative_Return": period_cum_ret - 1.0,
            "Symbols": symbols_str,
        })

    if not all_dates:
        return {"error": "无有效持仓期"}

    daily_returns = pd.Series(all_daily_rets, index=all_dates, name="Daily_Return")
    nav = (1.0 + daily_returns).cumprod()
    nav.name = "NAV"
    rebalance_returns = pd.Series(period_rets, index=period_dates, name="Period_Return")

    return {
        "daily_returns": daily_returns,
        "nav": nav,
        "rebalance_dates": period_dates,
        "rebalance_returns": rebalance_returns,
        "operations_df": pd.DataFrame(operations_records),
        "period_summary_df": pd.DataFrame(period_summary_records),
        "params": {
            "group_num": group_num,
            "target_rank": target_rank,
            "target_group": target_group,
            "rebalance_period": rebalance_period,
            "weight_method": weight_method,
            "data_start_offset_days": _get_data_offset(),
        },
    }


# ---------------------------------------------------------------------------
# 写入 Excel 报表
# ---------------------------------------------------------------------------

def write_detailed_report(result: dict, output_path: str) -> None:
    """Write detailed backtest results to multi-sheet Excel."""
    if "error" in result:
        raise ValueError(result["error"])

    params = result.get("params", {})
    dr = result["daily_returns"]
    nv = result["nav"]
    total_ret = float(nv.iloc[-1]) - 1.0 if len(nv) > 0 else np.nan
    ann_ret = (1 + total_ret) ** (252 / max(1, len(dr))) - 1 if len(dr) > 0 else np.nan
    vol = dr.std() * np.sqrt(252) * 100 if len(dr) > 1 else np.nan
    sharpe = (ann_ret - cfg.RISK_FREE_RATE) / (vol / 100) if vol and vol > 0 else np.nan
    max_dd = (nv / nv.cummax() - 1).min() * 100 if len(nv) > 0 else np.nan

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 0: Config & Performance (merged)
        config_rows = [
            ["Factor_Indices", str(STRATEGY_SELECTED_FACTOR_INDICES)],
            ["Composite_Factor", COMPOSITE_FACTOR_SHEET],
            ["Strategy_Param", _strategy_param_from_params(params)],
            ["Weight_Method", params.get("weight_method", "")],
            ["Group_Num", params.get("group_num", "")],
            ["Target_Rank", params.get("target_rank", "")],
            ["Rebalance_Period_TradingDays", params.get("rebalance_period", "")],
            ["Data_Start_Offset_TradingDays", params.get("data_start_offset_days", _get_data_offset())],
            ["Transaction_Cost_OneSide", f"{getattr(cfg, 'TRANSACTION_COST', 0.001):.3f}"],
            ["Timing_Convention", "Trade at T close, holding period (T, T_next]"],
            ["---", "---"],
            ["Total_Return", f"{total_ret:.4f}" if not np.isnan(total_ret) else "-"],
            ["Annual_Return", f"{ann_ret:.4f}" if not np.isnan(ann_ret) else "-"],
            ["Annual_Volatility_Pct", f"{vol:.2f}" if not np.isnan(vol) else "-"],
            ["Sharpe_Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "-"],
            ["Max_Drawdown_Pct", f"{max_dd:.2f}" if not np.isnan(max_dd) else "-"],
            ["Backtest_Range", f"{dr.index[0].date()} ~ {dr.index[-1].date()}" if len(dr) > 0 else "-"],
            ["Rebalance_Count", len(result["rebalance_dates"])],
        ]
        pd.DataFrame(config_rows, columns=["Parameter", "Value"]).to_excel(
            writer, sheet_name="Config_Performance", index=False
        )

        # Sheet 1: Rebalance Operations
        df_ops = result["operations_df"]
        if len(df_ops) > 0:
            df_ops.to_excel(writer, sheet_name="Operations", index=False)

        # Sheet 2: Daily Returns
        df_ret = result["daily_returns"].reset_index()
        df_ret.columns = ["Date", "Daily_Return"]
        df_ret.to_excel(writer, sheet_name="Daily_Returns", index=False)

        # Sheet 3: Cumulative Returns
        df_nav = result["nav"].reset_index()
        df_nav.columns = ["Date", "NAV"]
        df_nav["Cumulative_Return"] = df_nav["NAV"] - 1.0
        df_nav.to_excel(writer, sheet_name="Cumulative_Returns", index=False)

        # Sheet 4: Period Summary
        df_period = result["period_summary_df"]
        if len(df_period) > 0:
            df_period.to_excel(writer, sheet_name="Period_Summary", index=False)

    print(f"Report written: {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    weight_method, group_num, target_rank, rebalance_days = parse_strategy_param(STRATEGY_PARAM)

    print("=" * 64)
    print("  策略回测详细报表")
    print(f"  因子 {STRATEGY_SELECTED_FACTOR_INDICES} | 复合 {COMPOSITE_FACTOR_SHEET} | {STRATEGY_PARAM}")
    print("=" * 64)

    # 1. 加载复合因子
    print(f"\n[1/4] 加载复合因子: {COMPOSITE_FACTOR_SHEET}")
    factor_df = load_composite_factor(COMPOSITE_FACTOR_FILE, COMPOSITE_FACTOR_SHEET)
    print(f"      因子区间: {factor_df.index[0].date()} ~ {factor_df.index[-1].date()}")

    # 2. 加载日频收益率
    print(f"\n[2/4] 加载日频收益率")
    ret_df = load_return_data(PRICE_FILE, cfg.RETURN_COLUMN)
    ret_df.sort_index(inplace=True)
    print(f"      收益率区间: {ret_df.index[0].date()} ~ {ret_df.index[-1].date()}")

    # 3. 加载价格数据
    print(f"\n[3/4] 加载价格数据（Adj Close）")
    price_df = load_price_data(PRICE_FILE, "Adj Close")
    print(f"      价格区间: {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    # 4. 运行详细回测
    print(f"\n[4/4] 运行策略回测: {STRATEGY_PARAM}")
    result = run_detailed_backtest(
        factor_df=factor_df,
        ret_df=ret_df,
        price_df=price_df,
        group_num=group_num,
        target_rank=target_rank,
        rebalance_period=rebalance_days,
        weight_method=weight_method,
        config=cfg,
    )

    if "error" in result:
        print(f"错误: {result['error']}")
        return

    # 5. 写入 Excel
    report_name = build_detailed_report_filename(
        base_name=OUTPUT_EXCEL_NAME,
        composite_sheet=COMPOSITE_FACTOR_SHEET,
        strategy_param=STRATEGY_PARAM,
        data_start_offset_days=DATA_START_OFFSET_DAYS,
    )
    output_path = os.path.join(OUTPUT_DIR, report_name)
    write_detailed_report(result, output_path)

    print(f"\n共 {len(result['rebalance_dates'])} 次调仓，{len(result['operations_df'])} 条操作记录")
    print("=" * 64)


if __name__ == "__main__":
    main()
