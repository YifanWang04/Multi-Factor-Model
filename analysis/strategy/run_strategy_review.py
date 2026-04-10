"""
策略复盘报表 (run_strategy_review.py)
=====================================
完全自包含：无需前置运行 run_composite_factor。
根据 strategy_review_config 配置的五个因子、复合方式、策略参数，
自动从 factor_processed 读取 → 计算复合因子 → 运行策略回测 → 生成复盘报表。

输出 Excel（6 个 Sheet）：
  - Performance_Overview   : 策略 vs 基准全局绩效，含回测期/实盘期拆分
  - Rebalance_Period_Review: 逐调仓期收益、复合因子 IC、对比基准
  - Factor_Attribution     : 各单因子在各调仓期的 IC
  - Market_Regime_Analysis : 按基准涨跌幅分牛/平/熊市，统计策略各市况表现
  - Param_Sensitivity      : 参数网格对比（不同持仓周期/组数/权重方式）
  - Actual_vs_Model        : 券商成交记录 vs 模型建议对比（需 BROKER_RECORDS_FILE）

用法（项目根目录）：
  修改 analysis/strategy/strategy_review_config.py 后运行：
  python analysis/strategy/run_strategy_review.py
"""
import os
import sys
import io
from datetime import datetime

import numpy as np
import pandas as pd

# Windows UTF-8 兼容
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 路径注册 ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SF_DIR = os.path.join(_ROOT, "analysis", "single_factor")
_MF_DIR = os.path.join(_ROOT, "analysis", "multi_factor")

for _p in [_HERE, _SF_DIR, _MF_DIR, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_strategy import load_return_data
from run_detailed_backtest_report import run_detailed_backtest, parse_strategy_param
from strategy_utils import load_price_data
from analysis.single_factor.run_multi_factor_test import load_factor
from analysis.single_factor.rebalance_manager import RebalancePeriodManager
from analysis.multi_factor.composite_factor import compute_selected_composites

import strategy_config as cfg
from data.data_config import PRICE_FILE, DATA_START_OFFSET_DAYS

import strategy_review_config as rev_cfg

# 从 config 读取（便于函数内引用）
COMPOSITE_FACTOR_SHEET = rev_cfg.COMPOSITE_FACTOR_SHEET
STRATEGY_PARAM = rev_cfg.STRATEGY_PARAM
BENCHMARK_TICKER = rev_cfg.BENCHMARK_TICKER
BULL_THRESHOLD = rev_cfg.BULL_THRESHOLD
BEAR_THRESHOLD = rev_cfg.BEAR_THRESHOLD


# ---------------------------------------------------------------------------
# 因子加载与复合因子计算（自包含，无前置条件）
# ---------------------------------------------------------------------------

def load_factors_from_indices(
    factor_dir: str,
    indices: list[int],
) -> dict[str, pd.DataFrame]:
    """
    从 factor_processed 目录加载指定编号的因子。
    返回 {alpha032: DataFrame(date×stock), ...}
    """
    factor_dict = {}
    for i in indices:
        name = f"alpha{i:03d}"
        fpath = os.path.join(factor_dir, f"factor_{name}_processed.xlsx")
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"因子文件不存在: {fpath}")
        df = load_factor(fpath)
        df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric, errors="coerce")
        factor_dict[name] = df
    return factor_dict


def align_to_rebalance_periods(
    factor_dict: dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    rebalance_period: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    将日频因子与收益率对齐到调仓期截面。
    返回 (factor_periods_dict, ret_periods)
    """
    first_factor = next(iter(factor_dict.values()))
    manager = RebalancePeriodManager(first_factor, ret, rebalance_period)
    _, ret_periods = manager.align_factor_return_by_period()
    common_dates = ret_periods.index

    factor_periods_dict = {}
    for name, fdf in factor_dict.items():
        mgr = RebalancePeriodManager(fdf, ret, rebalance_period)
        fp, _ = mgr.align_factor_return_by_period()
        fp = fp.reindex(common_dates)
        factor_periods_dict[name] = fp

    return factor_periods_dict, ret_periods


def compute_composite_factor(
    factor_dir: str,
    indices: list[int],
    composite_sheet: str,
    rebalance_period: int,
    n_windows: list[int],
    m_windows: list[int],
    price_file: str,
    return_column: str = "Return",
) -> pd.DataFrame:
    """
    从 factor_processed 加载指定因子，计算复合因子，返回选定 sheet 的 DataFrame。
    """
    factor_dict = load_factors_from_indices(factor_dir, indices)
    ret = load_return_data(price_file, return_column)
    ret.sort_index(inplace=True)

    factor_periods_dict, ret_periods = align_to_rebalance_periods(
        factor_dict, ret, rebalance_period
    )

    # 扩展最后一期（与 run_composite_factor 一致）
    _first_daily = next(iter(factor_dict.values()))
    _mgr_last = RebalancePeriodManager(_first_daily, ret, rebalance_period)
    _all_daily_rb = _mgr_last.get_rebalance_dates()
    if _all_daily_rb:
        _last_rb = pd.Timestamp(_all_daily_rb[-1])
        _sample_periods = next(iter(factor_periods_dict.values()))
        if _last_rb not in _sample_periods.index:
            for _fname, _fdf_daily in factor_dict.items():
                if _fname not in factor_periods_dict:
                    continue
                _avail = _fdf_daily.index[_fdf_daily.index <= _last_rb]
                if len(_avail) == 0:
                    _new_row = pd.DataFrame(
                        [pd.Series(np.nan, index=_fdf_daily.columns)],
                        index=[_last_rb],
                    )
                else:
                    _sig = _avail[-1]
                    _new_row = _fdf_daily.loc[[_sig]].rename(index={_sig: _last_rb})
                factor_periods_dict[_fname] = pd.concat(
                    [factor_periods_dict[_fname], _new_row]
                )

    composite_dict = compute_selected_composites(
        factor_periods_dict,
        ret_periods,
        [composite_sheet],
        n_windows,
        m_windows,
    )
    if composite_sheet not in composite_dict:
        raise ValueError(
            f"复合因子 '{composite_sheet}' 不在计算结果中。"
            f"可选: {list(composite_dict.keys())}"
        )
    return composite_dict[composite_sheet]


# ---------------------------------------------------------------------------
# 基准数据加载
# ---------------------------------------------------------------------------

def load_benchmark_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    通过 yfinance 拉取基准（如 QQQ）的日线收盘价与日收益率。

    Returns
    -------
    prices : pd.Series  日线收盘价（index=date）
    returns : pd.Series 日收益率（index=date）
    """
    import yfinance as yf

    print(f"  拉取基准数据: {ticker} ({start_date} ~ {end_date or 'today'})")
    t = yf.Ticker(ticker)
    hist = t.history(start=start_date, end=end_date, auto_adjust=True)
    if hist.empty:
        print(f"  [Warning] 基准数据为空: {ticker}")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    prices = hist["Close"].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.sort_index(inplace=True)
    prices.name = ticker
    returns = prices.pct_change().dropna()
    returns.name = ticker
    return prices, returns


# ---------------------------------------------------------------------------
# 个别因子加载
# ---------------------------------------------------------------------------

def load_individual_factors(
    factor_processed_dir: str,
    factor_names: list[str],
) -> dict[str, pd.DataFrame]:
    """
    加载各单因子处理后文件，返回 {factor_name: DataFrame(index=date, columns=ticker)}。
    找不到的因子会打印警告并跳过。
    """
    factors: dict[str, pd.DataFrame] = {}
    if not os.path.isdir(factor_processed_dir):
        print(f"  [Warning] factor_processed 目录不存在: {factor_processed_dir}")
        return factors

    for name in factor_names:
        fname = f"factor_{name}_processed.xlsx"
        fpath = os.path.join(factor_processed_dir, fname)
        if not os.path.isfile(fpath):
            print(f"  [Warning] 因子文件不存在，跳过: {fpath}")
            continue
        df = pd.read_excel(fpath, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.sort_index(inplace=True)
        factors[name] = df
    return factors


# ---------------------------------------------------------------------------
# 绩效指标计算
# ---------------------------------------------------------------------------

def compute_metrics(
    daily_returns: pd.Series,
    rf: float = 0.02,
    label: str = "strategy",
) -> dict:
    """计算标准绩效指标字典。"""
    if daily_returns is None or len(daily_returns) == 0:
        return {
            "label": label,
            "Total_Return": np.nan,
            "Ann_Return": np.nan,
            "Ann_Vol": np.nan,
            "Sharpe": np.nan,
            "Max_Drawdown": np.nan,
            "Win_Rate_Daily": np.nan,
            "Trading_Days": 0,
            "Start_Date": "-",
            "End_Date": "-",
        }
    nav = (1 + daily_returns).cumprod()
    total_ret = float(nav.iloc[-1]) - 1.0
    n_days = len(daily_returns)
    ann_ret = (1 + total_ret) ** (252 / max(1, n_days)) - 1
    vol = float(daily_returns.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / vol if vol > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1).min())
    win_rate = float((daily_returns > 0).mean())
    return {
        "label": label,
        "Total_Return": total_ret,
        "Ann_Return": ann_ret,
        "Ann_Vol": vol,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "Win_Rate_Daily": win_rate,
        "Trading_Days": n_days,
        "Start_Date": str(daily_returns.index[0].date()),
        "End_Date": str(daily_returns.index[-1].date()),
    }


# ---------------------------------------------------------------------------
# IC 计算
# ---------------------------------------------------------------------------

def compute_ic_per_period(
    factor_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    period_summary_df: pd.DataFrame,
) -> pd.Series:
    """
    对每个调仓期计算 IC（Spearman 秩相关）：
    复合因子信号 vs. 全股票持仓期累计收益。

    Parameters
    ----------
    factor_df         : 复合因子 DataFrame（index=date, columns=ticker）
    ret_df            : 日收益率 DataFrame（index=date, columns=ticker）
    period_summary_df : run_detailed_backtest 返回的 period_summary_df

    Returns
    -------
    pd.Series（index=Rebalance_Date，name="IC"）
    """
    from scipy.stats import spearmanr

    ics: dict = {}
    periods = list(
        zip(
            period_summary_df["Rebalance_Date"],
            period_summary_df["Next_Rebalance_Date"],
        )
    )

    for rb_date, next_rb in periods:
        rb_date = pd.Timestamp(rb_date)
        next_rb = pd.Timestamp(next_rb)

        # 最近可用因子信号
        avail = factor_df.index[factor_df.index <= rb_date]
        if len(avail) == 0:
            ics[rb_date] = np.nan
            continue
        signal = factor_df.loc[avail[-1]].dropna()

        # 全部股票持仓期累计收益
        mask = (ret_df.index > rb_date) & (ret_df.index <= next_rb)
        period_r = ret_df.loc[mask]
        if len(period_r) == 0:
            ics[rb_date] = np.nan
            continue
        cum_ret = (1 + period_r).prod() - 1

        common = signal.index.intersection(cum_ret.index)
        valid = [(s, signal[s], cum_ret[s]) for s in common
                 if not np.isnan(signal[s]) and not np.isnan(cum_ret[s])]
        if len(valid) < 5:
            ics[rb_date] = np.nan
            continue

        f_vals = [v[1] for v in valid]
        r_vals = [v[2] for v in valid]
        corr, _ = spearmanr(f_vals, r_vals)
        ics[rb_date] = float(corr)

    return pd.Series(ics, name="IC")


# ---------------------------------------------------------------------------
# 因子归因
# ---------------------------------------------------------------------------

def compute_factor_attribution(
    individual_factors: dict[str, pd.DataFrame],
    ret_df: pd.DataFrame,
    period_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    对各单因子在各调仓期计算 IC，形成归因矩阵。

    Returns
    -------
    DataFrame（index=Rebalance_Date，columns=factor_names）
    """
    from scipy.stats import spearmanr

    if not individual_factors:
        return pd.DataFrame()

    periods = list(
        zip(
            period_summary_df["Rebalance_Date"],
            period_summary_df["Next_Rebalance_Date"],
        )
    )

    result: dict[str, dict] = {name: {} for name in individual_factors}

    for rb_date, next_rb in periods:
        rb_date = pd.Timestamp(rb_date)
        next_rb = pd.Timestamp(next_rb)

        # 持仓期累计收益（全股票，用于 IC 计算）
        mask = (ret_df.index > rb_date) & (ret_df.index <= next_rb)
        period_r = ret_df.loc[mask]
        if len(period_r) == 0:
            for name in individual_factors:
                result[name][rb_date] = np.nan
            continue
        cum_ret = (1 + period_r).prod() - 1

        for name, fdf in individual_factors.items():
            avail = fdf.index[fdf.index <= rb_date]
            if len(avail) == 0:
                result[name][rb_date] = np.nan
                continue
            signal = fdf.loc[avail[-1]].dropna()
            common = signal.index.intersection(cum_ret.index)
            valid = [(signal[s], cum_ret[s]) for s in common
                     if not np.isnan(signal[s]) and not np.isnan(cum_ret[s])]
            if len(valid) < 5:
                result[name][rb_date] = np.nan
                continue
            f_vals = [v[0] for v in valid]
            r_vals = [v[1] for v in valid]
            corr, _ = spearmanr(f_vals, r_vals)
            result[name][rb_date] = float(corr)

    df = pd.DataFrame(result)
    df.index.name = "Rebalance_Date"
    df["IC_Mean"] = df.mean(axis=1)
    return df


# ---------------------------------------------------------------------------
# 市场状态分析
# ---------------------------------------------------------------------------

def analyze_market_regime(
    benchmark_returns: pd.Series,
    period_summary_df: pd.DataFrame,
    bull_threshold: float = BULL_THRESHOLD,
    bear_threshold: float = BEAR_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    为每个调仓期标注市场状态（牛/平/熊），并统计各状态下策略表现。

    Returns
    -------
    period_df  : 含 Benchmark_Period_Return、Regime 列的明细 DataFrame
    summary_df : 各 Regime 的聚合统计 DataFrame
    """
    records = []
    for _, row in period_summary_df.iterrows():
        rb_date = pd.Timestamp(row["Rebalance_Date"])
        next_rb = pd.Timestamp(row["Next_Rebalance_Date"])
        strat_ret = row["Period_Return"]

        # 基准同期累计收益
        mask = (benchmark_returns.index > rb_date) & (benchmark_returns.index <= next_rb)
        bm_slice = benchmark_returns.loc[mask]
        bm_ret = float((1 + bm_slice).prod() - 1) if len(bm_slice) > 0 else np.nan

        if not np.isnan(bm_ret):
            if bm_ret >= bull_threshold:
                regime = "Bull"
            elif bm_ret <= bear_threshold:
                regime = "Bear"
            else:
                regime = "Flat"
        else:
            regime = "Unknown"

        records.append({
            "Rebalance_Date": rb_date,
            "Next_Rebalance_Date": next_rb,
            "Holding_Days": row["Holding_Days"],
            "Strategy_Return": strat_ret,
            "Benchmark_Return": bm_ret,
            "Excess_Return": strat_ret - bm_ret if not np.isnan(bm_ret) else np.nan,
            "Regime": regime,
        })

    period_df = pd.DataFrame(records)

    # 汇总统计
    summary_rows = []
    for regime in ["Bull", "Flat", "Bear", "Unknown"]:
        sub = period_df[period_df["Regime"] == regime]
        if len(sub) == 0:
            continue
        s_rets = sub["Strategy_Return"].dropna()
        b_rets = sub["Benchmark_Return"].dropna()
        e_rets = sub["Excess_Return"].dropna()
        summary_rows.append({
            "Regime": regime,
            "Period_Count": len(sub),
            "Strategy_Avg_Return": s_rets.mean() if len(s_rets) > 0 else np.nan,
            "Strategy_Win_Rate": (s_rets > 0).mean() if len(s_rets) > 0 else np.nan,
            "Strategy_Best_Period": s_rets.max() if len(s_rets) > 0 else np.nan,
            "Strategy_Worst_Period": s_rets.min() if len(s_rets) > 0 else np.nan,
            "Benchmark_Avg_Return": b_rets.mean() if len(b_rets) > 0 else np.nan,
            "Avg_Excess_Return": e_rets.mean() if len(e_rets) > 0 else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)

    return period_df, summary_df


# ---------------------------------------------------------------------------
# 参数敏感度
# ---------------------------------------------------------------------------

def run_param_sensitivity(
    factor_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    price_df: pd.DataFrame,
    param_grid: list[str],
    rf: float = 0.02,
) -> pd.DataFrame:
    """
    对 param_grid 中每个策略参数配置运行回测，汇总绩效指标。

    Returns
    -------
    DataFrame（每行一个参数配置，columns=各绩效指标）
    """
    rows = []
    total = len(param_grid)
    for idx, param in enumerate(param_grid, 1):
        print(f"  参数敏感度 [{idx}/{total}]: {param}")
        try:
            wm, gn, tr, rd = parse_strategy_param(param)
        except ValueError as e:
            print(f"    [Error] 参数解析失败: {e}")
            continue

        result = run_detailed_backtest(
            factor_df=composite_factor_df,
            ret_df=ret_df,
            price_df=price_df,
            group_num=gn,
            target_rank=tr,
            rebalance_period=rd,
            weight_method=wm,
            config=cfg,
        )
        if "error" in result:
            print(f"    [Error] 回测失败: {result['error']}")
            continue

        m = compute_metrics(result["daily_returns"], rf=rf, label=param)
        n_rb = len(result["rebalance_dates"])
        rows.append({
            "Strategy_Param": param,
            "Weight_Method": wm,
            "Group_Num": gn,
            "Target_Rank": tr,
            "Rebalance_Period_Days": rd,
            "Total_Return": m["Total_Return"],
            "Ann_Return": m["Ann_Return"],
            "Ann_Vol": m["Ann_Vol"],
            "Sharpe": m["Sharpe"],
            "Max_Drawdown": m["Max_Drawdown"],
            "Win_Rate_Daily": m["Win_Rate_Daily"],
            "Rebalance_Count": n_rb,
            "Start_Date": m["Start_Date"],
            "End_Date": m["End_Date"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 券商记录加载与对比
# ---------------------------------------------------------------------------

def load_broker_records(file_path: str) -> pd.DataFrame:
    """
    加载券商成交记录，标准化列名为：Date, Symbol, Side, Price, Shares。

    支持列名（大小写不敏感）：
      Date    : date, trade_date, tradedate, execution_date, filled_date
      Symbol  : symbol, ticker, stock
      Side    : side, action, type, transaction  (值应含 buy/sell)
      Price   : price, executed_price, fill_price, avg_price, filled_price
      Shares  : shares, quantity, qty, volume, filled_shares
    """
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.lower().strip().replace(" ", "_")
        if cl in ("date", "trade_date", "tradedate", "execution_date", "filled_date"):
            col_map[col] = "Date"
        elif cl in ("symbol", "ticker", "stock"):
            col_map[col] = "Symbol"
        elif cl in ("side", "action", "type", "transaction"):
            col_map[col] = "Side"
        elif cl in ("price", "executed_price", "fill_price", "avg_price", "filled_price"):
            col_map[col] = "Price"
        elif cl in ("shares", "quantity", "qty", "volume", "filled_shares"):
            col_map[col] = "Shares"
    df = df.rename(columns=col_map)

    required = {"Date", "Symbol", "Side", "Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"券商记录缺少必要列: {missing}\n"
            f"实际列名: {list(df.columns)}\n"
            "请将对应列名改为 Date / Symbol / Side / Price（大小写不限）"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df["Side"] = df["Side"].astype(str).str.lower().str.strip()
    if "Shares" not in df.columns:
        df["Shares"] = np.nan

    return df[["Date", "Symbol", "Side", "Price", "Shares"]].reset_index(drop=True)


def compare_actual_vs_model(
    broker_df: pd.DataFrame,
    model_ops_df: pd.DataFrame,
    date_tolerance_days: int = 3,
) -> pd.DataFrame:
    """
    将券商成交记录与模型操作明细对齐，计算买卖滑点与时间偏差。

    匹配规则：Symbol 完全匹配 + 成交日期在模型日期 ±date_tolerance_days 内。
    若一个 Symbol 在同一期有多笔成交，取价格加权均价。

    Returns
    -------
    DataFrame，每行对应模型操作中的一只股票一个调仓期：
      Rebalance_Date, Symbol, Model_Weight,
      Model_Buy_Price, Actual_Buy_Price, Buy_Slippage_Pct, Actual_Buy_Date, Buy_Timing_Diff_Days,
      Model_Sell_Price, Actual_Sell_Price, Sell_Slippage_Pct, Actual_Sell_Date, Sell_Timing_Diff_Days,
      Model_Period_Return
    """
    records = []
    tol = pd.Timedelta(days=date_tolerance_days)

    for _, model_row in model_ops_df.iterrows():
        rb_date = pd.Timestamp(model_row["Rebalance_Date"])
        next_rb = pd.Timestamp(model_row["Next_Rebalance_Date"])
        symbol = str(model_row["Symbol"]).upper().strip()
        model_buy = model_row.get("Buy_Price_Close", np.nan)
        model_sell = model_row.get("Sell_Price_Close", np.nan)
        model_weight = model_row.get("Weight", np.nan)

        # 匹配买入
        buy_mask = (
            (broker_df["Symbol"] == symbol)
            & (broker_df["Side"].isin(["buy", "b", "bought"]))
            & (broker_df["Date"] >= rb_date - tol)
            & (broker_df["Date"] <= rb_date + tol)
        )
        buy_trades = broker_df[buy_mask]

        # 匹配卖出
        sell_mask = (
            (broker_df["Symbol"] == symbol)
            & (broker_df["Side"].isin(["sell", "s", "sold"]))
            & (broker_df["Date"] >= next_rb - tol)
            & (broker_df["Date"] <= next_rb + tol)
        )
        sell_trades = broker_df[sell_mask]

        def _vwap(trades: pd.DataFrame) -> tuple:
            if len(trades) == 0:
                return np.nan, pd.NaT
            shares = trades["Shares"].fillna(1.0)
            avg_price = float((trades["Price"] * shares).sum() / shares.sum())
            first_date = trades["Date"].iloc[0]
            return avg_price, first_date

        actual_buy, actual_buy_date = _vwap(buy_trades)
        actual_sell, actual_sell_date = _vwap(sell_trades)

        def _slippage(actual, model):
            if np.isnan(actual) or np.isnan(model) or model == 0:
                return np.nan
            return (actual - model) / model * 100

        def _timing_diff(actual_date, model_date):
            if pd.isna(actual_date):
                return np.nan
            return (actual_date - model_date).days

        records.append({
            "Rebalance_Date": rb_date,
            "Next_Rebalance_Date": next_rb,
            "Symbol": symbol,
            "Model_Weight": model_weight,
            "Model_Buy_Price": model_buy,
            "Actual_Buy_Price": actual_buy,
            "Buy_Slippage_Pct": _slippage(actual_buy, model_buy),
            "Actual_Buy_Date": actual_buy_date,
            "Buy_Timing_Diff_Days": _timing_diff(actual_buy_date, rb_date),
            "Model_Sell_Price": model_sell,
            "Actual_Sell_Price": actual_sell,
            "Sell_Slippage_Pct": _slippage(actual_sell, model_sell),
            "Actual_Sell_Date": actual_sell_date,
            "Sell_Timing_Diff_Days": _timing_diff(actual_sell_date, next_rb),
            "Model_Period_Return": model_row.get("Period_Return", np.nan),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 报表写入
# ---------------------------------------------------------------------------

def _fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v * 100:.{decimals}f}%"


def _fmt_f(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{decimals}f}"


def write_review_report(
    result: dict,
    benchmark_prices: pd.Series,
    benchmark_returns: pd.Series,
    composite_factor_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    individual_factors: dict,
    param_sensitivity_df: pd.DataFrame | None,
    actual_vs_model_df: pd.DataFrame | None,
    live_start_date: pd.Timestamp | None,
    output_path: str,
) -> None:
    """将所有分析结果写入多 Sheet Excel 复盘报表。"""

    daily_returns = result["daily_returns"]
    nav = result["nav"]
    period_summary_df = result["period_summary_df"].copy()

    # ── Sheet 1: Performance_Overview ─────────────────────────────────────────
    def _overview_rows(dr: pd.Series, bm_r: pd.Series, label: str) -> list:
        """构建绩效对比行（策略 vs 基准）。"""
        if len(dr) == 0:
            return []
        bm_aligned = bm_r.reindex(dr.index).fillna(0)
        m_strat = compute_metrics(dr, rf=cfg.RISK_FREE_RATE, label="strategy")
        m_bm = compute_metrics(bm_aligned, rf=cfg.RISK_FREE_RATE, label="benchmark")
        bm_nav = (1 + bm_aligned).cumprod()
        bm_total = float(bm_nav.iloc[-1]) - 1.0
        bm_ann = (1 + bm_total) ** (252 / max(1, len(bm_aligned))) - 1

        return [
            [f"=== {label} ===", ""],
            ["Period", f"{dr.index[0].date()} ~ {dr.index[-1].date()}"],
            ["Trading_Days", m_strat["Trading_Days"]],
            ["", ""],
            ["Metric", f"Strategy vs {BENCHMARK_TICKER}"],
            ["Total_Return", f"{_fmt_pct(m_strat['Total_Return'])}  vs  {_fmt_pct(bm_total)}"],
            ["Ann_Return", f"{_fmt_pct(m_strat['Ann_Return'])}  vs  {_fmt_pct(bm_ann)}"],
            ["Ann_Vol", f"{_fmt_pct(m_strat['Ann_Vol'])}  vs  {_fmt_pct(m_bm['Ann_Vol'])}"],
            ["Sharpe", f"{_fmt_f(m_strat['Sharpe'], 3)}  vs  {_fmt_f(m_bm['Sharpe'], 3)}"],
            ["Max_Drawdown", f"{_fmt_pct(m_strat['Max_Drawdown'])}  vs  {_fmt_pct(m_bm['Max_Drawdown'])}"],
            ["Win_Rate_Daily", f"{_fmt_pct(m_strat['Win_Rate_Daily'])}  vs  {_fmt_pct(m_bm['Win_Rate_Daily'])}"],
            ["", ""],
        ]

    overview_rows = [
        ["Selected_Factor_Indices", str(rev_cfg.SELECTED_FACTOR_INDICES)],
        ["Strategy_Param", STRATEGY_PARAM],
        ["Composite_Factor_Sheet", COMPOSITE_FACTOR_SHEET],
        ["Benchmark", BENCHMARK_TICKER],
        ["Data_Start_Offset_Days", DATA_START_OFFSET_DAYS],
        ["Report_Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["", ""],
    ]
    overview_rows += _overview_rows(daily_returns, benchmark_returns, "Full Period")

    if live_start_date is not None:
        bt_mask = daily_returns.index < live_start_date
        live_mask = daily_returns.index >= live_start_date
        if bt_mask.sum() > 0:
            overview_rows += _overview_rows(
                daily_returns[bt_mask], benchmark_returns, f"Backtest (< {live_start_date.date()})"
            )
        if live_mask.sum() > 0:
            overview_rows += _overview_rows(
                daily_returns[live_mask], benchmark_returns, f"Live (>= {live_start_date.date()})"
            )

    df_overview = pd.DataFrame(overview_rows, columns=["Metric", "Value"])

    # ── Sheet 2: Rebalance_Period_Review ──────────────────────────────────────
    ic_series = compute_ic_per_period(composite_factor_df, ret_df, period_summary_df)

    period_review_rows = []
    cum_nav = 1.0
    for _, row in period_summary_df.iterrows():
        rb_date = pd.Timestamp(row["Rebalance_Date"])
        next_rb = pd.Timestamp(row["Next_Rebalance_Date"])
        strat_ret = row["Period_Return"]
        cum_nav *= 1.0 + strat_ret

        # 基准同期
        bm_mask = (benchmark_returns.index > rb_date) & (benchmark_returns.index <= next_rb)
        bm_slice = benchmark_returns.loc[bm_mask]
        bm_ret = float((1 + bm_slice).prod() - 1) if len(bm_slice) > 0 else np.nan

        is_live = (live_start_date is not None) and (rb_date >= live_start_date)

        period_review_rows.append({
            "Rebalance_Date": rb_date,
            "Next_Rebalance_Date": next_rb,
            "Holding_Days": row["Holding_Days"],
            "Strategy_Period_Return": strat_ret,
            "Strategy_Cum_NAV": cum_nav,
            "Composite_IC": ic_series.get(rb_date, np.nan),
            f"{BENCHMARK_TICKER}_Period_Return": bm_ret,
            "Excess_Return": strat_ret - bm_ret if not np.isnan(bm_ret) else np.nan,
            "Position_Count": row["Position_Count"],
            "Symbols": row["Symbols"],
            "Is_Live": is_live,
        })
    df_period_review = pd.DataFrame(period_review_rows)

    # ── Sheet 3: Factor_Attribution ───────────────────────────────────────────
    df_attribution = compute_factor_attribution(individual_factors, ret_df, period_summary_df)

    # ── Sheet 4: Market_Regime_Analysis ───────────────────────────────────────
    df_regime_detail, df_regime_summary = analyze_market_regime(
        benchmark_returns, period_summary_df
    )

    # ── Sheet 5: Param_Sensitivity ────────────────────────────────────────────
    df_sensitivity = param_sensitivity_df if param_sensitivity_df is not None else pd.DataFrame()

    # ── Sheet 6: Actual_vs_Model ──────────────────────────────────────────────
    df_actual = actual_vs_model_df if actual_vs_model_df is not None else pd.DataFrame()

    # ── 写入 ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_overview.to_excel(writer, sheet_name="Performance_Overview", index=False)
        df_period_review.to_excel(writer, sheet_name="Rebalance_Period_Review", index=False)
        if len(df_attribution) > 0:
            df_attribution.reset_index().to_excel(writer, sheet_name="Factor_Attribution", index=False)
        else:
            pd.DataFrame([["Factor processed files not found or attribution skipped"]],
                         columns=["Note"]).to_excel(writer, sheet_name="Factor_Attribution", index=False)
        df_regime_detail.to_excel(writer, sheet_name="Market_Regime_Detail", index=False)
        df_regime_summary.to_excel(writer, sheet_name="Market_Regime_Summary", index=False)
        if len(df_sensitivity) > 0:
            df_sensitivity.to_excel(writer, sheet_name="Param_Sensitivity", index=False)
        else:
            pd.DataFrame([["Param sensitivity skipped (strategy_review_config.RUN_PARAM_SENSITIVITY=False)"]],
                         columns=["Note"]).to_excel(writer, sheet_name="Param_Sensitivity", index=False)
        if len(df_actual) > 0:
            df_actual.to_excel(writer, sheet_name="Actual_vs_Model", index=False)
        else:
            pd.DataFrame([["No broker records provided (set strategy_review_config.BROKER_RECORDS_FILE)"]],
                         columns=["Note"]).to_excel(writer, sheet_name="Actual_vs_Model", index=False)

    print(f"  复盘报表已写入: {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    # 从 strategy_review_config 读取配置
    broker_records_file = rev_cfg.BROKER_RECORDS_FILE
    run_sensitivity = rev_cfg.RUN_PARAM_SENSITIVITY
    factor_dir = rev_cfg.get_factor_processed_dir()
    run_dir = rev_cfg.get_output_dir()

    live_start: pd.Timestamp | None = None
    raw_live = rev_cfg.LIVE_START_DATE
    if raw_live:
        try:
            live_start = pd.Timestamp(raw_live)
        except Exception:
            print(f"[Warning] live_start_date 格式无效: {raw_live}，已忽略")

    os.makedirs(run_dir, exist_ok=True)

    weight_method, group_num, target_rank, rebalance_days = parse_strategy_param(STRATEGY_PARAM)

    print("=" * 68)
    print("  策略复盘报表")
    print(f"  策略: {STRATEGY_PARAM} | 复合因子: {COMPOSITE_FACTOR_SHEET}")
    if live_start:
        print(f"  实盘起始: {live_start.date()}")
    print(f"  输出目录: {run_dir}")
    print("=" * 68)

    # 1. 加载五因子并计算复合因子（无前置条件）
    rebalance_period = rev_cfg.get_rebalance_period()
    print(f"\n[1/8] 加载因子 {rev_cfg.SELECTED_FACTOR_INDICES} 并计算复合因子 {COMPOSITE_FACTOR_SHEET}...")
    factor_df = compute_composite_factor(
        factor_dir=factor_dir,
        indices=rev_cfg.SELECTED_FACTOR_INDICES,
        composite_sheet=COMPOSITE_FACTOR_SHEET,
        rebalance_period=rebalance_period,
        n_windows=rev_cfg.N_WINDOWS,
        m_windows=rev_cfg.M_WINDOWS,
        price_file=PRICE_FILE,
        return_column=cfg.RETURN_COLUMN,
    )
    composite_factor_df = factor_df  # 重命名以区分复合因子与单因子
    print(f"      区间: {composite_factor_df.index[0].date()} ~ {composite_factor_df.index[-1].date()}")

    # 2. 加载日频收益率
    print("\n[2/8] 加载日频收益率...")
    ret_df = load_return_data(PRICE_FILE, cfg.RETURN_COLUMN)
    ret_df.sort_index(inplace=True)
    print(f"      区间: {ret_df.index[0].date()} ~ {ret_df.index[-1].date()}")

    # 3. 加载价格数据
    print("\n[3/8] 加载价格数据（Adj Close）...")
    price_df = load_price_data(PRICE_FILE, "Adj Close")
    print(f"      区间: {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    # 4. 加载基准数据
    print(f"\n[4/8] 加载基准数据（{BENCHMARK_TICKER}）...")
    data_start = str(composite_factor_df.index[0].date())
    benchmark_prices, benchmark_returns = load_benchmark_data(BENCHMARK_TICKER, data_start)
    if len(benchmark_prices) == 0:
        print(f"  [Warning] 基准数据为空，相关分析将跳过")
        benchmark_prices = pd.Series(dtype=float)
        benchmark_returns = pd.Series(dtype=float)

    # 5. 运行策略回测
    print(f"\n[5/8] 运行策略回测: {STRATEGY_PARAM}...")
    result = run_detailed_backtest(
        factor_df=composite_factor_df,
        ret_df=ret_df,
        price_df=price_df,
        group_num=group_num,
        target_rank=target_rank,
        rebalance_period=rebalance_days,
        weight_method=weight_method,
        config=cfg,
    )
    if "error" in result:
        print(f"[Error] 回测失败: {result['error']}")
        return
    print(f"      共 {len(result['rebalance_dates'])} 次调仓，"
          f"{len(result['operations_df'])} 条操作记录")

    # 6. 加载单因子文件（用于因子归因）
    selected_names = rev_cfg.get_selected_factor_names()
    print(f"\n[6/8] 加载单因子文件（{factor_dir}）...")
    individual_factors = load_individual_factors(factor_dir, selected_names)
    print(f"      成功加载 {len(individual_factors)}/{len(selected_names)} 个因子: "
          f"{list(individual_factors.keys())}")

    # 7. 参数敏感度
    param_sensitivity_df = None
    if run_sensitivity:
        print(f"\n[7/8] 参数敏感度分析（{len(rev_cfg.PARAM_GRID)} 个配置）...")
        param_sensitivity_df = run_param_sensitivity(
            factor_df=composite_factor_df,
            ret_df=ret_df,
            price_df=price_df,
            param_grid=rev_cfg.PARAM_GRID,
            rf=cfg.RISK_FREE_RATE,
        )
    else:
        print("\n[7/8] 参数敏感度已跳过（strategy_review_config.RUN_PARAM_SENSITIVITY=False）")

    # 8. 券商记录对比
    actual_vs_model_df = None
    if broker_records_file:
        print(f"\n[+] 加载券商成交记录: {broker_records_file}...")
        try:
            broker_df = load_broker_records(broker_records_file)
            print(f"    共 {len(broker_df)} 条成交记录")
            actual_vs_model_df = compare_actual_vs_model(broker_df, result["operations_df"])
            matched = actual_vs_model_df["Actual_Buy_Price"].notna().sum()
            print(f"    匹配到 {matched}/{len(actual_vs_model_df)} 条买入记录")
        except Exception as e:
            print(f"  [Error] 券商记录处理失败: {e}")
    else:
        print("\n[+] 未指定券商记录，跳过 Actual_vs_Model 分析")

    # 9. 写入报表
    print("\n[写入] 生成复盘报表...")
    output_path = os.path.join(run_dir, "strategy_review.xlsx")
    write_review_report(
        result=result,
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns,
        factor_df=composite_factor_df,
        ret_df=ret_df,
        individual_factors=individual_factors,
        param_sensitivity_df=param_sensitivity_df,
        actual_vs_model_df=actual_vs_model_df,
        live_start_date=live_start,
        output_path=output_path,
    )

    print("\n" + "=" * 68)
    print(f"  复盘完成！")
    print(f"  报表路径: {output_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
