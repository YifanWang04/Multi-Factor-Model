"""
调仓日全流程与报表 (run_rebalance_day.py)
=============================================
完整 Pipeline：pull_data → build_factors → data_process → run_composite_factor
使用固定策略参数生成持仓，输出调仓日判定、当前调仓日操作及未来调仓日列表。

所有输出保存至带日期时间的独立文件夹：output/rebalance_day_YYYY-MM-DD_HHMMSS/
  - data/                    # pull_data 输出
  - factor_raw/              # build_factors 输出
  - factor_processed/        # data_process 输出
  - composite_factor_reports/ # run_composite_factor 输出
  - rebalance_day_report.xlsx # 本脚本报表

时序约定（与 CLAUDE.md 一致）：
  - 交易：T 日收盘执行，买卖价格均使用 Adj Close（收盘价）
  - 持仓区间：(T, T_next]，T 日收益不计入当期持仓

用法（项目根目录）：
  python analysis/strategy/run_rebalance_day.py
  python analysis/strategy/run_rebalance_day.py --no-discord  # 不发送 Discord 通知
"""

import os
import sys
import io
import subprocess
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import requests

# 设置 UTF-8 输出（Windows 兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── 路径注册 ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SF_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "analysis", "single_factor")
_ROOT = os.path.dirname(os.path.dirname(_HERE))

for _p in [_HERE, _SF_DIR, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_strategy import load_composite_factor, load_return_data
from run_detailed_backtest_report import run_detailed_backtest
from strategy_backtest import _build_groups, _select_rebalance_dates
from portfolio_optimizer import compute_weights
import strategy_config as cfg


# ---------------------------------------------------------------------------
# 配置（本脚本独立配置，不依赖 run_detailed_backtest_report）
# ---------------------------------------------------------------------------

PROJECT_ROOT = r"D:\qqq"
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output")

# 复合因子
COMPOSITE_FACTOR_SHEET = "beta_m1"

# 选定因子（与 composite_config.SELECTED_FACTOR_INDICES [20, 16, 43, 17, 34] 对应）
# 仅构建和处理这 5 个因子，不创建多余因子
SELECTED_FACTOR_NAMES = ["alpha095", "alpha032", "alpha042", "alpha020", "alpha073"]

# 策略参数：COMPOSITE_FACTOR_SHEET = "mvo_10G_Top1_P30d"
TARGET_WEIGHT_METHOD = "mvo"
TARGET_GROUP_NUM = 10
TARGET_RANK = 1
TARGET_REBALANCE_DAYS = 30

# 调仓日偏移（天数）：正数=提前，负数=延后
# 例如：REBALANCE_DATE_OFFSET = 6 表示所有调仓日提前6天
REBALANCE_DATE_OFFSET = 6  # 将下一调仓日从 2026-03-11 提前到 2026-03-05

STRATEGY_PARAMS = {
    "weight_method": TARGET_WEIGHT_METHOD,
    "group_num": TARGET_GROUP_NUM,
    "target_rank": TARGET_RANK,
    "rebalance_period": TARGET_REBALANCE_DAYS,
}


def _strategy_name() -> str:
    """根据配置生成策略名称，如 mvo_10G_Top1_P30d。"""
    return f"{TARGET_WEIGHT_METHOD}_{TARGET_GROUP_NUM}G_Top{TARGET_RANK}_P{TARGET_REBALANCE_DAYS}d"


# Discord Webhook URL
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1478641216659652709/TRe7zHYv0x5AbYJMngnJbi1TbjUwXiOhIct-rze0wHFFYgi-Yqt320iGOCY4J1NUbq68"


def _get_run_dir(skip_pipeline: bool, run_dir_arg: str | None) -> str:
    """获取本次运行的输出目录。"""
    if run_dir_arg:
        return os.path.abspath(run_dir_arg)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.join(OUTPUT_BASE, f"rebalance_day_{ts}")


# ---------------------------------------------------------------------------
# 数据加载（本脚本自有实现）
# ---------------------------------------------------------------------------

def load_price_data(price_file: str, price_column: str = "Adj Close") -> pd.DataFrame:
    """加载日频价格数据，返回宽表 DataFrame(index=日期, columns=股票代码)。"""
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


# ---------------------------------------------------------------------------
# Pipeline：执行数据与因子流水线
# ---------------------------------------------------------------------------

def run_pipeline_subprocess(run_dir: str, skip_pull: bool = False) -> None:
    """通过 subprocess 依次调用各步骤，输出写入 run_dir。"""
    import shutil

    env = os.environ.copy()
    env["REBALANCE_RUN_DIR"] = run_dir
    env["REBALANCE_SELECTED_FACTORS"] = ",".join(SELECTED_FACTOR_NAMES)
    env["REBALANCE_SELECTED_COMPOSITE"] = COMPOSITE_FACTOR_SHEET

    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "factor_raw"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "factor_processed"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "composite_factor_reports"), exist_ok=True)

    if skip_pull:
        # 复制项目默认数据到 run_dir，供后续步骤使用
        src = os.path.join(PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
        dst = os.path.join(data_dir, "us_top100_daily_2023_present.xlsx")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  已复制数据至: {dst}")
        else:
            raise FileNotFoundError(f"skip_pull 时需存在 {src}")

    steps = []
    if not skip_pull:
        steps.append(("data/pull_yhfinance_Data.py", "拉取行情数据"))
    steps.extend([
        ("pipeline/build_factors.py", "构建因子"),
        ("pipeline/data_process.py", "因子数据处理"),
        ("analysis/multi_factor/run_composite_factor.py", "因子复合"),
    ])
    for i, (script, desc) in enumerate(steps, 1):
        print(f"[Pipeline {i}/{len(steps)}] {desc}...")
        r = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_ROOT,
            env=env,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Pipeline 步骤失败: {script}")


# ---------------------------------------------------------------------------
# 调仓日判定与未来调仓日推算
# ---------------------------------------------------------------------------

def get_rebalance_day_status(
    rebalance_dates: list,
    rebalance_period: int,
    as_of_date: pd.Timestamp,
    last_factor_date: pd.Timestamp,
) -> dict:
    """判定调仓日状态。"""
    rebalance_dates = sorted(rebalance_dates)
    if not rebalance_dates:
        return {
            "is_rebalance_today": False,
            "current_rebalance_date": None,
            "next_rebalance_date": None,
            "future_rebalance_dates": [],
        }

    past = [d for d in rebalance_dates if d <= as_of_date]
    current_rebalance_date = past[-1] if past else None
    is_rebalance_today = current_rebalance_date is not None and (
        current_rebalance_date.date() == as_of_date.date()
    )
    future_in_data = [d for d in rebalance_dates if d > as_of_date]
    next_rebalance_date = future_in_data[0] if future_in_data else None

    last_rb = rebalance_dates[-1]
    future_dates = []
    d = last_rb
    while d <= as_of_date + timedelta(days=rebalance_period * 5):
        d = d + timedelta(days=rebalance_period)
        # >= 确保「下一调仓日恰好是今天」时也能正确显示（不会被跳过到再下一期）
        if d >= as_of_date:
            future_dates.append(d)
        if len(future_dates) >= 12:
            break

    # 数据内无未来调仓日时，用 extrapolated future_dates 推算下一调仓日
    if next_rebalance_date is None and future_dates:
        next_rebalance_date = future_dates[0]

    # 当推算出的下一调仓日就是今天时，今日也应视为调仓日（数据尚未包含今日时 current 为上一期）
    if next_rebalance_date is not None and next_rebalance_date.date() == as_of_date.date():
        is_rebalance_today = True
        current_rebalance_date = pd.Timestamp(as_of_date.date())  # 语义上今日即为当前调仓日

    return {
        "is_rebalance_today": is_rebalance_today,
        "current_rebalance_date": current_rebalance_date,
        "next_rebalance_date": next_rebalance_date,
        "future_rebalance_dates": future_dates,
        "all_rebalance_dates": rebalance_dates,
    }


def _get_price_on_date(price_df: pd.DataFrame, date: pd.Timestamp, stocks: list) -> pd.Series:
    """获取指定日期各标的收盘价，缺失则前向填充。"""
    if date not in price_df.index:
        idx = price_df.index[price_df.index <= date]
        if len(idx) == 0:
            return pd.Series(dtype=float)
        date = idx[-1]
    row = price_df.loc[date]
    return row.reindex(stocks).dropna()


# ---------------------------------------------------------------------------
# 当前调仓日操作明细
# ---------------------------------------------------------------------------

def get_current_rebalance_operations(
    result: dict,
    current_rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    获取当前调仓日操作：卖出上一期持仓 + 买入本期标的。
    返回含 Action 列（Sell/Buy）的 DataFrame，先卖后买。
    """
    if "error" in result:
        return pd.DataFrame()

    df = result.get("operations_df", pd.DataFrame())
    sell_ops = pd.DataFrame()
    buy_ops = pd.DataFrame()

    # 1. 卖出操作：上一调仓日买入、今日卖出的持仓
    if not df.empty and "Next_Rebalance_Date" in df.columns:
        mask_sell = df["Next_Rebalance_Date"] == current_rebalance_date
        sell_ops = df.loc[mask_sell].copy()
        if not sell_ops.empty:
            sell_ops.insert(0, "Action", "Sell")

    # 2. 买入操作：今日买入的标的
    if not df.empty and "Rebalance_Date" in df.columns:
        mask_buy = df["Rebalance_Date"] == current_rebalance_date
        buy_ops = df.loc[mask_buy].copy()
        if not buy_ops.empty:
            buy_ops.insert(0, "Action", "Buy")
    if buy_ops.empty:
        buy_ops = _compute_last_rebalance_ops(
            factor_df=result.get("_factor_df"),
            ret_df=result.get("_ret_df"),
            price_df=result.get("_price_df"),
            rb_date=current_rebalance_date,
            config=result.get("_config"),
            group_num=STRATEGY_PARAMS["group_num"],
            target_rank=STRATEGY_PARAMS["target_rank"],
            weight_method=STRATEGY_PARAMS["weight_method"],
        )
        if not buy_ops.empty:
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
    factor_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    price_df: pd.DataFrame,
    rb_date: pd.Timestamp,
    config,
    group_num: int,
    target_rank: int,
    weight_method: str,
) -> pd.DataFrame:
    """对最后一个调仓日（无 next_rb）计算买入操作明细。"""
    if factor_df is None or ret_df is None or price_df is None or config is None:
        return pd.DataFrame()
    target_group = group_num - (target_rank - 1)
    lookback = getattr(config, "OPTIMIZATION_LOOKBACK", 252)
    rf = getattr(config, "RISK_FREE_RATE", 0.02)
    max_weight = getattr(config, "MAX_WEIGHT", 0.4)

    if rb_date in factor_df.index:
        signal_date = rb_date
    else:
        avail = factor_df.index[factor_df.index <= rb_date]
        if len(avail) == 0:
            return pd.DataFrame()
        signal_date = avail[-1]

    factor_signal = factor_df.loc[signal_date]
    groups = _build_groups(factor_signal, group_num)
    if target_group not in groups or len(groups[target_group]) == 0:
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
        return pd.DataFrame()

    w = weights.reindex(valid_stocks).fillna(0)
    w = w / w.sum()
    buy_p = buy_prices.reindex(valid_stocks).dropna()
    common = w.index.intersection(buy_p.index)
    if len(common) == 0:
        return pd.DataFrame()
    w = w[common] / w[common].sum()

    records = []
    for sym in common:
        bp = float(buy_p[sym])
        factor_val = factor_signal[sym] if sym in factor_signal.index else np.nan
        wt = float(w[sym])
        buy_value = wt * 1.0
        shares = buy_value / bp if bp > 0 else np.nan
        records.append({
            "Rebalance_Date": rb_date,
            "Next_Rebalance_Date": pd.NaT,
            "Holding_Days": np.nan,
            "Symbol": sym,
            "Weight": wt,
            "Buy_Price_Close": bp,
            "Sell_Price_Close": np.nan,
            "Period_Return": np.nan,
            "Buy_Value": buy_value,
            "Sell_Value": np.nan,
            "Shares": shares,
            "Factor_Value": factor_val,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 写入 Excel 报表
# ---------------------------------------------------------------------------

def write_rebalance_day_report(
    result: dict,
    status: dict,
    current_ops: pd.DataFrame,
    output_path: str,
) -> None:
    """写入调仓日报表。"""
    if "error" in result:
        raise ValueError(result["error"])

    params = result.get("params", {})
    as_of = pd.Timestamp(datetime.now().date())

    dr = result["daily_returns"]
    nv = result["nav"]
    total_ret = float(nv.iloc[-1]) - 1.0 if len(nv) > 0 else np.nan
    ann_ret = (1 + total_ret) ** (252 / max(1, len(dr))) - 1 if len(dr) > 0 else np.nan
    vol = dr.std() * np.sqrt(252) * 100 if len(dr) > 1 else np.nan
    sharpe = (ann_ret - cfg.RISK_FREE_RATE) / (vol / 100) if vol and vol > 0 else np.nan
    max_dd = (nv / nv.cummax() - 1).min() * 100 if len(nv) > 0 else np.nan

    config_summary = [
        ["Selected_Factors", ", ".join(SELECTED_FACTOR_NAMES)],
        ["Composite_Factor", COMPOSITE_FACTOR_SHEET],
        ["Composite_Method", "多元回归beta加权 (M=3月, N=10日)"],
        ["Weight_Method", params.get("weight_method", TARGET_WEIGHT_METHOD)],
        ["Group_Num", params.get("group_num", TARGET_GROUP_NUM)],
        ["Target_Rank", params.get("target_rank", TARGET_RANK)],
        ["---", "---"],
        ["Total_Return", f"{total_ret:.4f}" if not np.isnan(total_ret) else "-"],
        ["Annual_Return", f"{ann_ret:.4f}" if not np.isnan(ann_ret) else "-"],
        ["Annual_Volatility_Pct", f"{vol:.2f}" if not np.isnan(vol) else "-"],
        ["Sharpe_Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "-"],
        ["Max_Drawdown_Pct", f"{max_dd:.2f}" if not np.isnan(max_dd) else "-"],
    ]
    status_rows = [
        ["As_Of_Date", str(as_of.date())],
        ["Is_Rebalance_Today", "是" if status["is_rebalance_today"] else "否"],
        ["Current_Rebalance_Date", str(status["current_rebalance_date"].date()) if status["current_rebalance_date"] else "-"],
        ["Next_Rebalance_Date", str(status["next_rebalance_date"].date()) if status["next_rebalance_date"] else "-"],
        ["---", "---"],
        ["Price_Convention", "Adj Close（收盘价）；T 日收盘执行，买卖均用当日收盘价"],
        ["Rebalance_Period_Days", STRATEGY_PARAMS["rebalance_period"]],
        ["---", "---"],
        *config_summary,
    ]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(status_rows, columns=["Parameter", "Value"]).to_excel(
            writer, sheet_name="Rebalance_Day_Status", index=False
        )

        if not current_ops.empty:
            current_ops.to_excel(writer, sheet_name="Current_Operations", index=False)
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

        df_ops = result["operations_df"]
        if len(df_ops) > 0:
            df_ops.to_excel(writer, sheet_name="All_Operations", index=False)

        df_period = result["period_summary_df"]
        if len(df_period) > 0:
            df_period.to_excel(writer, sheet_name="Period_Summary", index=False)

    print(f"调仓日报表已写入: {output_path}")


# ---------------------------------------------------------------------------
# Discord 推送
# ---------------------------------------------------------------------------

def send_discord_notification(
    status: dict,
    current_ops: pd.DataFrame,
    result: dict,
    webhook_url: str = DISCORD_WEBHOOK_URL,
) -> None:
    """发送 Discord 通知。"""
    if not webhook_url:
        print("未配置 Discord Webhook URL，跳过推送")
        return

    try:
        as_of = pd.Timestamp(datetime.now().date())
        is_rebalance = status["is_rebalance_today"]
        current_rb = status["current_rebalance_date"]
        next_rb = status["next_rebalance_date"]

        # 构建消息
        # 策略基本信息（因子选择、复合方式）
        factor_info = (
            f"**选定因子：** {', '.join(SELECTED_FACTOR_NAMES)}\n"
            f"**复合因子：** {COMPOSITE_FACTOR_SHEET}（多元回归beta加权 M3/N10）\n"
        )

        if is_rebalance:
            title = "🔔 调仓日提醒 - 今日需要操作"
            color = 0x00FF00  # 绿色

            # 性能指标
            dr = result["daily_returns"]
            nv = result["nav"]
            total_ret = float(nv.iloc[-1]) - 1.0 if len(nv) > 0 else 0
            ann_ret = (1 + total_ret) ** (252 / max(1, len(dr))) - 1 if len(dr) > 0 else 0
            sharpe = result.get("sharpe_ratio", 0)

            description = (
                factor_info
                + f"**调仓日期：** {current_rb.date()}\n"
                + f"**策略：** {_strategy_name()}\n"
                f"**执行时间建议：** 美东时间 15:45-16:00（收盘前15分钟）\n\n"
                f"**策略表现：**\n"
                f"• 总收益率：{total_ret:.2%}\n"
                f"• 年化收益率：{ann_ret:.2%}\n"
                f"• 夏普比率：{sharpe:.2f}\n"
            )

            # 操作明细
            if not current_ops.empty:
                sell_ops = current_ops[current_ops["Action"] == "Sell"]
                buy_ops = current_ops[current_ops["Action"] == "Buy"]

                fields = []

                if not sell_ops.empty:
                    sell_text = ""
                    for _, row in sell_ops.iterrows():
                        symbol = row["Symbol"]
                        weight = row.get("Weight", 0) * 100
                        sell_price = row.get("Sell_Price_Close", 0)
                        sell_text += f"• {symbol}: {weight:.1f}% @ ${sell_price:.2f}\n"
                    fields.append({
                        "name": f"🔴 卖出操作 ({len(sell_ops)} 只)",
                        "value": sell_text or "无",
                        "inline": False
                    })

                if not buy_ops.empty:
                    buy_text = ""
                    for _, row in buy_ops.iterrows():
                        symbol = row["Symbol"]
                        weight = row.get("Weight", 0) * 100
                        buy_price = row.get("Buy_Price_Close", 0)
                        buy_text += f"• {symbol}: {weight:.1f}% @ ${buy_price:.2f}\n"
                    fields.append({
                        "name": f"🟢 买入操作 ({len(buy_ops)} 只)",
                        "value": buy_text or "无",
                        "inline": False
                    })
            else:
                fields = [{
                    "name": "⚠️ 操作明细",
                    "value": "无操作数据（可能是数据不足或最后一个调仓日）",
                    "inline": False
                }]

            # 下一调仓日
            if next_rb:
                fields.append({
                    "name": "📅 下一调仓日",
                    "value": f"{next_rb.date()}",
                    "inline": False
                })
        else:
            title = "ℹ️ 非调仓日 - 无需操作"
            color = 0x808080  # 灰色
            description = (
                factor_info
                + f"**当前日期：** {as_of.date()}\n"
                + f"**最近调仓日：** {current_rb.date() if current_rb else '无'}\n"
                + f"**下一调仓日：** {next_rb.date() if next_rb else '未知'}\n\n"
                + "今日无需操作，请等待下一调仓日。"
            )
            fields = []

        # 构建 embed
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {
                "text": f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 因子：{', '.join(SELECTED_FACTOR_NAMES)}"
            }
        }

        payload = {
            "embeds": [embed]
        }

        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ Discord 通知已发送（状态码：{response.status_code}）")

    except Exception as e:
        print(f"❌ Discord 通知发送失败：{e}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(
    skip_pipeline: bool = False,
    skip_pull: bool = False,
    run_dir_arg: str | None = None,
    send_discord: bool = True,
):
    """
    skip_pipeline: 若为 True，跳过 pipeline，从 run_dir_arg 或默认路径读取
    skip_pull: 若 pipeline 运行，是否跳过 pull_data
    run_dir_arg: 指定运行目录。若 skip_pipeline 且未指定，则使用默认项目路径并创建新时间戳目录保存报表
    send_discord: 是否发送 Discord 通知（默认 True）
    """
    run_dir = _get_run_dir(skip_pipeline, run_dir_arg)
    os.makedirs(run_dir, exist_ok=True)

    # 确定数据路径
    if skip_pipeline:
        if run_dir_arg:
            composite_file = os.path.join(run_dir, "composite_factor_reports", "composite_factors.xlsx")
            price_file = os.path.join(run_dir, "data", "us_top100_daily_2023_present.xlsx")
        else:
            composite_file = os.path.join(PROJECT_ROOT, "output", "composite_factor_reports", "composite_factors.xlsx")
            price_file = os.path.join(PROJECT_ROOT, "data", "us_top100_daily_2023_present.xlsx")
    else:
        composite_file = os.path.join(run_dir, "composite_factor_reports", "composite_factors.xlsx")
        price_file = os.path.join(run_dir, "data", "us_top100_daily_2023_present.xlsx")

    print("=" * 64)
    print("  调仓日全流程与报表")
    print(f"  输出目录: {run_dir}")
    print(f"  策略参数: {_strategy_name()} | 价格: Adj Close（收盘价）")
    print("=" * 64)

    if not skip_pipeline:
        print("\n[阶段 1] 执行 Pipeline（输出至上述目录）...")
        run_pipeline_subprocess(run_dir, skip_pull=skip_pull)
    else:
        print("\n[阶段 1] 跳过 Pipeline")

    print("\n[阶段 2] 加载复合因子与收益率...")
    factor_df = load_composite_factor(composite_file, COMPOSITE_FACTOR_SHEET)
    ret_df = load_return_data(price_file, cfg.RETURN_COLUMN)
    ret_df.sort_index(inplace=True)
    price_df = load_price_data(price_file, "Adj Close")

    print(f"\n[阶段 3] 运行策略回测（{_strategy_name()}）...")
    result = run_detailed_backtest(
        factor_df=factor_df,
        ret_df=ret_df,
        price_df=price_df,
        group_num=TARGET_GROUP_NUM,
        target_rank=TARGET_RANK,
        rebalance_period=TARGET_REBALANCE_DAYS,
        weight_method=TARGET_WEIGHT_METHOD,
        config=cfg,
        rebalance_date_offset=REBALANCE_DATE_OFFSET,
    )

    if "error" in result:
        print(f"错误: {result['error']}")
        return

    result["_factor_df"] = factor_df
    result["_ret_df"] = ret_df
    result["_price_df"] = price_df
    result["_config"] = cfg

    rebalance_dates = _select_rebalance_dates(
        factor_df.index,
        TARGET_REBALANCE_DAYS,
        offset_days=REBALANCE_DATE_OFFSET,
    )
    last_factor_date = factor_df.index[-1]
    as_of_date = pd.Timestamp(datetime.now().date())

    status = get_rebalance_day_status(
        rebalance_dates=rebalance_dates,
        rebalance_period=TARGET_REBALANCE_DAYS,
        as_of_date=as_of_date,
        last_factor_date=last_factor_date,
    )

    current_rebalance_date = status.get("current_rebalance_date")
    current_ops = pd.DataFrame()
    if current_rebalance_date is not None:
        current_ops = get_current_rebalance_operations(result, current_rebalance_date)

    output_path = os.path.join(run_dir, "rebalance_day_report.xlsx")
    write_rebalance_day_report(result, status, current_ops, output_path)

    print("\n" + "-" * 64)
    print("策略概要:")
    print(f"  选定因子: {', '.join(SELECTED_FACTOR_NAMES)}")
    print(f"  复合因子: {COMPOSITE_FACTOR_SHEET} (多元回归beta加权 M3/N10)")
    print("调仓日判定:")
    print(f"  今日是否调仓日: {'是' if status['is_rebalance_today'] else '否'}")
    print(f"  当前调仓日: {current_rebalance_date}")
    print(f"  下一调仓日: {status.get('next_rebalance_date')}")
    print(f"  当前调仓操作数: {len(current_ops)} 条")
    print(f"  全部输出目录: {run_dir}")
    print("=" * 64)

    # 发送 Discord 通知
    if send_discord:
        print("\n[阶段 4] 发送 Discord 通知...")
        send_discord_notification(status, current_ops, result)
    else:
        print("\n[阶段 4] 跳过 Discord 通知")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pipeline", action="store_true", help="跳过 pipeline，使用已有数据")
    parser.add_argument("--skip-pull", action="store_true", help="pipeline 中跳过 pull_data")
    parser.add_argument("--run-dir", type=str, default=None, help="指定运行目录（skip-pipeline 时复用该目录数据）")
    parser.add_argument("--no-discord", action="store_true", help="不发送 Discord 通知")
    args = parser.parse_args()
    main(
        skip_pipeline=args.skip_pipeline,
        skip_pull=args.skip_pull,
        run_dir_arg=args.run_dir,
        send_discord=not args.no_discord,
    )
