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
  - rebalance_day_report.xlsx # 本脚本报表（已合并全部 sheet：Config、Opers、Returns 等）

时序约定（与 README.md 一致）：
  - 交易：T 日收盘执行，买卖价格均使用 Adj Close（收盘价）
  - 调仓日且未收盘时：用当日开盘价（Today_Open）与现价（收盘价估计）替代买入价
  - 持仓区间：(T, T_next]，T 日收益不计入当期持仓
  - 报表/Discord：若下一调仓日尚未到（或卖出价缺失），用 As_Of 日收盘价或实时价作假设卖出价，
    重算 Period_Return / Sell_Value（列 Sell_Price_Source 标明来源）

用法（项目根目录）：
  python analysis/strategy/run_rebalance_day.py
  python analysis/strategy/run_rebalance_day.py --no-discord  # 不发送 Discord 通知
  python analysis/strategy/run_rebalance_day.py --inline       # Pipeline 在同一进程中执行（更快）
"""

from __future__ import annotations

import os
import sys
import io
import subprocess
import time
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# 设置 UTF-8 输出（Windows 兼容）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 路径注册 ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.dirname(os.path.dirname(_HERE))
_SF_DIR = os.path.join(_ANALYSIS, "single_factor")
_MF_DIR = os.path.join(_ANALYSIS, "multi_factor")
_ROOT = os.path.dirname(_ANALYSIS)

for _p in (_HERE, _SF_DIR, _MF_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_strategy import load_composite_factor as _load_composite_factor_from_run_strategy, load_return_data as _load_ret_data
from run_detailed_backtest_report import run_detailed_backtest, parse_strategy_param
from strategy_backtest import _build_groups, _select_rebalance_dates
from portfolio_optimizer import compute_weights
import strategy_config as cfg
from data.data_config import DATA_START_OFFSET_DAYS, _price_filename, COMPOSITE_FACTOR_OUTPUT_DIR, COMPOSITE_FACTOR_FILE as _COMPOSITE_FACTOR_FILE
from strategy_utils import (
    # 价格/复合因子加载
    load_price_data as _load_price_data,
    _get_price_on_date as _get_price_util,
    load_composite_factor_with_fallback,
    load_composite_factor as _load_composite_factor,
    # 因子后缀
    build_factor_suffix,
    composite_factors_path,
    # 操作过滤
    filter_weight_lt,
    # Discord 格式化工具
    format_metric as _fmt_metric,
    truncate_text as _trunc_text,
    # MarkToMarket 重估封装
    MarkToMarket,
    patch_period_summary_from_mtm,
)


# ---------------------------------------------------------------------------
# 全局常量（消除魔法数字）
# ---------------------------------------------------------------------------

# 权重过滤阈值：Weight < 此值时忽略该操作
WEIGHT_FILTER_THRESHOLD: float = 0.0001

# 调仓日外推：最大外推期数
REBALANCE_EXTRAPOLATE_MAX_PERIODS: int = 24
# 调仓日外推：外推到 as_of_date 之后至少多少期后停止
REBALANCE_EXTRAPOLATE_FUTURE_MIN: int = 12

# 实时价格获取：重试次数
LIVE_PRICE_MAX_RETRIES: int = 3
# 实时价格获取：重试间隔（秒），首次重试
LIVE_PRICE_RETRY_DELAY_BASE: float = 0.5
# 实时价格获取：每次重试间隔乘数
LIVE_PRICE_RETRY_DELAY_MULT: float = 2.0
# 实时价格获取：请求超时（秒）
LIVE_PRICE_TIMEOUT: float = 10.0

# Pipeline subprocess：超时时间（秒）
PIPELINE_SUBPROCESS_TIMEOUT: int = 600

# ---------------------------------------------------------------------------
# 优化器回看天数（用于 _compute_last_rebalance_ops）
# ⚠️ 优先从 strategy_config 读取；本地值仅作 fallback，避免魔法数字泄漏
DEFAULT_OPTIMIZATION_LOOKBACK: int = getattr(cfg, "OPTIMIZATION_LOOKBACK", 252)

# Discord embed：单字段最大字符数
DISCORD_FIELD_MAX_CHARS: int = 1024
# Discord embed：操作列表最大行数
DISCORD_OPS_MAX_LINES: int = 20


# ---------------------------------------------------------------------------
# 配置（本脚本独立配置，策略相关参数从 strategy_config 派生）
# ---------------------------------------------------------------------------

PROJECT_ROOT = r"D:\qqq"
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output")

# 复合因子合成方法（Excel sheet 名）
COMPOSITE_FACTOR_SHEET = "ic_m3_N20"

# ── 手动因子配置区 ─────────────────────────────────────────────────────────────
# ⚠️ 如需切换因子，直接修改此列表
MANUALLY_SELECTED_FACTOR_INDICES = [95, 101, 62, 65, 32]  # 3/17
# MANUALLY_SELECTED_FACTOR_INDICES = [95, 24, 64, 65, 32]  # 3/25 备选
# ─────────────────────────────────────────────────────────────────────────────

# 策略参数：整串配置，格式 {weight_method}_{N}G_Top{R}_P{D}d
# STRATEGY_PARAM = "max_return_10G_Top1_P20d"  # 3/25
STRATEGY_PARAM = "max_return_5G_Top1_P10d"  # 3/17

# 选定因子（直接使用手动配置）
SELECTED_FACTOR_INDICES = MANUALLY_SELECTED_FACTOR_INDICES
SELECTED_FACTOR_NAMES = [f"alpha{i:03d}" for i in SELECTED_FACTOR_INDICES]

# 解析后供内部使用
_parsed = parse_strategy_param(STRATEGY_PARAM)
STRATEGY_PARAMS = {
    "weight_method": _parsed[0],
    "group_num": _parsed[1],
    "target_rank": _parsed[2],
    "rebalance_period": _parsed[3],
}


# Discord Webhook URL（优先从环境变量读取，硬编码兜底）
DISCORD_WEBHOOK_URL = os.environ.get(
    "REBALANCE_DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1478641216659652709/TRe7zHYv0x5AbYJMngnJbi1TbjUwXiOhIct-rze0wHFFYgi-Yqt320iGOCY4J1NUbq68",
)


def _composite_factors_path(base_dir: str) -> str:
    """返回 composite_factor_reports 目录下带因子后缀的文件路径。"""
    return composite_factors_path(base_dir, SELECTED_FACTOR_INDICES)


def _get_run_dir(run_dir_arg: Optional[str], skip_pipeline: bool) -> str:
    """获取本次运行的输出目录。"""
    if run_dir_arg:
        return os.path.abspath(run_dir_arg)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.join(OUTPUT_BASE, f"rebalance_day_{ts}")


# ---------------------------------------------------------------------------
# 数据加载（来自 strategy_utils）
# ---------------------------------------------------------------------------
# load_price_data: 见 strategy_utils（从 strategy_utils 导入为 _load_price_data）


# ---------------------------------------------------------------------------
# Pipeline：执行数据与因子流水线
# ---------------------------------------------------------------------------

def _run_pipeline_inline(run_dir: str, skip_pull: bool = False) -> None:
    """在同一进程中依次执行 pipeline 各步骤（避免 subprocess 进程启动开销）。"""
    from pipeline.build_factors import run as run_build_factors
    from pipeline.data_process import run as run_data_process
    from analysis.multi_factor.run_composite_factor import main as run_composite

    data_dir = os.path.join(run_dir, "data")
    factor_raw_dir = os.path.join(run_dir, "factor_raw")
    factor_processed_dir = os.path.join(run_dir, "factor_processed")
    composite_dir = os.path.join(run_dir, "composite_factor_reports")
    for d in (data_dir, factor_raw_dir, factor_processed_dir, composite_dir):
        os.makedirs(d, exist_ok=True)

    if not skip_pull:
        print("[Pipeline] 拉取行情数据...")
        from data import pull_yhfinance_Data
        pull_yhfinance_Data.main()

    print("[Pipeline] 构建因子...")
    run_build_factors()

    print("[Pipeline] 因子数据处理...")
    run_data_process()

    print("[Pipeline] 因子复合...")
    run_composite()


def _run_pipeline_subprocess(run_dir: str, skip_pull: bool = False) -> None:
    """通过 subprocess 依次调用 pipeline 各步骤（stdout/stderr 实时流式打印）。"""
    import shutil
    import subprocess as sp

    env = os.environ.copy()
    env["REBALANCE_RUN_DIR"] = run_dir
    env["REBALANCE_SELECTED_FACTORS"] = ",".join(SELECTED_FACTOR_NAMES)
    env["REBALANCE_SELECTED_FACTOR_INDICES"] = ",".join(str(i) for i in SELECTED_FACTOR_INDICES)
    env["REBALANCE_SELECTED_COMPOSITE"] = COMPOSITE_FACTOR_SHEET

    data_dir = os.path.join(run_dir, "data")
    for sub_dir in (data_dir, os.path.join(run_dir, "factor_raw"),
                    os.path.join(run_dir, "factor_processed"),
                    os.path.join(run_dir, "composite_factor_reports")):
        os.makedirs(sub_dir, exist_ok=True)

    if skip_pull:
        from data.data_config import PRICE_FILE as _src_price, _price_filename
        src = _src_price
        dst = os.path.join(data_dir, _price_filename())
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  已复制数据至: {dst}")
        else:
            raise FileNotFoundError(
                f"skip_pull 时需存在 {src}，请先运行 pull 或确保已在 data/data_config.py 中设好 DATA_START_OFFSET_DAYS"
            )

    steps = []
    if not skip_pull:
        steps.append(("data/pull_yhfinance_Data.py", "拉取行情数据"))
    steps.extend([
        ("pipeline/build_factors.py", "构建因子"),
        ("pipeline/data_process.py", "因子数据处理"),
        ("analysis/multi_factor/run_composite_factor.py", "因子复合"),
    ])

    for i, (script, desc) in enumerate(steps, 1):
        print(f"[Pipeline {i}/{len(steps)}] {desc}...", flush=True)
        proc = sp.Popen(
            [sys.executable, script],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
        )
        # 实时逐行打印子进程输出
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                if line:
                    print(line, end="", flush=True)
        returncode = proc.wait()

        if returncode != 0:
            raise RuntimeError(f"Pipeline 步骤失败 [{returncode}]: {script}")
        print(f"  ✅ {desc} 完成\n", flush=True)


def run_pipeline_subprocess(run_dir: str, skip_pull: bool = False) -> None:
    """通过 subprocess 依次调用 pipeline 各步骤（保留旧接口）。"""
    _run_pipeline_subprocess(run_dir, skip_pull)


# ---------------------------------------------------------------------------
# 调仓日判定与未来调仓日推算
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

    # 从历史最后一个调仓日起，每次 +rebalance_period 个交易日外推
    anchor = rebalance_dates[-1]
    extrapolated = []
    current_date = anchor
    sorted_td = sorted(trading_dates) if trading_dates else []

    for _ in range(REBALANCE_EXTRAPOLATE_MAX_PERIODS):
        if sorted_td:
            # 优先使用实际交易日序列外推（与回测日历完全一致）
            try:
                idx = next(i for i, x in enumerate(sorted_td) if x >= current_date)
            except StopIteration:
                idx = len(sorted_td)
            next_idx = idx + rebalance_period
            if next_idx < len(sorted_td):
                current_date = sorted_td[next_idx]
            else:
                # sorted_td 已用尽：用 pd.bdate_range 工作日兜底
                bdate_range = pd.bdate_range(start=current_date, periods=rebalance_period + 1, freq="B")
                current_date = pd.Timestamp(bdate_range[-1])
        else:
            # 无 trading_dates：用 pd.bdate_range 工作日序列
            bdate_range = pd.bdate_range(start=current_date, periods=rebalance_period + 1, freq="B")
            current_date = pd.Timestamp(bdate_range[-1])
        extrapolated.append(current_date)
        future_so_far = [x for x in extrapolated if x > as_of_date]
        if len(future_so_far) >= REBALANCE_EXTRAPOLATE_FUTURE_MIN:
            break

    # 合并历史 + 外推，去重排序
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


def _get_price_on_date_local(
    price_df: pd.DataFrame,
    date: pd.Timestamp,
    stocks: list,
) -> pd.Series:
    """获取指定日期各标的收盘价，缺失则取最近可交易日。"""
    if date not in price_df.index:
        idx = price_df.index[price_df.index <= date]
        if len(idx) == 0:
            return pd.Series(dtype=float)
        date = idx[-1]
    return price_df.loc[date].reindex(stocks).dropna()


def _get_price_on_date(
    price_df: pd.DataFrame,
    date: pd.Timestamp,
    stocks: list,
) -> pd.Series:
    """兼容性别名：指向 _get_price_on_date_local。"""
    return _get_price_on_date_local(price_df, date, stocks)


def fetch_live_prices(symbols: list[str]) -> dict[str, dict[str, float]]:
    """
    通过 yfinance 获取当日开盘价和现价（带重试机制）。
    返回 {symbol: {"open": float, "current": float}}，失败的标的不包含在结果中。
    异常处理：捕获网络错误（HTTPError）、API 属性变化（AttributeError）等，
    重试失败后打印错误信息（不再静默吞掉）。
    """
    import requests as _req  # requests 在文件顶部已导入，但这里局部导入便于类型细分

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
                # yfinance API 字段名变化（如 last_price → price）
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


def _mark_price_for_symbol(
    price_df: pd.DataFrame,
    symbol: str,
    as_of: pd.Timestamp,
    live_prices: Optional[dict[str, dict[str, float]]],
) -> float:
    """
    取 as_of 及之前最近可用的 Adj Close；若无列或全缺失，则用 live_prices[sym]['current']。
    供 collect_live_prices_for_mtm 使用。
    """
    live_prices = live_prices or {}
    if symbol not in price_df.columns:
        lp = live_prices.get(symbol, {})
        cur = lp.get("current", np.nan)
        return float(cur) if cur is not None else float("nan")
    series = price_df[symbol].dropna()
    if len(series) == 0:
        lp = live_prices.get(symbol, {})
        cur = lp.get("current", np.nan)
        return float(cur) if cur is not None else float("nan")
    valid = series[series.index <= as_of]
    if len(valid) > 0:
        return float(valid.iloc[-1])
    lp = live_prices.get(symbol, {})
    cur = lp.get("current", np.nan)
    return float(cur) if cur is not None else float("nan")


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

    # 取 as_of 及之前的最新收盘价
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

    # 过滤条件：Next_Rebalance_Date 非空且（未到期 OR 卖出价缺失）
    valid_next = ops["Next_Rebalance_Date"].notna()
    open_period = ops["Next_Rebalance_Date"] > as_of
    missing_sell = ops["Sell_Price_Close"].isna()
    need_mask = valid_next & (open_period | missing_sell)
    need_ops = ops[need_mask]

    if need_ops.empty:
        return {}

    # 向量化查本地价格：取 as_of 及之前的最新收盘价
    syms = need_ops["Symbol"].unique().tolist()
    local_prices = _get_price_for_symbols_vectorized(price_df, as_of, syms)
    missing_syms = [s for s in syms if pd.isna(local_prices.get(s))]
    if not missing_syms:
        return {}

    print(f"  MTM：对 {len(missing_syms)} 只标的拉取实时价（本地无 as_of 收盘价）")
    return fetch_live_prices(missing_syms)


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

    # 先插入 Today_Open 列（若不存在）
    if "Today_Open" not in ops.columns:
        ops.insert(ops.columns.get_loc("Symbol") + 1, "Today_Open", np.nan)

    for idx, row in ops.iterrows():
        sym = row["Symbol"]
        if sym not in live_prices:
            continue
        o, c = live_prices[sym]["open"], live_prices[sym]["current"]
        ops.at[idx, "Today_Open"] = o
        # 仅对 Buy 操作更新 Buy_Price_Close；Sell 操作保持原值（调仓日收盘价）
        if row["Action"] == "Buy":
            ops.at[idx, "Buy_Price_Close"] = c
            bv = row.get("Buy_Value", np.nan)
            if not (np.isnan(bv) if isinstance(bv, float) else False) and bv is not None and c > 0:
                ops.at[idx, "Shares"] = float(bv) / c
        # Sell 操作的 Sell_Price_Close 不修改（已在回测中使用调仓日收盘价）

    return ops, True


# ---------------------------------------------------------------------------
# 当前调仓日操作明细
# ---------------------------------------------------------------------------

def get_current_rebalance_operations(
    result: dict,
    current_rebalance_date: pd.Timestamp,
    next_rebalance_date: Optional[pd.Timestamp] = None,
    rebalance_dates: Optional[list] = None,
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

        # 过滤条件：上一期买入，卖出日在 current_rb，且有实际卖出价格
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
    factor_df: Optional[pd.DataFrame],
    ret_df: Optional[pd.DataFrame],
    price_df: Optional[pd.DataFrame],
    rb_date: pd.Timestamp,
    config,
    next_rb_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """对最后一个调仓日（无 next_rb）计算买入操作明细。next_rb_date 为外推的下一调仓日。"""
    if factor_df is None or ret_df is None or price_df is None or config is None:
        print(f"  ⚠️ _compute_last_rebalance_ops：缺少必要数据，跳过")
        return pd.DataFrame()

    group_num = STRATEGY_PARAMS["group_num"]
    target_rank = STRATEGY_PARAMS["target_rank"]
    weight_method = STRATEGY_PARAMS["weight_method"]
    target_group = group_num - (target_rank - 1)
    lookback = getattr(config, "OPTIMIZATION_LOOKBACK", DEFAULT_OPTIMIZATION_LOOKBACK)
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


# ---------------------------------------------------------------------------
# 绩效指标计算
# ---------------------------------------------------------------------------

def compute_extended_metrics(
    daily_returns: pd.Series,
    nav: pd.Series,
    rf_rate: float = 0.02,
) -> dict:
    """
    计算完整绩效指标集，供 Discord 通知使用。
    """
    if daily_returns.empty or nav.empty:
        return {}

    total_ret = float(nav.iloc[-1]) - 1.0 if len(nav) > 0 else float("nan")
    ann_ret = (1 + total_ret) ** (252 / max(1, len(daily_returns))) - 1 if len(daily_returns) > 0 else float("nan")
    vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else float("nan")
    sharpe = (ann_ret - rf_rate) / vol if vol and vol > 0 else float("nan")

    # 最大回撤
    max_dd = float((nav / nav.cummax() - 1).min()) if len(nav) > 0 else float("nan")
    max_dd_pct = max_dd * 100

    # Calmar 比率
    calmar = ann_ret / abs(max_dd) if max_dd and max_dd != 0 else float("nan")

    # 胜率：正收益天数 / 总交易天数
    win_days = int((daily_returns > 0).sum())
    total_days = len(daily_returns)
    win_rate = win_days / total_days if total_days > 0 else float("nan")

    # 盈亏比
    avg_win = float(daily_returns[daily_returns > 0].mean()) if win_days > 0 else 0.0
    loss_days = int((daily_returns < 0).sum())
    avg_loss = float(daily_returns[daily_returns < 0].mean()) if loss_days > 0 else 0.0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("nan")

    return {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "volatility": vol,
        "volatility_pct": vol * 100 if not np.isnan(vol) else float("nan"),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "calmar": calmar,
        "win_rate": win_rate,
        "win_days": win_days,
        "total_days": total_days,
        "profit_loss_ratio": profit_loss_ratio,
    }


# ---------------------------------------------------------------------------
# 写入 Excel 报表
# ---------------------------------------------------------------------------

def _filter_weight_lt(ops: pd.DataFrame, threshold: float = WEIGHT_FILTER_THRESHOLD) -> pd.DataFrame:
    """过滤 Weight 列 < threshold 的行（使用 utils 中的统一实现）。"""
    return filter_weight_lt(ops, threshold, logger=print)


def write_rebalance_day_report(
    result: dict,
    status: dict,
    current_ops: pd.DataFrame,
    output_path: str,
    used_live_prices: bool = False,
    mtm_applied: bool = False,
) -> None:
    """
    写入合并后的调仓日报表（单文件，含全部 sheet）。
    """
    if "error" in result:
        raise ValueError(result["error"])

    params = result.get("params", {})
    as_of = pd.Timestamp(datetime.now().date())

    daily_returns = result["daily_returns"]
    nav = result["nav"]
    metrics = compute_extended_metrics(daily_returns, nav, rf_rate=cfg.RISK_FREE_RATE)

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

    # 统一状态行（消除 config_summary / status_rows 重复）
    status_rows = [
        ["Parameter", "Value"],
        ["As_Of_Date", str(as_of.date())],
        ["Is_Rebalance_Today", "是" if status["is_rebalance_today"] else "否"],
        ["Current_Rebalance_Date", str(status["current_rebalance_date"].date()) if status["current_rebalance_date"] else "-"],
        ["Next_Rebalance_Date", str(status["next_rebalance_date"].date()) if status["next_rebalance_date"] else "-"],
        ["Price_Convention", price_conv],
        ["Rebalance_Period_TradingDays", STRATEGY_PARAMS["rebalance_period"]],
        ["Data_Start_Offset_TradingDays", DATA_START_OFFSET_DAYS],
        ["---", "---"],
        ["Factor_Indices", str(SELECTED_FACTOR_INDICES)],
        ["Selected_Factors", ", ".join(SELECTED_FACTOR_NAMES)],
        ["Composite_Factor", COMPOSITE_FACTOR_SHEET],
        ["Composite_Method", f"IC加权 {COMPOSITE_FACTOR_SHEET} (M=3月, N=20日)"],
        ["Strategy_Param", STRATEGY_PARAM],
        ["Weight_Method", params.get("weight_method", STRATEGY_PARAMS["weight_method"])],
        ["Group_Num", params.get("group_num", STRATEGY_PARAMS["group_num"])],
        ["Target_Rank", params.get("target_rank", STRATEGY_PARAMS["target_rank"])],
        ["---", "---"],
        ["Total_Return", f"{metrics.get('total_return', float('nan')):.4f}" if not np.isnan(metrics.get("total_return", float("nan"))) else "-"],
        ["Annual_Return", f"{metrics.get('annual_return', float('nan')):.4f}" if not np.isnan(metrics.get("annual_return", float("nan"))) else "-"],
        ["Annual_Volatility_Pct", f"{metrics.get('volatility_pct', float('nan')):.2f}" if not np.isnan(metrics.get("volatility_pct", float("nan"))) else "-"],
        ["Sharpe_Ratio", f"{metrics.get('sharpe', float('nan')):.2f}" if not np.isnan(metrics.get("sharpe", float("nan"))) else "-"],
        ["Max_Drawdown_Pct", f"{metrics.get('max_drawdown_pct', float('nan')):.2f}" if not np.isnan(metrics.get("max_drawdown_pct", float("nan"))) else "-"],
        ["Calmar_Ratio", f"{metrics.get('calmar', float('nan')):.2f}" if not np.isnan(metrics.get("calmar", float("nan"))) else "-"],
        ["Win_Rate", f"{metrics.get('win_rate', float('nan')):.2%}" if not np.isnan(metrics.get("win_rate", float("nan"))) else "-"],
        ["Profit_Loss_Ratio", f"{metrics.get('profit_loss_ratio', float('nan')):.2f}" if not np.isnan(metrics.get("profit_loss_ratio", float("nan"))) else "-"],
    ]

    # 过滤低权重操作
    filtered_ops = _filter_weight_lt(current_ops, WEIGHT_FILTER_THRESHOLD)
    df_ops_raw = result["operations_df"]
    df_ops_filtered = _filter_weight_lt(df_ops_raw, WEIGHT_FILTER_THRESHOLD)

    # 将 NaN 替换为 "-" 以便 Excel 显示更友好
    def _nan_to_dash(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 使用 pandas replace 而非逐列 apply，避免类型变化警告
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

        if len(df_ops_filtered) > 0:
            df_ops_filtered.to_excel(writer, sheet_name="All_Operations", index=False)

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


# ---------------------------------------------------------------------------
# Discord 推送
# ---------------------------------------------------------------------------

def _format_metric(value: float, fmt: str, is_pct: bool = False) -> str:
    """安全格式化指标值（使用 utils 中的统一实现，is_pct 参数已废弃）。"""
    return _fmt_metric(value, fmt)


def _truncate_text(text: str, max_chars: int) -> str:
    """截断文本并加省略号（使用 utils 中的统一实现）。"""
    return _trunc_text(text, max_chars)


def _get_holding_period_info(
    operations_df: pd.DataFrame,
    current_rb: pd.Timestamp,
    price_df: pd.DataFrame,
    current_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    获取当前持仓（上一次调仓日买入、至今未卖出的标的）的盈亏情况。
    传入 current_rb（当前调仓日，即这些仓位的买入日），
    返回含 Symbol / Buy_Price / Current_Price / Change_Pct 的 DataFrame。
    若无持仓或数据不足，返回 None。

    容错逻辑：当 current_rb 对应的记录不存在时，
    尝试使用 operations_df 中最新一期的持仓记录。
    """
    if operations_df.empty or "Rebalance_Date" not in operations_df.columns:
        return None

    # 转换日期列为 Timestamp（避免类型不一致导致比较失败）
    ops = operations_df.copy()
    ops["Rebalance_Date"] = pd.to_datetime(ops["Rebalance_Date"], errors="coerce")
    ops["Next_Rebalance_Date"] = pd.to_datetime(ops["Next_Rebalance_Date"], errors="coerce")
    current_rb_ts = pd.Timestamp(current_rb)
    current_date_ts = pd.Timestamp(current_date)

    # 过滤：属于 current_rb 买入的，且至今未卖出（即 Next_Rebalance_Date > current_date）
    holding = ops[
        (ops["Rebalance_Date"] == current_rb_ts)
        & (
            ops["Next_Rebalance_Date"].isna()
            | (ops["Next_Rebalance_Date"] > current_date_ts)
        )
    ].copy()

    # 容错：如果 current_rb 对应的记录为空，尝试使用最新一期（最大 Rebalance_Date）的持仓记录
    if holding.empty:
        # 获取 operations_df 中最新一期（最大 Rebalance_Date）的未卖出持仓
        latest_rb = ops["Rebalance_Date"].max()
        holding = ops[ops["Rebalance_Date"] == latest_rb].copy()

    # 去重：同一股票可能有多条记录（保留权重最大的那条）
    if "Weight" in holding.columns:
        holding = holding.sort_values("Weight", ascending=False)
        holding = holding.drop_duplicates(subset=["Symbol"], keep="first")

    if holding.empty:
        return None

    # 获取 current_date 当日收盘价（缺失则用最近可交易日）
    if current_date_ts in price_df.index:
        current_prices = price_df.loc[current_date_ts]
    else:
        available = price_df.index[price_df.index <= current_date_ts]
        if len(available) == 0:
            return None
        current_prices = price_df.loc[available[-1]]

    holding["Current_Price"] = holding["Symbol"].map(
        lambda s: float(current_prices[s]) if s in current_prices.index else float("nan")
    )
    holding["Buy_Price"] = pd.to_numeric(holding["Buy_Price_Close"], errors="coerce")
    holding["Change_Pct"] = (holding["Current_Price"] - holding["Buy_Price"]) / holding["Buy_Price"]

    holding = holding.dropna(subset=["Buy_Price", "Current_Price", "Change_Pct"])
    if holding.empty:
        return None

    # 返回结果包含 Rebalance_Date（实际使用的调仓日，可能与传入的 current_rb 不同）
    return holding[["Symbol", "Weight", "Buy_Price", "Current_Price", "Change_Pct", "Rebalance_Date"]].sort_values(
        by="Change_Pct", ascending=False
    )


def send_discord_notification(
    status: dict,
    current_ops: pd.DataFrame,
    result: dict,
    webhook_url: str = DISCORD_WEBHOOK_URL,
    used_live_prices: bool = False,
) -> None:
    """发送 Discord 通知（含完整绩效指标 + 持仓盈亏，低权重操作已过滤）。"""
    if not webhook_url:
        print("未配置 Discord Webhook URL，跳过推送")
        return

    try:
        as_of = pd.Timestamp(datetime.now().date())
        is_rebalance = status["is_rebalance_today"]
        current_rb = status["current_rebalance_date"]
        next_rb = status["next_rebalance_date"]

        factor_info = (
            f"**选定因子：** {', '.join(SELECTED_FACTOR_NAMES)}\n"
            f"**复合因子：** {COMPOSITE_FACTOR_SHEET}（IC加权 M3/N20）\n"
            f"**策略参数：** {STRATEGY_PARAM}\n"
            f"**权重方式：** {STRATEGY_PARAMS['weight_method']}　"
            f"**分组数：** {STRATEGY_PARAMS['group_num']}　"
            f"**目标组：** Top{STRATEGY_PARAMS['target_rank']}　"
            f"**调仓周期：** {STRATEGY_PARAMS['rebalance_period']} 交易日　"
            f"**数据起始日偏移：** {DATA_START_OFFSET_DAYS} 交易日\n"
        )

        # 计算绩效指标
        dr = result.get("daily_returns", pd.Series(dtype=float))
        nv = result.get("nav", pd.Series(dtype=float))
        price_df = result.get("_price_df", pd.DataFrame())
        ops_df = result.get("operations_df", pd.DataFrame())
        metrics = compute_extended_metrics(dr, nv, rf_rate=cfg.RISK_FREE_RATE)

        # ── 当前持仓盈亏区块（当期调仓买入的标的至今的盈亏）────
        # 使用 filtered_ops（已过滤低权重）计算持仓盈亏
        # 注意：operations_df 可能缺少最后一期（因无法获取未来卖出价格），
        # 因此直接使用 current_ops 中 Action=="Buy" 的记录
        # 持仓盈亏使用与实际操作一致的阈值（>= 0.0001）
        holding_field: Optional[dict] = None
        if current_rb is not None and not price_df.empty and not current_ops.empty:
            # 先过滤低权重操作（与实际操作一致）
            ops_for_holding = _filter_weight_lt(current_ops, WEIGHT_FILTER_THRESHOLD)

            # 筛选当期买入的标的（用于当前持仓）
            if "Action" in ops_for_holding.columns:
                buy_ops = ops_for_holding[ops_for_holding["Action"] == "Buy"].copy()
            else:
                buy_ops = ops_for_holding.copy()

            if not buy_ops.empty:
                buy_ops = buy_ops.copy()
                # 与 Excel 一致：若已做 MTM，直接用 Period_Return / Sell_Price_Close / Sell_Value
                use_mtm = (
                    "Period_Return" in buy_ops.columns
                    and buy_ops["Period_Return"].notna().any()
                    and "Sell_Price_Close" in buy_ops.columns
                )
                if "Weight" in buy_ops.columns:
                    buy_ops = buy_ops.sort_values("Weight", ascending=False)
                    buy_ops = buy_ops.drop_duplicates(subset=["Symbol"], keep="first")

                if use_mtm:
                    holding_info = buy_ops[
                        buy_ops["Period_Return"].notna()
                        & buy_ops["Buy_Price_Close"].notna()
                        & buy_ops["Sell_Price_Close"].notna()
                    ].copy()
                    if not holding_info.empty:
                        w = pd.to_numeric(holding_info["Weight"], errors="coerce").fillna(0.0)
                        r = pd.to_numeric(holding_info["Period_Return"], errors="coerce").fillna(0.0)
                        total_change = float((w * r).sum())
                        lines = []
                        for _, row in holding_info.iterrows():
                            wt = float(row.get("Weight", 0)) * 100
                            pr = float(row["Period_Return"])
                            bp = float(row["Buy_Price_Close"])
                            sp = float(row["Sell_Price_Close"])
                            sv = pd.to_numeric(row.get("Sell_Value"), errors="coerce")
                            pos_s = f"${float(sv):.2f}" if pd.notna(sv) else "-"
                            lines.append(
                                f"• {row['Symbol']}: 权重 {wt:.1f}% | 区间 {pr*100:+.2f}% | "
                                f"买 ${bp:.2f} → 假设卖 ${sp:.2f} | 头寸 {pos_s}"
                            )
                        if len(lines) > DISCORD_OPS_MAX_LINES:
                            lines = lines[:DISCORD_OPS_MAX_LINES]
                            lines.append(f"  ...另有 {len(holding_info) - DISCORD_OPS_MAX_LINES} 只")
                        holding_text = _truncate_text(
                            "\n".join(lines), DISCORD_FIELD_MAX_CHARS
                        )
                        holding_field = {
                            "name": (
                                f"📊 当前持仓（买入日 {current_rb.date()}，"
                                f"加权区间 {total_change*100:+.2f}%）"
                            ),
                            "value": (
                                f"假设卖出价=As_Of 收盘或现价（与报表 MTM 一致）\n"
                                f"**详情（共 {len(holding_info)} 只）：**\n{holding_text}"
                            ),
                            "inline": False,
                        }
                else:
                    # 回退：用行情表最新价估算
                    buy_ops["Buy_Price"] = pd.to_numeric(buy_ops["Buy_Price_Close"], errors="coerce")

                    def _get_latest_price(sym: str) -> float:
                        if sym not in price_df.columns:
                            return float("nan")
                        series = price_df[sym].dropna()
                        if len(series) == 0:
                            return float("nan")
                        valid = series[series.index <= as_of]
                        if len(valid) > 0:
                            return float(valid.iloc[-1])
                        return float(series.iloc[-1])

                    buy_ops["Current_Price"] = buy_ops["Symbol"].map(_get_latest_price)
                    buy_ops["Change_Pct"] = (
                        buy_ops["Current_Price"] - buy_ops["Buy_Price"]
                    ) / buy_ops["Buy_Price"]

                    holding_info = buy_ops[
                        buy_ops["Buy_Price"].notna() & buy_ops["Current_Price"].notna()
                    ].copy()

                    if not holding_info.empty:
                        total_change = float(
                            (holding_info["Weight"] * holding_info["Change_Pct"]).sum()
                        )

                        lines = []
                        for _, row in holding_info.iterrows():
                            pct = float(row["Change_Pct"]) * 100
                            bp = float(row["Buy_Price"])
                            cp = float(row["Current_Price"])
                            wt = float(row.get("Weight", 0)) * 100
                            pos_value = wt * 100.0
                            lines.append(
                                f"• {row['Symbol']}: {pct:+.2f}% | 权重 {wt:.1f}% | "
                                f"买入 ${bp:.2f} → 当前 ${cp:.2f} | 约 ${pos_value:.0f}"
                            )
                        if len(lines) > DISCORD_OPS_MAX_LINES:
                            lines = lines[:DISCORD_OPS_MAX_LINES]
                            lines.append(f"  ...另有 {len(holding_info) - DISCORD_OPS_MAX_LINES} 只")

                        holding_text = _truncate_text(
                            "\n".join(lines), DISCORD_FIELD_MAX_CHARS
                        )
                        holding_field = {
                            "name": (
                                f"📊 当前持仓（买入日 {current_rb.date()}，"
                                f"区间 {total_change*100:+.2f}%）"
                            ),
                            "value": (
                                f"整体区间涨跌：{total_change*100:+.2f}%\n"
                                f"**详情（共 {len(holding_info)} 只）：**\n{holding_text}"
                            ),
                            "inline": False,
                        }

        # ── 构造 embed 主体 ───────────────────────────────────────────
        if is_rebalance:
            title = "🔔 调仓日提醒 - 今日需要操作"
            color = 0x00FF00  # 绿色

            total_ret = metrics.get("total_return", float("nan"))
            ann_ret = metrics.get("annual_return", float("nan"))
            sharpe = metrics.get("sharpe", float("nan"))
            max_dd_pct = metrics.get("max_drawdown_pct", float("nan"))
            calmar = metrics.get("calmar", float("nan"))
            win_rate = metrics.get("win_rate", float("nan"))
            pl_ratio = metrics.get("profit_loss_ratio", float("nan"))

            description = (
                factor_info
                + f"**调仓日期：** {current_rb.date()}\n"
                + f"**策略：** {STRATEGY_PARAM}\n"
                f"**执行时间建议：** 美东时间 15:45-16:00（收盘前15分钟）\n\n"
                f"**策略表现：**\n"
                f"• 总收益率：{_format_metric(total_ret, '{:.2%}')}\n"
                f"• 年化收益率：{_format_metric(ann_ret, '{:.2%}')}\n"
                f"• 夏普比率：{_format_metric(sharpe, '{:.2f}')}\n"
                f"• 最大回撤：{_format_metric(max_dd_pct, '{:.2f}%')}\n"
                f"• Calmar 比率：{_format_metric(calmar, '{:.2f}')}\n"
                f"• 胜率：{_format_metric(win_rate, '{:.2%}')}\n"
                f"• 盈亏比：{_format_metric(pl_ratio, '{:.2f}')}\n"
            )

            fields: list = []

            # 操作明细（已过滤低权重）
            filtered_ops = _filter_weight_lt(current_ops, WEIGHT_FILTER_THRESHOLD)

            if not filtered_ops.empty:
                sell_ops = filtered_ops[filtered_ops["Action"] == "Sell"]
                buy_ops = filtered_ops[filtered_ops["Action"] == "Buy"].sort_values(
                    by="Weight", ascending=False
                )

                if not sell_ops.empty:
                    lines = []
                    for _, row in sell_ops.iterrows():
                        sym = row["Symbol"]
                        weight = row.get("Weight", 0) * 100
                        sell_price = row.get("Sell_Price_Close", 0)
                        pr = row.get("Period_Return", np.nan)
                        sv = row.get("Sell_Value", np.nan)
                        pr_s = f" 区间 {float(pr)*100:+.2f}%" if pd.notna(pr) else ""
                        sv_s = f" | 卖出头寸 ${float(sv):.2f}" if pd.notna(sv) else ""
                        lines.append(
                            f"• {sym}: 权重 {weight:.1f}% | 卖价 ${float(sell_price):.2f}{pr_s}{sv_s}"
                        )
                    sell_text = _truncate_text(
                        "\n".join(lines), DISCORD_FIELD_MAX_CHARS
                    )
                    fields.append({
                        "name": f"🔴 卖出操作 ({len(sell_ops)} 只)",
                        "value": sell_text,
                        "inline": False,
                    })

                if not buy_ops.empty:
                    lines = []
                    for _, row in buy_ops.iterrows():
                        sym = row["Symbol"]
                        weight = row.get("Weight", 0) * 100
                        buy_price = row.get("Buy_Price_Close", 0)
                        sp = row.get("Sell_Price_Close", np.nan)
                        pr = row.get("Period_Return", np.nan)
                        sv = row.get("Sell_Value", np.nan)
                        mtm = ""
                        if pd.notna(sp) and pd.notna(pr):
                            mtm = f" | 假设卖 ${float(sp):.2f} | 区间 {float(pr)*100:+.2f}%"
                        sv_s = f" | 头寸 ${float(sv):.2f}" if pd.notna(sv) else ""
                        lines.append(
                            f"• {sym}: 权重 {weight:.1f}% | 买 ${float(buy_price):.2f}{mtm}{sv_s}"
                        )
                    buy_text = _truncate_text(
                        "\n".join(lines[:DISCORD_OPS_MAX_LINES]),
                        DISCORD_FIELD_MAX_CHARS,
                    )
                    fields.append({
                        "name": f"🟢 买入操作 ({len(buy_ops)} 只)",
                        "value": buy_text,
                        "inline": False,
                    })
            else:
                fields.append({
                    "name": "⚠️ 操作明细",
                    "value": "无操作数据（可能是数据不足或最后一个调仓日）",
                    "inline": False,
                })

            if next_rb:
                fields.append({
                    "name": "📅 下一调仓日",
                    "value": str(next_rb.date()),
                    "inline": False,
                })

            # 当前持仓盈亏
            if holding_field:
                fields.append(holding_field)

        else:
            title = "ℹ️ 非调仓日 - 无需操作"
            color = 0x808080  # 灰色
            description = (
                factor_info
                + f"**当前日期：** {as_of.date()}\n"
                + f"**最近调仓日：** {current_rb.date() if current_rb else '无'}\n"
                + f"**下一调仓日：** {next_rb.date() if next_rb else '未知'}\n\n"
            )
            fields = []
            if holding_field:
                fields.append(holding_field)
            description += "\n今日无需操作，请等待下一调仓日。"

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {
                "text": (
                    f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"因子：{', '.join(SELECTED_FACTOR_NAMES)}"
                ),
            },
        }

        payload = {"embeds": [embed]}
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ Discord 通知已发送（状态码：{response.status_code}）")

    except Exception as e:
        print(f"❌ Discord 通知发送失败：{e}")


# ---------------------------------------------------------------------------
# 同步复合因子至标准路径
# ---------------------------------------------------------------------------

def _sync_composite_factor_to_standard(run_dir: str, sheet: str) -> None:
    """
    将 Pipeline 生成的复合因子同步到标准路径，
    使 run_detailed_backtest_report.py 使用最新数据。

    原子写保护：先写临时文件，再 os.replace() 原子替换，
    避免写入过程中崩溃导致目标文件损坏。
    """
    import openpyxl

    from data.data_config import COMPOSITE_FACTOR_OUTPUT_DIR

    src = _composite_factors_path(run_dir)
    dst = _composite_factors_path(COMPOSITE_FACTOR_OUTPUT_DIR)

    if not os.path.isfile(src):
        print(f"  [同步跳过] 源文件不存在: {src}")
        return

    try:
        src_df = pd.read_excel(src, sheet_name=sheet, index_col=0)
        src_df.index = pd.to_datetime(src_df.index)
        src_df = src_df.apply(pd.to_numeric, errors="coerce")
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # 原子写：先写入临时文件，再 os.replace() 原子替换
        tmp_path = dst + ".tmp"
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            src_df.to_excel(writer, sheet_name=sheet)

        os.replace(tmp_path, dst)  # 原子替换

        print(f"  [同步完成] 复合因子 {sheet} 已更新至: {dst}")
        print(f"             因子日期范围: {src_df.index[0].date()} ~ {src_df.index[-1].date()}")
    except Exception as e:
        print(f"  [同步警告] 同步复合因子失败（不影响本次调仓报表）: {e}")


# ---------------------------------------------------------------------------
# 加载复合因子（含回退逻辑）
# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(
    skip_pipeline: bool = False,
    skip_pull: bool = False,
    run_dir_arg: Optional[str] = None,
    send_discord: bool = True,
    inline_pipeline: bool = False,
) -> None:
    """
    skip_pipeline: 若为 True，跳过 pipeline，从 run_dir_arg 或默认路径读取。
    skip_pull: pipeline 中是否跳过 pull_data。
    run_dir_arg: 指定运行目录。
    send_discord: 是否发送 Discord 通知。
    inline_pipeline: 若为 True，pipeline 在同一进程中执行（更快）。
    """
    run_dir = _get_run_dir(run_dir_arg, skip_pipeline)
    os.makedirs(run_dir, exist_ok=True)

    if skip_pipeline:
        if run_dir_arg:
            composite_file = _composite_factors_path(run_dir)
            price_file = os.path.join(run_dir, "data", _price_filename())
        else:
            from data.data_config import COMPOSITE_FACTOR_FILE, PRICE_FILE
            composite_file = COMPOSITE_FACTOR_FILE
            price_file = PRICE_FILE
    else:
        composite_file = _composite_factors_path(run_dir)
        price_file = os.path.join(run_dir, "data", _price_filename())

    print("=" * 64)
    print("  调仓日全流程与报表")
    print(f"  输出目录: {run_dir}")
    print(f"  策略参数: {STRATEGY_PARAM} | 价格: Adj Close（收盘价）")
    print(f"  Pipeline 模式: {'内联（inline）' if inline_pipeline else '子进程（subprocess）'}")
    print("=" * 64)

    if not skip_pipeline:
        print("\n[阶段 1] 执行 Pipeline...")
        if inline_pipeline:
            _run_pipeline_inline(run_dir, skip_pull=skip_pull)
        else:
            _run_pipeline_subprocess(run_dir, skip_pull=skip_pull)
        _sync_composite_factor_to_standard(run_dir=run_dir, sheet=COMPOSITE_FACTOR_SHEET)
    else:
        print("\n[阶段 1] 跳过 Pipeline")

    print("\n[阶段 2] 加载复合因子与收益率...")
    factor_df = load_composite_factor_with_fallback(composite_file, COMPOSITE_FACTOR_SHEET, _COMPOSITE_FACTOR_FILE)
    ret_df = _load_ret_data(price_file, cfg.RETURN_COLUMN)
    ret_df.sort_index(inplace=True)
    price_df = _load_price_data(price_file, "Adj Close")

    print(f"\n[阶段 3] 运行策略回测（{STRATEGY_PARAM}）...")
    result = run_detailed_backtest(
        factor_df=factor_df,
        ret_df=ret_df,
        price_df=price_df,
        group_num=STRATEGY_PARAMS["group_num"],
        target_rank=STRATEGY_PARAMS["target_rank"],
        rebalance_period=STRATEGY_PARAMS["rebalance_period"],
        weight_method=STRATEGY_PARAMS["weight_method"],
        config=cfg,
    )

    if "error" in result:
        print(f"错误: {result['error']}")
        return

    result["_factor_df"] = factor_df
    result["_ret_df"] = ret_df
    result["_price_df"] = price_df
    result["_config"] = cfg

    as_of_date = pd.Timestamp(datetime.now().date())

    # 未到期持仓按市值计价（MTM Round 1）：
    # 使用 MarkToMarket 统一封装，重算 Period_Return / Sell_Value / Shares，同步 period_summary_df
    mtm_live = collect_live_prices_for_mtm(result["operations_df"], price_df, as_of_date)
    mtm = MarkToMarket(result["operations_df"], price_df, as_of_date)
    mtm.apply(live_prices=mtm_live)
    result["operations_df"] = mtm.operations_df
    patch_period_summary_from_mtm(result, mtm.operations_df, as_of_date)
    mtm_applied = mtm.was_applied()

    rebalance_dates = _select_rebalance_dates(
        factor_df.index,
        ret_df.index,
        STRATEGY_PARAMS["rebalance_period"],
    )
    last_factor_date = factor_df.index[-1]

    status = get_rebalance_day_status(
        rebalance_dates=rebalance_dates,
        rebalance_period=STRATEGY_PARAMS["rebalance_period"],
        as_of_date=as_of_date,
        last_factor_date=last_factor_date,
        trading_dates=ret_df.index.tolist(),
    )

    current_rb_date = status.get("current_rebalance_date")
    next_rb_date = status.get("next_rebalance_date")
    current_ops = pd.DataFrame()
    used_live_prices = False
    # mtm_applied 已在上方根据 operations_df 初算；Current_Operations 可能再置 True

    if current_rb_date is not None:
        current_ops = get_current_rebalance_operations(
            result, current_rb_date, next_rebalance_date=next_rb_date,
            rebalance_dates=rebalance_dates,
        )
        current_ops, used_live_prices = apply_live_prices_to_operations(
            current_ops, price_df, current_rb_date, as_of_date
        )
        # 调仓日盘中更新买入价后，对 Current_Operations 再跑一次 MTM（MTM Round 2）
        if not current_ops.empty:
            live_co = collect_live_prices_for_mtm(current_ops, price_df, as_of_date)
            mtm_co = MarkToMarket(current_ops, price_df, as_of_date)
            mtm_co.apply(live_prices=live_co)
            current_ops = mtm_co.operations_df
            # 同步更新 period_summary_df（用 MTM Round 2 的 current_ops）
            patch_period_summary_from_mtm(result, current_ops, as_of_date)
            mtm_applied = mtm_applied or mtm_co.was_applied()

    output_path = os.path.join(run_dir, "rebalance_day_report.xlsx")
    write_rebalance_day_report(
        result,
        status,
        current_ops,
        output_path,
        used_live_prices=used_live_prices,
        mtm_applied=mtm_applied,
    )

    print("\n" + "-" * 64)
    print("策略概要:")
    print(f"  选定因子: {', '.join(SELECTED_FACTOR_NAMES)}")
    print(f"  复合因子: {COMPOSITE_FACTOR_SHEET} (IC加权 M3/N20)")
    print(f"  策略参数: {STRATEGY_PARAM}")
    print(f"    权重方式: {STRATEGY_PARAMS['weight_method']}")
    print(f"    分组数:   {STRATEGY_PARAMS['group_num']}")
    print(f"    目标组:   Top{STRATEGY_PARAMS['target_rank']}")
    print(f"    调仓周期: {STRATEGY_PARAMS['rebalance_period']} 交易日")
    print(f"    数据起始日偏移: {DATA_START_OFFSET_DAYS} 交易日")
    print("调仓日判定:")
    print(f"  今日是否调仓日: {'是' if status['is_rebalance_today'] else '否'}")
    print(f"  当前调仓日: {current_rb_date}")
    print(f"  下一调仓日: {status.get('next_rebalance_date')}")
    print(f"  当前调仓操作数: {len(current_ops)} 条")
    print(f"  全部输出目录: {run_dir}")
    print("=" * 64)

    if send_discord:
        print("\n[阶段 4] 发送 Discord 通知...")
        send_discord_notification(
            status, current_ops, result, used_live_prices=used_live_prices
        )
    else:
        print("\n[阶段 4] 跳过 Discord 通知")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="调仓日全流程与报表")
    parser.add_argument("--skip-pipeline", action="store_true", help="跳过 pipeline，使用已有数据")
    parser.add_argument("--skip-pull", action="store_true", help="pipeline 中跳过 pull_data")
    parser.add_argument("--run-dir", type=str, default=None, help="指定运行目录")
    parser.add_argument("--no-discord", action="store_true", help="不发送 Discord 通知")
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Pipeline 在同一进程中执行（更快，skip-pipeline 时无效）",
    )
    args = parser.parse_args()

    main(
        skip_pipeline=args.skip_pipeline,
        skip_pull=args.skip_pull,
        run_dir_arg=args.run_dir,
        send_discord=not args.no_discord,
        inline_pipeline=args.inline,
    )
