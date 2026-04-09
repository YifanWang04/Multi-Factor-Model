"""
Yahoo Finance 行情数据下载脚本 (data/pull_yhfinance_Data.py)
=====================================
本脚本通过 yfinance 下载指定股票列表的日频行情（含复权收盘价、成交量等），并写入单个 Excel 文件，每个标的一个 sheet，便于后续因子构建与回测使用。

行为说明：
- 股票列表与时间范围：见 data/data_config.py（YFINANCE_TICKERS、yfinance_pull_start_date、DATA_BASE_START_DATE）。
- end_date 使用运行当日（datetime.today()）。
- DATA_START_OFFSET_DAYS：数据起始日提前的交易日数，0=不提前；正数=提前 N 个交易日，使调仓日历整体前移。
- 输出：offset=0 时为 us_top100_daily_2023_present.xlsx；offset!=0 时为 us_top100_daily_2023_present_offset{N}d.xlsx，避免覆盖原数据。
- 自动补全收盘价：当日线中 Open/High/Low 已有但 Close 缺失（通常为收盘后短期延迟），
  且目标日期已收盘（早于今日），则用 fast_info.last_price 补全 Close/Adj Close。
  注意：盘中运行时不会对当日数据做补全（避免用盘中价冒充收盘价）。
"""

import os
import sys
import time

# 确保项目根目录在 path 中，以便 import data.data_config
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

from data.data_config import (
    DATA_START_OFFSET_DAYS,
    YFINANCE_DOWNLOAD_AUTO_ADJUST,
    YFINANCE_DOWNLOAD_PROGRESS,
    YFINANCE_TICKERS,
    _price_filename,
    yfinance_pull_start_date,
)

# 1. 运行参数（配置均在 data_config）

start_date = yfinance_pull_start_date()
if DATA_START_OFFSET_DAYS > 0:
    print(f"DATA_START_OFFSET_DAYS={DATA_START_OFFSET_DAYS}，起始日提前至 {start_date}")

# yfinance 的 end 为不包含结束日，+1 天以纳入「运行当日」的日线
end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
codes = YFINANCE_TICKERS

# 2. get data

data_dict = {}

total = len(codes)
print(f"开始下载 {total} 只标的...")

for i, code in enumerate(codes, 1):
    df = yf.download(
        code,
        start=start_date,
        end=end_date,
        auto_adjust=YFINANCE_DOWNLOAD_AUTO_ADJUST,
        progress=YFINANCE_DOWNLOAD_PROGRESS,
    )

    if df.empty:
        print(f"  [{i}/{total}] {code} ✗ (无数据)")
        continue

    # 单标的 yf.download 可能返回单级或 MultiIndex 列名
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Ticker"] = code

    data_dict[code] = df
    # print(f"  [{i}/{total}] {code} ✓")

print(f"下载完成，成功获取 {len(data_dict)}/{total} 只")

# ── 美股收盘判断（美东时区）─────────────────────────────────────────────────────
# 美股交易日：周一至周五，节假日除外
# 收盘时间：美东 16:00（对应 UTC 21:00 / 北京次日 04:00）
_MARKET_CLOSE_HOUR_UTC = 21  # 美东 16:00 = UTC 21:00
_MARKET_CLOSE_MINUTE = 0


def _is_target_date_session_closed(target_date: pd.Timestamp) -> bool:
    """
    判断目标日期（交易日）的美股是否已收盘。

    原理：美股在交易日 D 的 21:00 UTC 收盘（DST 期间，ET=UTC-4）。
    D 的收盘一定早于 D+1 的 00:00 UTC。因此：
    若当前 UTC 时间 > target_date + 1 天 00:00 UTC，则该交易日必已收盘。
    若 target_date >= today（今天或未来），则未收盘。
    """
    now_utc = datetime.now(timezone.utc)
    today = pd.Timestamp(datetime.today().date())
    if target_date >= today:
        return False
    # D+1 天 00:00 UTC = 收盘截止判断线
    cutoff_utc = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=0, minute=0, second=0,
        tzinfo=timezone.utc,
    ) + pd.Timedelta(days=1)
    return now_utc > cutoff_utc


def _is_market_closed_now() -> bool:
    """判断美股当前是否已收盘（UTC 时间），用于盘中实时监控。"""
    now_utc = datetime.now(timezone.utc)
    # 美股交易日（周一=0，周日=6）
    if now_utc.weekday() >= 5:  # 周六、周日
        return True
    # 检查是否已过收盘时间
    if now_utc.hour > _MARKET_CLOSE_HOUR_UTC:
        return True
    if now_utc.hour == _MARKET_CLOSE_HOUR_UTC and now_utc.minute >= _MARKET_CLOSE_MINUTE:
        return True
    return False


def _find_missing_close_rows(df: pd.DataFrame, target_date: pd.Timestamp) -> list[int]:
    """
    找出目标日期行中 Open/High/Low 已有但 Close（及 Adj Close）缺失的行索引（相对于 df）。
    返回 df 中缺失 Close 的行的 .index 列表。
    """
    if "Date" not in df.columns:
        return []
    df_dt = df["Date"].copy()
    if not pd.api.types.is_datetime64_any_dtype(df_dt):
        df_dt = pd.to_datetime(df_dt, errors="coerce")
    rows = df[df_dt.dt.normalize() == target_date.normalize()]
    missing = []
    for idx, row in rows.iterrows():
        close_vals = []
        if "Close" in df.columns:
            close_vals.append(row["Close"])
        if "Adj Close" in df.columns:
            close_vals.append(row["Adj Close"])
        if not close_vals or all(pd.isna(v) or v == 0 for v in close_vals):
            # OHL 至少有一个非空
            ohl = [row[c] for c in ["Open", "High", "Low"] if c in df.columns]
            if ohl and not all(pd.isna(v) or v == 0 for v in ohl):
                missing.append(idx)
    return missing


def _backfill_close_fast_info(
    data_dict: dict[str, pd.DataFrame],
    max_retries: int = 3,
    retry_delay_base: float = 0.5,
    retry_delay_mult: float = 2.0,
) -> dict[str, dict[str, float]]:
    """
    对 data_dict 中所有标的，检查目标日期（最新日期）是否有 OHL 但无 Close，
    若无 Close 且当前已过美股收盘时间，则用 fast_info.last_price 补全。
    仅补全"昨天"及更早的日期（避免盘中运行时用实时价冒充收盘价）。

    返回 {ticker: close_price} 的字典（包含所有尝试补全的标的，含失败）。
    """
    result: dict[str, dict[str, float]] = {}
    if not data_dict:
        return result

    # 目标日期：昨天（最近一个已收盘交易日）
    today = pd.Timestamp(datetime.today().date())
    yesterday = today - pd.Timedelta(days=1)
    # 若昨天是周末，往前回溯到最近交易日
    while yesterday.weekday() >= 5:
        yesterday -= pd.Timedelta(days=1)

    # 若目标日期的收盘尚未确认（盘中运行），跳过 backfill
    if not _is_target_date_session_closed(yesterday):
        print(f"  [Backfill 跳过] 目标日期 {yesterday.date()} 的收盘尚未确认（可能仍在盘中），不对其做收盘价补全")
        return result

    print(f"  [Backfill] 美股已收盘，开始检查 {yesterday.date()} 收盘价缺失情况...")

    # 统计缺失 Close 的标的
    need_fetch: list[str] = []
    for ticker, df in data_dict.items():
        missing_idx = _find_missing_close_rows(df, yesterday)
        if missing_idx:
            need_fetch.append(ticker)

    if not need_fetch:
        print(f"  [Backfill] 无收盘价缺失标的，跳过")
        return result

    print(f"  [Backfill] 发现 {len(need_fetch)} 只标的收盘价缺失，正在获取 fast_info.last_price...")

    # 批量获取收盘价
    fetched: dict[str, float] = {}
    for sym in need_fetch:
        delay = retry_delay_base
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(sym)
                fi = ticker.fast_info
                last_p = getattr(fi, "last_price", None)
                if last_p is not None:
                    fetched[sym] = float(last_p)
                    break
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= retry_delay_mult
        if sym not in fetched:
            print(f"    WARNING: {sym} 收盘价获取失败")

    if not fetched:
        print(f"  [Backfill] 所有标的收盘价获取均失败，跳过")
        return result

    # 写入 data_dict（原地修改，to_excel 时一并保存）
    for ticker, close_val in fetched.items():
        df = data_dict[ticker]
        missing_idx = _find_missing_close_rows(df, yesterday)
        for idx in missing_idx:
            if "Adj Close" in df.columns:
                df.at[idx, "Adj Close"] = close_val
            if "Close" in df.columns:
                df.at[idx, "Close"] = close_val
        result[ticker] = close_val

    print(f"  [Backfill] 完成，共补全 {len(result)} 只")
    return result


# ── 收盘价 Backfill（仅已收盘日）─────────────────────────────────────────────
_backfill_close_fast_info(data_dict)

# ── 主流程 ────────────────────────────────────────────────────────────────────

if not data_dict:
    raise RuntimeError("没有成功下载任何股票数据，请检查网络或股票代码")

_run_dir = os.environ.get("REBALANCE_RUN_DIR")
_price_name = _price_filename()
if _run_dir:
    _out_path = os.path.join(_run_dir, "data", _price_name)
    os.makedirs(os.path.dirname(_out_path), exist_ok=True)
else:
    # 必须按当前 offset 写入对应文件名；勿用 PRICE_FILE（offset 文件不存在时会回退到基线路径并覆盖）
    _out_path = os.path.join(_ROOT, "data", _price_filename())
    os.makedirs(os.path.dirname(_out_path), exist_ok=True)

with pd.ExcelWriter(_out_path, engine="xlsxwriter") as writer:
    for sheet_name, df in data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

print(f"Excel 写入完成，共写入 {len(data_dict)} 个 sheet → {_price_name}")

print("Data download completed.")
