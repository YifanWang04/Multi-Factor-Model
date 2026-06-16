"""
Yahoo Finance 行情数据下载脚本 (data/pull_yhfinance_Data.py)
===========================================================
通过 yfinance 下载指定股票列表的日频行情，写入单个 Excel 文件。

- yfinance 使用 auto_adjust=False，保留原始 OHLC/Close/Adj Close。
- 若存在 Adj Close 与 Close，则用 Adj Close / Close 派生 Adj Open/Adj High/Adj Low，
  后续因子构建优先使用复权 OHLC，避免拆股附近 OHLC 与 Adj Close 混用。
- 历史收盘价缺失时，仅尝试重新拉取目标日期的已完成日线；不再用 fast_info.last_price
  写入历史 close，避免用实时价污染历史数据。
- 导入本模块无副作用；只有调用 main() 或直接运行脚本才会下载和写文件。
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from collections.abc import Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import yfinance as yf

from data.data_config import (
    YFINANCE_DOWNLOAD_AUTO_ADJUST,
    YFINANCE_DOWNLOAD_PROGRESS,
    _price_filename,
    _resolve_offset,
    resolve_ticker_universe_source,
    resolve_yfinance_tickers,
    yfinance_pull_start_date,
)


def _normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """单标的 yf.download 可能返回 MultiIndex；统一成普通列名。"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _add_adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    用 Adj Close / Close 的复权比例派生 Adj Open/High/Low。
    若缺少必要列或 Close 非正，则保持 NaN，不猜测。
    """
    required = {"Open", "High", "Low", "Close", "Adj Close"}
    if not required.issubset(df.columns):
        return df

    close = pd.to_numeric(df["Close"], errors="coerce")
    adj_close = pd.to_numeric(df["Adj Close"], errors="coerce")
    ratio = adj_close.div(close.where(close > 0))
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA)

    for src, dst in (("Open", "Adj Open"), ("High", "Adj High"), ("Low", "Adj Low")):
        df[dst] = pd.to_numeric(df[src], errors="coerce") * ratio
    return df


def _is_target_date_session_closed(target_date: pd.Timestamp) -> bool:
    """
    判断目标日期是否已确认收盘。
    仅用于历史 bar 回补，今天或未来一律不回补。
    """
    now_utc = datetime.now(timezone.utc)
    today = pd.Timestamp(datetime.today().date())
    if target_date >= today:
        return False
    cutoff_utc = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=0,
        minute=0,
        second=0,
        tzinfo=timezone.utc,
    ) + pd.Timedelta(days=1)
    return now_utc > cutoff_utc


def _latest_completed_business_date() -> pd.Timestamp:
    """返回最近一个已完成的普通工作日；yfinance 回补只作为缺失 close 的保守兜底。"""
    target = pd.Timestamp(datetime.today().date()) - pd.Timedelta(days=1)
    while target.weekday() >= 5:
        target -= pd.Timedelta(days=1)
    return target


def _find_missing_close_rows(df: pd.DataFrame, target_date: pd.Timestamp) -> list[int]:
    """查找目标日期 OHL 有值但 Close/Adj Close 缺失的行。"""
    if "Date" not in df.columns:
        return []
    df_dt = pd.to_datetime(df["Date"], errors="coerce")
    rows = df[df_dt.dt.normalize() == target_date.normalize()]
    missing: list[int] = []
    for idx, row in rows.iterrows():
        close_vals = [row[c] for c in ("Close", "Adj Close") if c in df.columns]
        close_missing = not close_vals or all(pd.isna(v) or v == 0 for v in close_vals)
        ohl_vals = [row[c] for c in ("Open", "High", "Low") if c in df.columns]
        has_ohl = bool(ohl_vals) and not all(pd.isna(v) or v == 0 for v in ohl_vals)
        if close_missing and has_ohl:
            missing.append(idx)
    return missing


def _fetch_completed_daily_bar(symbol: str, target_date: pd.Timestamp) -> pd.Series | None:
    """重新拉取目标日期的已完成日线。失败或缺目标日时返回 None。"""
    start = target_date.strftime("%Y-%m-%d")
    end = (target_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        hist = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=YFINANCE_DOWNLOAD_AUTO_ADJUST,
            progress=False,
        )
    except Exception:
        return None
    if hist.empty:
        return None
    hist = _normalize_yfinance_columns(hist)
    row = hist.iloc[-1].copy()
    return row


def _backfill_completed_close_bars(
    data_dict: dict[str, pd.DataFrame],
    max_retries: int = 3,
    retry_delay_base: float = 0.5,
    retry_delay_mult: float = 2.0,
) -> dict[str, float]:
    """
    仅对已完成历史交易日缺失 close 的行重新拉取日线并写入。
    不使用 fast_info.last_price 写历史，避免实时价污染历史 close。
    """
    result: dict[str, float] = {}
    if not data_dict:
        return result

    target_date = _latest_completed_business_date()
    if not _is_target_date_session_closed(target_date):
        print(f"  [Backfill 跳过] {target_date.date()} 收盘尚未确认，不写入历史 close")
        return result

    need_fetch = [
        ticker
        for ticker, df in data_dict.items()
        if _find_missing_close_rows(df, target_date)
    ]
    if not need_fetch:
        print("  [Backfill] 无收盘价缺失标的，跳过")
        return result

    print(f"  [Backfill] 发现 {len(need_fetch)} 只标的缺失 {target_date.date()} close，重新拉取完成日线")
    for sym in need_fetch:
        delay = retry_delay_base
        row = None
        for attempt in range(max_retries):
            row = _fetch_completed_daily_bar(sym, target_date)
            if row is not None:
                break
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= retry_delay_mult
        if row is None:
            print(f"    WARNING: {sym} 完成日线回补失败，未写入实时价")
            continue

        df = data_dict[sym]
        missing_idx = _find_missing_close_rows(df, target_date)
        for idx in missing_idx:
            for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
                if col in df.columns and col in row.index and pd.notna(row[col]):
                    df.at[idx, col] = row[col]
        data_dict[sym] = _add_adjusted_ohlc(df)
        if "Adj Close" in row.index and pd.notna(row["Adj Close"]):
            result[sym] = float(row["Adj Close"])
        elif "Close" in row.index and pd.notna(row["Close"]):
            result[sym] = float(row["Close"])

    print(f"  [Backfill] 完成，共补全 {len(result)} 只；失败项未使用实时价污染历史文件")
    return result


def _download_one_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=YFINANCE_DOWNLOAD_AUTO_ADJUST,
        progress=YFINANCE_DOWNLOAD_PROGRESS,
    )
    if df.empty:
        return df
    df = _normalize_yfinance_columns(df)
    df.reset_index(inplace=True)
    df["Ticker"] = symbol
    return _add_adjusted_ohlc(df)


def main(
    ticker_universe: str | None = None,
    tickers: Sequence[str] | None = None,
    ticker_source: str | None = None,
) -> str:
    """下载行情并写入 Excel，返回输出文件路径。"""
    offset = _resolve_offset()
    start_date = yfinance_pull_start_date()
    if offset > 0:
        print(f"DATA_START_OFFSET_DAYS={offset}，NYSE 起始日提前至 {start_date}")

    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    if tickers is not None:
        codes = list(tickers)
        universe_name = ticker_universe or "explicit_tickers"
        source = ticker_source or "argument:tickers"
    else:
        universe_name, source = resolve_ticker_universe_source(ticker_universe)
        if ticker_source is not None:
            source = ticker_source
        codes = resolve_yfinance_tickers(universe_name)
    data_dict: dict[str, pd.DataFrame] = {}

    print(
        "YFinance ticker universe: "
        f"{universe_name} | source={source} | tickers={len(codes)}"
    )
    print(f"开始下载 {len(codes)} 只标的...")
    for i, code in enumerate(codes, 1):
        df = _download_one_symbol(code, start_date, end_date)
        if df.empty:
            print(f"  [{i}/{len(codes)}] {code} ✗ (无数据)")
            continue
        data_dict[code] = df

    print(f"下载完成，成功获取 {len(data_dict)}/{len(codes)} 只")
    _backfill_completed_close_bars(data_dict)

    if not data_dict:
        raise RuntimeError("没有成功下载任何股票数据，请检查网络或股票代码")

    run_dir = os.environ.get("REBALANCE_RUN_DIR")
    price_name = _price_filename()
    if run_dir:
        out_path = os.path.join(run_dir, "data", price_name)
    else:
        out_path = os.path.join(_ROOT, "data", price_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Excel 写入完成，共写入 {len(data_dict)} 个 sheet → {price_name}")
    print("Data download completed.")
    return out_path


if __name__ == "__main__":
    main()
